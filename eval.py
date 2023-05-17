import argparse
import torch
import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'external_cython'))
from yacs.config import CfgNode as CN
import json
import pickle
import numpy as np
from utils.func_lab import print_info, get_from_mapping, trans_3D_2D
from utils import structs
from modeling.my_class import TNT3D
from dataloader.argo_loader import make_dataloader
from optimization.utils_optimize import select_goals_by_optimization
from dataloader.argo_api import post_eval
from utils.calibration import Calibration
from utils.calibration_ns import Calibration_NS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model_path', type=str, default='')
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--eval_device', default='cuda:0', type=str)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--core_num', default=1, type=int)

    test_args = parser.parse_args()

    # load args
    checkpoint_path = test_args.load_model_path

    args_path = os.path.split(checkpoint_path)[0]
    save_dir = args_path
    with open(os.path.join(args_path, 'config.json')) as f:
        cfg = json.load(f)
    cfg = CN(cfg)

    if torch.cuda.is_available():
        device = test_args.eval_device
    else:
        device = 'cpu'

    # 调整cfg
    cfg.DATA.batch_size = test_args.eval_batch_size
    cfg.DATALOADER.shuffle = False
    cfg.EVAL.core_num = test_args.core_num

    with torch.no_grad():
        test(test_args, cfg, save_dir, device, checkpoint_path)


def test(test_args, cfg, save_dir, device, checkpoint_path):
    save_eval_dir = os.path.join(save_dir, 'eval_results')
    if not os.path.exists(save_eval_dir):
        os.makedirs(save_eval_dir)

    print('preparing test dataloader')
    test_dataloader = make_dataloader(cfg, 'val')

    tnt3d = TNT3D(cfg, device, cfg.modality).to(device)

    if not os.path.exists(checkpoint_path):
        print('fail loading from {}'.format(checkpoint_path))
    else:
        if cfg.distributed == 1:
            checkpoint = torch.load(checkpoint_path, map_location={cfg.device: device})
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)

        tnt3d.load_state_dict(checkpoint['model'])

        print('loading from {}'.format(checkpoint_path))

    # if (cfg.modality == 'both') and (test_args.eval_modality is not None):
    #     cfg.modality = test_args.eval_modality
    #     tnt3d.m = cfg.modality
    #     tnt3d.model_dict['scores_net'].m = cfg.modality

    tnt3d.eval()
    infer = True
    if cfg.modality == 'both':
        modal2FDEs = {'bev': [], 'fpv': []}  # 存放所有数据的FDE
        file2pred = {'bev': {}, 'fpv': {}}
        file2labels = {'bev': {}, 'fpv': {}}
        DEs = {'bev': [], 'fpv': []}
    else:
        modal2FDEs = {cfg.modality: []}
        file2pred = {cfg.modality: {}}
        file2labels = {cfg.modality: {}}
        DEs = {cfg.modality: []}

    argo_pred = structs.ArgoPred()
    vis_dicts = []

    for step, batch in enumerate(test_dataloader):
        if (test_args.vis) and (step > 50):
            break
        loss, FDE_dict, pred_trajs_dict, pred_probs_dict = tnt3d(batch, infer)  # ?, list[bs]=int, np=[bs,k,30,2], np=[bs,k]
        batch_size = len(batch)

        if test_args.vis:
            save_vis_result(test_args, cfg, batch_size, vis_dicts, batch, pred_probs_dict)  # pred_trajs_dict里面是nostd的结果

        # bev
        if (cfg.modality == 'both') or (cfg.modality == 'bev'):
            m = 'bev'
            mapping = get_from_mapping(batch, m)
            pred_trajs = pred_trajs_dict[m]
            pred_probs = pred_probs_dict[m]
            for i in range(batch_size):
                assert pred_trajs[i].shape == (6, cfg.DATA.pred_len, 2)
                assert pred_probs[i].shape == (6,)
                argo_pred[mapping[i]['file_name']] = structs.MultiScoredTrajectory(pred_probs[i].copy(), pred_trajs[i].copy())

            modal2FDEs[m] += FDE_dict[m]  # 加入一个batch的list的FDE

            eval_instance_argoverse(cfg, batch_size, cfg.DATA.pred_len, pred_trajs, mapping, file2pred[m], file2labels[m], DEs[m])

        if (cfg.modality == 'both') or (cfg.modality == 'fpv'):
            m = 'fpv'
            mapping = get_from_mapping(batch, m)
            pred_trajs = pred_trajs_dict[m]
            pred_probs = pred_probs_dict[m]
            for i in range(batch_size):
                assert pred_trajs[i].shape == (6, cfg.DATA.pred_len, 2)
                assert pred_probs[i].shape == (6,)
                argo_pred[mapping[i]['file_name']] = structs.MultiScoredTrajectory(pred_probs[i].copy(), pred_trajs[i].copy())

            modal2FDEs[m] += FDE_dict[m]  # 加入一个batch的list的FDE

            eval_instance_argoverse(cfg, batch_size, cfg.DATA.pred_len, pred_trajs, mapping, file2pred[m], file2labels[m], DEs[m])

        print('[{}/{}]'.format(step + 1, len(test_dataloader)))

    if cfg.EVAL.optimization and (cfg.EVAL.core_num > 1):
        select_goals_by_optimization(cfg, None, None, None, close=True)
        pass

    if not test_args.vis:
        post_eval(cfg, file2pred, file2labels, DEs, modal2FDEs, save_eval_dir)

    if len(vis_dicts) != 0:
        with open(os.path.join(save_eval_dir, 'vis_dicts.pkl'), 'wb') as f:
            pickle.dump(vis_dicts, f)


def eval_instance_argoverse(cfg, batch_size, pred_len, pred, mapping, file2pred, file2labels, DEs):
    for i in range(batch_size):
        a_pred = pred[i]
        assert a_pred.shape == (6, pred_len, 2)
        if 'ns' in cfg.save_tag:
            file_name_int = int(mapping[i]['file_name'][:8])
        else:
            file_name_int = int(os.path.split(mapping[i]['file_name'])[1][:-4])
        file2pred[file_name_int] = a_pred
        file2labels[file_name_int] = mapping[i]['origin_labels']

    DE = np.zeros([batch_size, pred_len])
    for i in range(batch_size):
        origin_labels = mapping[i]['origin_labels']
        for j in range(pred_len):
            DE[i][j] = np.sqrt((origin_labels[j][0] - pred[i, 0, j, 0]) ** 2 + (
                    origin_labels[j][1] - pred[i, 0, j, 1]) ** 2)
    DEs.append(DE)


def save_vis_result(test_args, cfg, batch_size, vis_dicts, batch, pred_probs_dict):
    if 'ns' not in cfg.save_tag:  # argo
        with open(cfg.DATALOADER.calib_path, 'r') as f:
            calib_file = json.load(f)
        camera_data = calib_file['camera_data_'][5]
        assert 'ring_front_center' in camera_data['key']
        calib = Calibration(camera_data)
        calib.fpv_type = cfg.fpv_type
    else:  # nuscenes
        with open(cfg.DATALOADER.calib_path, 'r') as f:
            intrinsic = json.load(f)
        intrinsic = np.concatenate([np.array(intrinsic), np.zeros((3, 1))], axis=-1)
        calib = Calibration_NS(intrinsic)

    mapping_b = get_from_mapping(batch, 'bev')
    mapping_f = get_from_mapping(batch, 'fpv')
    for i in range(batch_size):
        per_dict = {
            'file_name': None,
            'bev': {'gt_traj': None, 'pred_traj': None, 'pred_prob': None, 'goals_2D': None, 'scores': None,
                    'agents': None, 'lanes': None},
            'fpv': {'gt_traj': None, 'pred_traj': None, 'pred_prob': None, 'goals_2D': None, 'scores': None,
                    'agents': None, 'lanes': None}
        }

        file_name = mapping_b[i]['file_name']
        goals_3D = mapping_b[i]['vis.goals_3D']  # np[d_goals,3]
        scores = mapping_b[i]['vis.scores']  # np[d_goals]

        per_dict['file_name'] = file_name

        if (cfg.modality == 'both') or (cfg.modality == 'bev'):
            per_dict['bev']['gt_traj'] = mapping_b[i]['labels'].reshape(-1, 2)  # np[30,2]
            per_dict['bev']['pred_traj'] = mapping_b[i]['vis.predict_trajs']  # np[6,30,2]
            per_dict['bev']['pred_prob'] = pred_probs_dict['bev'][i]  # np[6]
            goals_2D, masks = trans_3D_2D(calib, i, mapping_b, goals_3D, 'bev')  # np[d_goals,2]
            goals_2D[masks, :] = np.nan  # mask掉的用nan替换
            per_dict['bev']['goals_2D'] = goals_2D
            per_dict['bev']['scores'] = scores  # np[d_goals]
            per_dict['bev']['agents'] = mapping_b[i]['agents']
            per_dict['bev']['lanes'] = mapping_b[i]['vis_lanes']

        if (cfg.modality == 'both') or (cfg.modality == 'fpv'):
            per_dict['fpv']['gt_traj'] = mapping_f[i]['labels'].reshape(-1, 2)  # np[30,2]
            per_dict['fpv']['pred_traj'] = mapping_f[i]['vis.predict_trajs']  # np[6,30,2]
            per_dict['fpv']['pred_prob'] = pred_probs_dict['fpv'][i]  # np[6]
            goals_2D, masks = trans_3D_2D(calib, i, mapping_b, goals_3D, 'fpv')  # np[d_goals,2]
            goals_2D[masks, :] = np.nan  # mask掉的用nan替换
            per_dict['fpv']['goals_2D'] = goals_2D
            per_dict['fpv']['scores'] = scores  # np[d_goals]
            per_dict['fpv']['agents'] = mapping_f[i]['agents']
            per_dict['fpv']['lanes'] = mapping_f[i]['vis_lanes']

        vis_dicts.append(per_dict)



if __name__ == '__main__':
    main()