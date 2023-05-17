import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from modeling.vectornet import VectorNet
from modeling.decoder import Decoder, TrajCompleter
from modeling.scores_net import ScoresNet
from utils.func_lab import get_from_mapping, batch_init_origin, to_origin_coordinate, trans_3D_2D, get_dis_point_2_points
from optimization.utils_optimize import select_goals_by_optimization
import json
from utils.calibration import Calibration
from utils.calibration_ns import Calibration_NS


class TNT3D(nn.Module):
    def __init__(self, cfg, device, modality):
        super(TNT3D, self).__init__()
        self.cfg = cfg
        self.device = device
        self.m = modality
        self.future_frame_num = cfg.DATA.pred_len
        self.mode_num = cfg.MODEL.mode_num

        if 'ns' not in cfg.save_tag:  # argo
            with open(cfg.DATALOADER.calib_path, 'r') as f:
                calib_file = json.load(f)
            camera_data = calib_file['camera_data_'][5]
            assert 'ring_front_center' in camera_data['key']
            self.calib = Calibration(camera_data)
            self.calib.fpv_type = cfg.fpv_type
        else:  # nuscenes
            with open(cfg.DATALOADER.calib_path, 'r') as f:
                intrinsic = json.load(f)
            intrinsic = np.concatenate([np.array(intrinsic), np.zeros((3, 1))], axis=-1)
            self.calib = Calibration_NS(intrinsic)

        if (self.m == 'both') or (self.m == 'bev'):
            self.bev_enc = VectorNet(cfg, device)
        if (self.m == 'both') or (self.m == 'fpv'):
            self.fpv_enc = VectorNet(cfg, device)
        self.scores_net = ScoresNet(cfg, device, self.calib)
        self.bev_com = TrajCompleter(device, cfg.MODEL.hidden_size, self.future_frame_num)
        self.fpv_com = TrajCompleter(device, cfg.MODEL.hidden_size, self.future_frame_num)

    def goals_3D_eval(self, batch_size, mapping, labels, goals_3D_list, scores_list):
        goals_scores_tuple = []
        for i in range(batch_size):
            goals_scores_tuple.append((goals_3D_list[i][:, 0:2].astype(np.float32), scores_list[i].detach().cpu().numpy()))
        pred_goals_batch, pred_probs_batch, _ = select_goals_by_optimization(
            self.cfg, np.array(labels).reshape([batch_size, self.future_frame_num, 2]), mapping, goals_scores_tuple)

        pred_goals_batch = np.stack(pred_goals_batch, axis=0)
        pred_probs_batch = np.stack(pred_probs_batch, axis=0)
        assert pred_goals_batch.shape == (batch_size, self.mode_num, 3)
        assert pred_probs_batch.shape == (batch_size, self.mode_num)

        # if args.visualize:  todo: add 可视化
        #     for i in range(batch_size):
        #         utils.visualize_goals_2D(mapping[i], mapping[i]['vis.goals_2D'], mapping[i]['vis.scores'], self.future_frame_num,
        #                                  labels=mapping[i]['vis.labels'],
        #                                  labels_is_valid=mapping[i]['vis.labels_is_valid'],
        #                                  predict=mapping[i]['vis.predict_trajs'])

        return pred_goals_batch, pred_probs_batch

    def goals_to_traj(self, i, loss, FDEs, mapping, highest_list, goals_3D_list, target, inputs, lens, h_states, gt_points, labels_is_valid, infer, tag):
        assert not infer

        if (self.m != 'both') and (tag != self.m):  # 不需要预测这个轨迹
            return torch.zeros(1, 30, 2), torch.zeros(2)

        final_idx = mapping[i].get('final_idx', -1)

        if tag == 'bev':
            completer = self.bev_com
        else:
            completer = self.fpv_com

        pred_traj = completer(target, inputs[i][:lens[i]], h_states[i], infer)  # [k,30,2]

        # 计算traj loss和score loss
        gt_points_tensor = torch.tensor(gt_points, dtype=torch.float, device=self.device)
        label_valid_tensor = torch.tensor(labels_is_valid[i], dtype=torch.float, device=self.device).view(
            self.future_frame_num, 1)
        loss[i] += (F.smooth_l1_loss(pred_traj.squeeze(0), gt_points_tensor,
                                     reduction='none') * label_valid_tensor).mean()

        index = highest_list[i]
        pred_goals, _ = trans_3D_2D(self.calib, i, mapping, goals_3D_list[i][index:index + 1, :], tag)

        FDEs[i] = np.linalg.norm(pred_goals[0] - gt_points[final_idx], ord=2)

        return pred_traj, pred_goals

    def cross_view_fusion(self, scores_b, scores_f, goals_b, goals_f, ego_pos):
        aaa = 1

    def forward(self, batch, infer):
        # if self.m == 'both':
        FDE_dict = {'bev': None, 'fpv': None}
        pred_trajs_dict = {'bev': [], 'fpv': []}
        pred_probs_dict = {'bev': [], 'fpv': []}
        mapping_b = get_from_mapping(batch, 'bev')
        mapping_f = get_from_mapping(batch, 'fpv')
        bs = len(mapping_b)
        loss_s = torch.zeros(bs).to(self.device)

        # encoder
        if (self.m == 'both') or (self.m == 'bev'):
            f_lane_b, l_masks_b, inputs_b, lens_b, h_states_b = self.bev_enc(mapping_b, self.device, infer)
        else:
            f_lane_b, l_masks_b, inputs_b, lens_b, h_states_b = (None,) * 5
        if (self.m == 'both') or (self.m == 'fpv'):
            f_lane_f, l_masks_f, inputs_f, lens_f, h_states_f = self.fpv_enc(mapping_f, self.device, infer)
        else:
            f_lane_f, l_masks_f, inputs_f, lens_f, h_states_f = (None,) * 5

        loss_b, loss_f, scores_list, goals_3D_list, highest_list = self.scores_net(
            bs, mapping_b, mapping_f, f_lane_b, f_lane_f, l_masks_b, l_masks_f,
            inputs_b, inputs_f, lens_b, lens_f, h_states_b, h_states_f, infer
        )

        labels_b = get_from_mapping(mapping_b, 'labels')
        labels_f = get_from_mapping(mapping_f, 'labels')
        labels_is_valid_b = get_from_mapping(mapping_b, 'labels_is_valid')
        labels_is_valid_f = get_from_mapping(mapping_f, 'labels_is_valid')
        labels_np_b = np.array(labels_b).reshape([bs, self.future_frame_num, 2])
        labels_np_f = np.array(labels_f).reshape([bs, self.future_frame_num, 2])

        if not infer:
            FDEs_b = np.zeros(bs)
            FDEs_f = np.zeros(bs)
            pred_batch_b = None
            pred_batch_f = None
        else:
            pred_3D, pred_probs = self.goals_3D_eval(bs, mapping_b, labels_b, goals_3D_list, scores_list)  # np[bs,6,2], np[bs,6]
            origin_point_b, origin_angle_b = batch_init_origin(mapping_b)
            origin_point_f, origin_angle_f = batch_init_origin(mapping_f)
            pred_probs_dict['bev'] = pred_probs
            pred_probs_dict['fpv'] = pred_probs

            FDEs_b = []  # method2FDE=list[bs]=FDE
            FDEs_f = []
            pred_batch_b = []
            pred_batch_f = []

        for i in range(bs):
            final_idx = mapping_b[i].get('final_idx', -1)
            gt_points_b = labels_b[i].reshape([self.future_frame_num, 2])  # np=[30,2]
            gt_points_f = labels_f[i].reshape([self.future_frame_num, 2])  # np=[30,2]

            if not infer:
                if self.cfg.MODEL.gt_decoder:
                    target_b = gt_points_b[final_idx][None, :]
                    target_f = gt_points_f[final_idx][None, :]
                else:
                    print('not recommend')
                    target_b = None
                    target_f = None

                pred_traj_b, pred_goals_b = self.goals_to_traj(
                    i, loss_b, FDEs_b, mapping_b, highest_list, goals_3D_list, target_b, inputs_b, lens_b, h_states_b,
                    gt_points_b, labels_is_valid_b, infer, 'bev'
                )
                pred_traj_f, pred_goals_f = self.goals_to_traj(
                    i, loss_f, FDEs_f, mapping_b, highest_list, goals_3D_list, target_f, inputs_f, lens_f, h_states_f,
                    gt_points_f, labels_is_valid_f, infer, 'fpv'
                )

                grid_labels = torch.tensor([mapping_b[i]['goals_2D_labels']], device=self.device)
                loss_s[i] += F.nll_loss(scores_list[i].unsqueeze(0), grid_labels)

                pred_trajs_dict['bev'].append(pred_traj_b.detach().cpu().numpy())
                pred_trajs_dict['fpv'].append(pred_traj_f.detach().cpu().numpy())

            else:
                pred_goals_b, _ = trans_3D_2D(self.calib, i, mapping_b, pred_3D[i, :, :], 'bev')  # [6,2]
                pred_goals_f, _ = trans_3D_2D(self.calib, i, mapping_b, pred_3D[i, :, :], 'fpv')
                pred_batch_b.append(pred_goals_b)
                pred_batch_f.append(pred_goals_f)
                FDE_b = np.min(get_dis_point_2_points(labels_np_b[i][-1], pred_goals_b))
                FDE_f = np.min(get_dis_point_2_points(labels_np_f[i][-1], pred_goals_f))
                FDEs_b.append(FDE_b)
                FDEs_f.append(FDE_f)

                # bev
                if (self.m == 'both') or (self.m == 'bev'):
                    target = pred_batch_b[i]

                    pred_traj = self.bev_com(target, inputs_b[i][:lens_b[i]], h_states_b[i], infer)  # [k,30,2]

                    # 把终点改成预测的，然后to原本coord
                    pred_traj = pred_traj.detach().cpu().numpy()
                    final_idx = mapping_b[i].get('final_idx', -1)
                    pred_traj[:, final_idx, :] = pred_batch_b[i]
                    mapping_b[i]['vis.predict_trajs'] = pred_traj.copy()

                    for each in pred_traj:
                        to_origin_coordinate(each, i, origin_point_b, origin_angle_b)
                else:
                    pred_traj = np.zeros([1, 30, 2])

                pred_trajs_dict['bev'].append(pred_traj)

                # fpv
                if (self.m == 'both') or (self.m == 'fpv'):
                    target = pred_batch_f[i]

                    pred_traj = self.fpv_com(target, inputs_f[i][:lens_f[i]], h_states_f[i], infer)  # [k,30,2]

                    # 把终点改成预测的，然后to原本coord
                    pred_traj = pred_traj.detach().cpu().numpy()
                    final_idx = mapping_f[i].get('final_idx', -1)
                    pred_traj[:, final_idx, :] = pred_batch_f[i]
                    mapping_f[i]['vis.predict_trajs'] = pred_traj.copy()

                    for each in pred_traj:
                        to_origin_coordinate(each, i, origin_point_f, origin_angle_f)
                else:
                    pred_traj = np.zeros([1, 30, 2])

                pred_trajs_dict['fpv'].append(pred_traj)

        FDE_dict['bev'] = FDEs_b
        FDE_dict['fpv'] = FDEs_f
        pred_trajs_dict['bev'] = np.stack(pred_trajs_dict['bev'], axis=0)  # [bs,k,30,2]
        pred_trajs_dict['fpv'] = np.stack(pred_trajs_dict['fpv'], axis=0)  # [bs,k,30,2]

        loss = torch.mean(loss_b) * self.cfg.SOLVER.bev_weight + torch.mean(loss_f) * self.cfg.SOLVER.fpv_weight + torch.mean(loss_s)  # 加入weight

        return loss, FDE_dict, pred_trajs_dict, pred_probs_dict  # loss, list[bs]=float, np=[bs,k,30,2], np=[bs,k]
