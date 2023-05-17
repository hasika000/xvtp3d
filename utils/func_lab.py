import torch
import numpy as np
from torch import nn
# import logging
from utils import log_help
import torch.nn.utils.rnn as rnn
import os
import inspect
import pickle
import math
import json
from utils.calibration import Calibration
from utils.calibration_ns import Calibration_NS


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def print_info(args, cfg, device, save_dir, logger):
    logger.info('\n---------basic information---------\n')
    logger.info('save_tag = {}'.format(cfg.save_tag))
    logger.info('save_dir = {}'.format(save_dir))
    logger.info('device = {}'.format(device))
    logger.info('distributed = {}'.format(cfg.distributed))
    logger.info('fpv_type = {}'.format(cfg.fpv_type))
    logger.info('save_model = {}'.format(cfg.save_model))
    logger.info('modality = {}'.format(cfg.modality))
    logger.info('scheduler_lr_type = {}'.format(cfg.SOLVER.scheduler_lr_type))
    if cfg.SOLVER.scheduler_lr_type == 'p':
        logger.info('patience = {}'.format(cfg.SOLVER.patience))
    elif cfg.SOLVER.scheduler_lr_type == 'm':
        logger.info('milestones = {}'.format(cfg.SOLVER.milestones))
    elif cfg.SOLVER.scheduler_lr_type == 's':
        logger.info('step_size = {}'.format(cfg.SOLVER.step_size))
    logger.info('batch_size = {}'.format(cfg.DATA.batch_size))
    logger.info('subset = {}'.format(cfg.DATA.subset))
    logger.info('bev_loss_weight = {}'.format(cfg.SOLVER.bev_weight))
    logger.info('fpv_loss_weight = {}'.format(cfg.SOLVER.fpv_weight))
    logger.info('\n-----------------------------------\n')
    logger.info('\n---------modeling information---------\n')
    logger.info('hidden_size = {}'.format(cfg.MODEL.hidden_size))
    logger.info('mode_num = {}'.format(cfg.MODEL.mode_num))
    logger.info('gt_decoder = {}'.format(cfg.MODEL.gt_decoder))
    logger.info('cross_type = {}'.format(cfg.MODEL.cross_type))
    logger.info('random_mask = {}'.format(cfg.MODEL.random_mask))
    logger.info('cross_enc = {}'.format(cfg.MODEL.cross_enc))
    logger.info('self_rein = {}'.format(cfg.MODEL.self_rein))
    logger.info('share_weight = {}.'.format(cfg.MODEL.share_weight))
    logger.info('\n-----------------------------------\n')


class MyLogger(object):
    def __init__(self, directory):
        self.directory = directory

    def info(self, message):
        with open(self.directory, 'a') as f:
            f.write(message + '\n')

        print(message)




def update_best_metric(epoch, metric_dict_best, metric, modality_set):
    for modality in modality_set:
        for metric_type in list(metric[modality].keys()):
            if metric_dict_best[modality][metric_type][1] > metric[metric_type]:
                metric_dict_best[modality][metric_type][0] = epoch
                metric_dict_best[modality][metric_type][1] = metric[metric_type]


def get_metric_dict(pred_pos, future_traj, num_sample):
    # 统一地计算ade和fde，返回一个dict，包括ade和fde
    # pred_pos=[pred_len,K*bs,2], future_traj=[pred_len,bs,2]  都是np
    metric_dict_batch = {'ade': 0.0, 'fde': 0.0}

    future_traj = np.tile(future_traj[:, None, :, :], (1, num_sample, 1, 1))  # [pred_len,K,bs,2]
    pred_pos = pred_pos.reshape(pred_pos.shape[0], num_sample, -1, pred_pos.shape[-1])  # [pred_len,K,bs,2]

    ade_all = np.mean(np.sqrt(np.sum((pred_pos - future_traj) ** 2, axis=3)), axis=0)  # [K,bs]
    fde_all = np.sqrt(np.sum((pred_pos - future_traj) ** 2, axis=3))[-1, :, :]  # [K,bs]
    if num_sample == 1:
        # ml
        ade = np.mean(ade_all)
        fde = np.mean(fde_all)
    else:
        # bon
        ade = np.mean(np.min(ade_all, axis=0))
        fde = np.mean(np.min(fde_all, axis=0))
    metric_dict_batch['ade'] = ade
    metric_dict_batch['fde'] = fde

    return metric_dict_batch  # {'ade': , 'fde': }


def get_miss_rate(li_FDE, dis=2.0):
    return np.sum(np.array(li_FDE) > dis) / len(li_FDE) if len(li_FDE) > 0 else None


def get_from_mapping(mapping, key=None):  # todo: 检查both之后这里面会不会不加内容
    if key is None:
        line_context = inspect.getframeinfo(inspect.currentframe().f_back).code_context[0]
        key = line_context.split('=')[0].strip()
    return [each[key] for each in mapping]


def get_dis_point_2_points(point, points):
    assert points.ndim == 2
    return np.sqrt(np.square(points[:, 0] - point[0]) + np.square(points[:, 1] - point[1]))


# def get_calib_from_file(calib_path, ego_pos, in_xyz, tag):  # todo: 改成process的时候保存，这个时候读取
#     with open(calib_path, 'r') as f:
#         calib_file = json.load(f)
#     camera_data = calib_file['camera_data_'][5]
#     assert 'ring_front_center' in camera_data['key']
#     calib = Calibration(camera_data)
#     # ego_pos = instance['agents'][1][-1, :]
#     ego_pos = np.concatenate([ego_pos, np.zeros(1)], axis=0)  # [3]
#     ego_rot = -1 * ego_pos[:2]  # [2]
#     ego_hight = 2.0  # 假定的
#     calib.update_extrinsic(ego_pos, ego_rot, ego_hight)
#
#     if tag == 'world_image':
#         out_xyz = calib.project_ego_to_image(in_xyz)
#     else:
#         out_xyz = calib.project_image_to_ego(in_xyz)
#
#     return out_xyz


# def batch_init(mapping):
#     global traj_last, origin_point, origin_angle
#     batch_size = len(mapping)
#
#     global origin_point, origin_angle
#     origin_point = np.zeros([batch_size, 2])
#     origin_angle = np.zeros([batch_size])
#     for i in range(batch_size):
#         origin_point[i][0], origin_point[i][1] = rotate(0 - mapping[i]['cent_x'], 0 - mapping[i]['cent_y'],
#                                                         mapping[i]['angle'])
#         origin_angle[i] = -mapping[i]['angle']
#
#     def load_file2pred():
#         global file2pred
#         if len(file2pred) == 0:
#             with open(args.other_params['set_predict_file2pred'], 'rb') as pickle_file:
#                 file2pred = pickle.load(pickle_file)


def batch_init_origin(mapping):  # 生成一个batch的origin_point和origin_angle
    batch_size = len(mapping)

    origin_point = np.zeros([batch_size, 2])
    origin_angle = np.zeros([batch_size])
    for i in range(batch_size):
        origin_point[i][0], origin_point[i][1] = rotate(0 - mapping[i]['cent_x'], 0 - mapping[i]['cent_y'],
                                                        mapping[i]['angle'])
        origin_angle[i] = -mapping[i]['angle']
    return origin_point, origin_angle

    # def load_file2pred():
    #     global file2pred
    #     if len(file2pred) == 0:
    #         with open(args.other_params['set_predict_file2pred'], 'rb') as pickle_file:
    #             file2pred = pickle.load(pickle_file)


def merge_tensors(tensors, device, hidden_size):
    """
    merge a list of tensors into a tensor
    """
    lengths = []
    for tensor in tensors:
        lengths.append(tensor.shape[0] if tensor is not None else 0)
    res = torch.zeros([len(tensors), max(lengths), hidden_size], device=device)
    for i, tensor in enumerate(tensors):
        if tensor is not None:
            res[i][:tensor.shape[0]] = tensor
    return res, lengths


def de_merge_tensors(tensor, lengths):
    return [tensor[i, :lengths[i]] for i in range(len(lengths))]


def trans_3D_2D(calib, i, mapping, point_3D, tag):  # todo: 处理数据的时候保存matrix
    # point_2D=np[n_goals,3]
    if isinstance(calib, Calibration):  # argo
        if calib.fpv_type == 'ego':  # 以ego为fpv
            ego_pos = mapping[i]['agents'][1][-1, :]  # [2] agent的第二个就是av，取他的最后一帧
        elif calib.fpv_type == 'self':  # 以self为fpv
            ego_pos = mapping[i]['agents'][0][0, :]  # 取self的第一帧
        else:
            ego_pos = mapping[i]['agents'][1][-1, :]
            print('fpv_type error')
        ego_pos = np.concatenate([ego_pos, np.zeros(1)], axis=0)  # [3]
        ego_rot = -1 * ego_pos[:2]  # [2]
        ego_hight = 2.0  # 假定的
    elif isinstance(calib, Calibration_NS):  # nuscenes
        ego_pos = np.array([-20, 0])  # [2] 向后平移20m  todo:如果20改了，那么数据处理和这里都要改
        ego_pos = np.concatenate([ego_pos, np.zeros(1)], axis=0)  # [3]
        ego_rot = -1 * ego_pos[:2]  # [2]
        ego_hight = 2.0  # 假定的
    else:
        pass
    calib.update_extrinsic(ego_pos, ego_rot, ego_hight)
    if tag == 'bev':
        point_2D = point_3D[:, 0:2]
        masks = np.ones(point_2D.shape[0]).astype(np.bool)
    elif tag == 'fpv':
        point_2D = calib.project_ego_to_image(point_3D)
        masks = point_2D[:, -1] > 0.3
        point_2D = point_2D[:, :2]
    else:
        point_2D = point_3D[:, 0:2]
        masks = np.ones(point_2D.shape[0]).astype(np.bool)
        print('error tag')
    masks = ~masks  # mask=true是不要的
    point_2D[masks, :] = np.zeros(2)  # mask掉的用0替换
    return point_2D, masks


def get_neighbour_points(points, topk_ids=None, mapping=None, neighbour_dis=2):
    # grid = np.zeros([300, 300], dtype=int)
    grid = {}
    for fake_idx, point in enumerate(points):
        if np.isnan(point).any():
            continue  # todo: 这里如何双视角对应，是加载数据之前统一计算dense goal还是怎么办
        x, y = round(float(point[0])), round(float(point[1]))

        # not compatible argo
        for i in range(-neighbour_dis, neighbour_dis + 1):
            for j in range(-neighbour_dis, neighbour_dis + 1):
                grid[(x + i, y + j)] = 1
    points = list(grid.keys())
    return points


def get_neighbour_3Dpoints(points, topk_ids=None, neighbour_dis=2):
    # grid = np.zeros([300, 300], dtype=int)
    grid = {}
    for fake_idx, point in enumerate(points):
        x, y = round(float(point[0])), round(float(point[1]))
        z = 0

        # not compatible argo
        for i in range(-neighbour_dis, neighbour_dis + 1):
            for j in range(-neighbour_dis, neighbour_dis + 1):
                grid[(x + i, y + j, z)] = 1
    points = list(grid.keys())
    return np.array(points)  # [n,3]


def to_origin_coordinate(points, idx_in_batch, origin_point, origin_angle, scale=None):
    for point in points:
        point[0], point[1] = rotate(point[0] - origin_point[idx_in_batch][0],
                                    point[1] - origin_point[idx_in_batch][1], origin_angle[idx_in_batch])
        if scale is not None:
            point[0] *= scale
            point[1] *= scale


def rotate(x, y, angle):
    res_x = x * math.cos(angle) - y * math.sin(angle)
    res_y = x * math.sin(angle) + y * math.cos(angle)
    return res_x, res_y


def get_dis(points: np.ndarray, point_label):
    return np.sqrt(np.square((points[:, 0] - point_label[0])) + np.square((points[:, 1] - point_label[1])))


# def logger_and_writer(cfg, iter, epoch, output_dict, optimizer, loader_len, weight, time_cost, writer):
#     # logger
#     logging.info(
#         '{}/{}, [epoch {}/{}], loss = {:.6f}, lr = {}, kl_weight = {:.7f}, time/batch = {:.3f}'.format(
#             iter, loader_len, epoch, cfg.num_epochs,
#             output_dict['loss'].item(),
#             optimizer.param_groups[0]['lr'], weight,
#             time_cost))
#     for modal_conf in cfg.train_modal_configs:
#         logging.info(
#             '\t \t |{}: total_loss = {:.6f}, recons_loss = {:.6f}, kl_loss = {:.6f}'.format(
#                 modal_conf,
#                 output_dict[modal_conf]['total_loss'].item(),
#                 output_dict[modal_conf]['recons_loss'].item(),
#                 output_dict[modal_conf]['kl_loss'].item()))
#     if cfg.MODEL.share_module:
#         logging.info(
#             '\t \t |shared module: sim_loss = {:.6f}, cross_kl_loss = {:.6f}, pse_loss = {:.6f}'.format(
#                 output_dict['sim_loss'].item(),
#                 output_dict['cross_kl_loss'].item(),
#                 output_dict['pseudo_label_loss'].item()))
#
#     # writer
#     writer.add_scalar('loss/iter', output_dict['loss'].item(), iter + (epoch - 1) * loader_len)
#     for modal_conf in cfg.train_modal_configs:
#         writer.add_scalar('total_loss_' + modal_conf + '/iter',
#                           output_dict[modal_conf]['total_loss'].item(),
#                           iter + (epoch - 1) * loader_len)
#         writer.add_scalar('recons_loss_' + modal_conf + '/iter',
#                           output_dict[modal_conf]['recons_loss'].item(),
#                           iter + (epoch - 1) * loader_len)
#         writer.add_scalar('kl_loss_' + modal_conf + '/iter',
#                           output_dict[modal_conf]['kl_loss'].item(),
#                           iter + (epoch - 1) * loader_len)


