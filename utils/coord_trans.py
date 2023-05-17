import numpy as np
from utils.calibration import Calibration


# def trans_3D_2D(calib, i, mapping, point_3D, tag):  # todo: 处理数据的时候保存matrix
#     # point_2D=np[n_goals,3]
#     ego_pos = mapping[i]['agents'][1][-1, :]  # [2] agent的第二个就是av，取他的最后一帧
#     ego_pos = np.concatenate([ego_pos, np.zeros(1)], axis=0)  # [3]
#     ego_rot = -1 * ego_pos[:2]  # [2]
#     ego_hight = 2.0  # 假定的
#     calib.update_extrinsic(ego_pos, ego_rot, ego_hight)
#     if tag == 'bev':
#         point_2D = point_3D[:, 0:2]
#         masks = np.ones(point_2D.shape[0]).astype(np.bool)
#     elif tag == 'fpv':
#         point_2D = calib.project_ego_to_image(point_3D)
#         masks = point_2D[:, -1] > 0.3
#         point_2D = point_2D[:, :2]
#     else:
#         point_2D = point_3D[:, 0:2]
#         masks = np.ones(point_2D.shape[0]).astype(np.bool)
#         print('error tag')
#     return point_2D, masks