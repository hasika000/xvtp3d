# 过滤label在视野外的情况，顺便生成subset

import os
import argparse
import numpy as np
import pickle
import zlib
from tqdm import tqdm


def filter_out(fpv_file, tag, dataset_tag, subset):
    if dataset_tag == 'argo':
        x_image_lim = 1920
        y_image_lim = 1200
    elif dataset_tag == 'ns':
        x_image_lim = 1600
        y_image_lim = 900
    else:
        x_image_lim = 0
        y_image_lim = 0

    xlim = (-2 * x_image_lim, 3 * x_image_lim)
    ylim = (-2 * y_image_lim, 3 * y_image_lim)

    with open(fpv_file, 'rb') as f:
        ex_list = pickle.load(f)

    re_dict = {'traj':[], 'lane': [], 'goal': [], 'total': []}
    output_idx = []

    for idx in tqdm(range(len(ex_list))):
        data_compress = ex_list[idx]
        instance = pickle.loads(zlib.decompress(data_compress))

        flag1 = True  # 未来轨迹要求
        flag2 = True  # lane要求
        flag3 = True  # goal要求
        flag4 = True  # subset

        # flag1
        labels = instance['labels']  # [30,2]
        if np.isnan(labels).any():
            flag1 = False
        if ((labels[:, 0] < xlim[0]) | (labels[:, 0] > xlim[1]) | (labels[:, 1] < ylim[0]) | (labels[:, 1] > ylim[1])).any():
            flag1 = False

        # flag2
        polygons = instance['polygons']  # list[n_lane]=[n,2]
        stage_one_label = instance['stage_one_label']
        polygon = polygons[stage_one_label]
        if np.isnan(polygon).all():
            flag2 = False
        if ((polygon[:, 0] < xlim[0]) | (polygon[:, 0] > xlim[1]) | (polygon[:, 1] < ylim[0]) | (polygon[:, 1] > ylim[1])).any():
            flag2 = False

        # flag3
        goals_2D = instance['goals_2D']  # [n,2]
        goals_2D_labels = instance['goals_2D_labels']
        goal = goals_2D[goals_2D_labels]  # [2]
        if np.isnan(goal).any():
            flag3 = False
        if ((goal[0] < xlim[0]) | (goal[0] > xlim[1]) | (goal[1] < ylim[0]) | (goal[1] > ylim[1])).any():
            flag3 = False

        # flag4
        if subset:
            if idx % 10 != 0:
                flag4 = False

        output_idx.append(flag1 and flag2 and flag3 and flag4)  # 都是true才可用

        re_dict['traj'].append(flag1)
        re_dict['lane'].append(flag2)
        re_dict['goal'].append(flag3)
        re_dict['total'].append(flag1 and flag2 and flag3)

    # final
    if subset:
        file_name = os.path.join(os.path.split(fpv_file)[0], tag + '_idx.pkl')
    else:
        file_name = os.path.join(os.path.split(fpv_file)[0], tag + '_full' + '_idx.pkl')
    with open(file_name, 'wb') as f:
        pickle.dump(output_idx, f)

    print('total data is {}, filter is {:.2f}'.format(len(ex_list), np.sum(~np.array(output_idx)) / len(ex_list) * 100))
    for key in re_dict.keys():
        print(key + ' filter data is {} ({:.2f}%)'.format(
            np.sum(~np.array(re_dict[key])), np.sum(~np.array(re_dict[key])) / len(ex_list) * 100)
              )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_tag', default='argo', type=str)  # argo or ns
    parser.add_argument('--subset', action='store_true')
    args = parser.parse_args()

    dataset_tag = args.dataset_tag
    if args.subset:
        subset = True
    else:
        subset = False

    if dataset_tag == 'argo':
        data_dir = '../argoverse-data/processed'
    elif dataset_tag == 'ns':
        data_dir = '../nuscenes-data/processed'
    else:
        data_dir = 'none'

    train_fpv_file = os.path.join(data_dir, 'train_fpv_traj.pkl')

    test_fpv_file = os.path.join(data_dir, 'val_fpv_traj.pkl')

    filter_out(train_fpv_file, 'train', dataset_tag, subset)
    filter_out(test_fpv_file, 'val', dataset_tag, subset)


if __name__ == '__main__':
    main()