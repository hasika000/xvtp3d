import torch
import numpy as np
import os
import json
import pickle
import zlib
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


def make_dataloader(cfg, type):
    dataset_classes = {'unify': Argo_Dataset, 'splited': Argo_Dataset_Splited}
    dname = 'splited' if cfg.DATA.split else 'unify'
    dset = dataset_classes[dname]
    if (cfg.distributed == 1) or (type != 'train'):
        dataset = dset(cfg, type, cfg.modality)
        dataloader = DataLoader(dataset,
                                batch_size=cfg.DATA.batch_size,
                                shuffle=cfg.DATALOADER.shuffle,
                                num_workers=cfg.DATALOADER.num_worker,
                                collate_fn=batch_list_to_batch_tensors)
    else:
        world_size = cfg.distributed
        dataset = dset(cfg, type, cfg.modality)
        train_sampler = DistributedSampler(dataset, shuffle=cfg.DATALOADER.shuffle)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 sampler=train_sampler,
                                                 batch_size=cfg.DATA.batch_size // world_size,
                                                 collate_fn=batch_list_to_batch_tensors)

    return dataloader


# 'file_name', 'start_time', 'city_name', 'cent_x', 'cent_y', 'agent_pred_index', 'two_seconds', 'origin_labels',
#  'angle', 'trajs', 'agents', 'map_start_polyline_idx', 'vis_lanes', 'polygons', 'goals_2D', 'goals_2D_labels',
#  'stage_one_label', 'matrix', 'labels', 'polyline_spans', 'labels_is_valid', 'eval_time'


class Argo_Dataset(Dataset):  # 好在tnt处理数据没有加入scale，所以fpv不会因为这个影响
    def __init__(self, cfg, type_, modality):
        super(Argo_Dataset, self).__init__()
        data_dir = cfg.DATALOADER.data_dir
        self.cfg = cfg
        self.type = type_
        # self.modality = modality
        # self.modality = 'both'

        with open(os.path.join(data_dir, type_ + '_bev_traj.pkl'), 'rb') as f:
            self.ex_list_1 = pickle.load(f)
        with open(os.path.join(data_dir, type_ + '_fpv_traj.pkl'), 'rb') as f:
            self.ex_list_2 = pickle.load(f)

        self.batch_size = cfg.DATA.batch_size

        # load_idx
        if type_ == 'test':
            self.index = [True] * len(self.ex_list_1)
        else:
            id_type = type_ if cfg.DATA.subset else type_ + '_full'
            with open(os.path.join(data_dir, id_type + '_idx.pkl'), 'rb') as f:
                self.index = pickle.load(f)
        assert len(self.index) == len(self.ex_list_1)
        assert len(self.index) == len(self.ex_list_2)
        self.ex_list_1 = [y for x, y in zip(self.index, self.ex_list_1) if x]
        self.ex_list_2 = [y for x, y in zip(self.index, self.ex_list_2) if x]

    def __len__(self):
        return np.sum(self.index)

    def __getitem__(self, idx):
        # file = self.ex_list[idx]
        # pickle_file = open(file, 'rb')
        # instance = pickle.load(pickle_file)
        # pickle_file.close()

        data_compress1 = self.ex_list_1[idx]
        instance1 = pickle.loads(zlib.decompress(data_compress1))
        data_compress2 = self.ex_list_2[idx]
        instance2 = pickle.loads(zlib.decompress(data_compress2))
        # return merge_dict(instance1, instance2)
        return {'bev': instance1, 'fpv': instance2}


class Argo_Dataset_Splited(Dataset):  # 好在tnt处理数据没有加入scale，所以fpv不会因为这个影响
    def __init__(self, cfg, type_, modality):
        super(Argo_Dataset_Splited, self).__init__()
        data_dir = cfg.DATALOADER.data_dir
        self.cfg = cfg
        self.type = type_  # train or test
        # self.modality = modality
        # self.modality = 'both'

        self.batch_size = cfg.DATA.batch_size

        if cfg.DATA.subset:
            id_type = self.type + '_' + 'subset'
        else:
            id_type = self.type + '_' + 'full'
        self.load_datapath = os.path.join(data_dir, 'splited', id_type)
        self.file_name_list = os.listdir(self.load_datapath)

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, idx):
        # file = self.ex_list[idx]
        # pickle_file = open(file, 'rb')
        # instance = pickle.load(pickle_file)
        # pickle_file.close()

        file_name = os.path.join(self.load_datapath, self.file_name_list[idx])
        with open(file_name, 'rb') as f:
            mapping = pickle.loads(zlib.decompress(pickle.load(f)))

        assert 'bev' in mapping.keys()
        assert 'fpv' in mapping.keys()

        return mapping




# def merge_dict(dict1, dict2):
#     list1 = list(dict1.items())
#     list1 = [('bev_' + x[0], x[1]) for x in list1]
#     list2 = list(dict2.items())
#     list2 = [('fpv_' + x[0], x[2]) for x in list2]
#     return dict(list1 + list2)


def batch_list_to_batch_tensors(batch):
    return [each for each in batch]
