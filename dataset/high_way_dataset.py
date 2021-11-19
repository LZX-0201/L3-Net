from dataset.dataset_base import DatasetBase
from torch.utils.data import DataLoader
from pathlib import Path
import torch
import numpy as np
from collections import defaultdict
import os
import pickle
import math


'''
    The dataset gives data including following keys:
        'frame_id', 
        'calib', 
        'gt_boxes', 
        'points', 
        'use_lead_xyz',
        'voxels', 
        'voxel_coords', 
        'voxel_num_points', 
        'image_shape', 
        'batch_size'
'''


class HighWayDataset(DatasetBase):
    def __init__(self, config):
        super().__init__()
        self._is_train = config['paras']['for_train']                # necessary
        self._data_root = config['paras']['data_root']               # necessary
        self._batch_size = config['paras']['batch_size']             # necessary
        self._shuffle = config['paras']['shuffle']                   # necessary
        self._num_workers = config['paras']['num_workers']           # necessary
        self.online_data_root = config['paras']['online_data_root']  # necessary
        self.map_data_root = config['paras']['map_data_root']        # necessary
        self.gt_offset_root = config['paras']['gt_offset_root']
        self.gt_offset = np.load(self.gt_offset_root)
        self.count = 0

    def __len__(self):
        return 1300    # this will be modified after last

    def __getitem__(self, item):                                                 # item begin from 0
        assert item <= self.__len__()                                            # todo: assert item < self.__len__() ??
        online_frame = np.load(self.online_data_root + str(item + 1) + '.npy')  # array(N*64*3)
        map_frame = np.load(self.map_data_root + str(item + 1) + '.npy')        # array(N*n*64*3)
        map_frame = map_frame[:, (114, 115, 116, 121, 122, 123, 128, 129, 130,
                                    163, 164, 165, 170, 171, 172, 177, 178, 179,
                                    212, 213, 214, 219, 220, 221, 226, 227, 228), :, :]  # array(N*27*64*3)

        gt_offset_frame = self.gt_offset[item, :]                               # array(3,)
        data_frame = {'online_frame': online_frame, 'map_frame': map_frame, 'gt_offset_frame': gt_offset_frame}
        return data_frame

    # modify to dataset into a tensor according to batch_size
    def get_data_loader(self, distributed=False):
        if distributed:
            if self._is_train:
                sampler = torch.utils.data.distributed.DistributedSampler(self)
            else:
                raise NotImplementedError
        else:
            sampler = None
        self.count = self.count + 1
        # print(self.count)

        return DataLoader(
            dataset=self,
            batch_size=self._batch_size,
            shuffle=(sampler is None) and self._shuffle,
            num_workers=self._num_workers,
            pin_memory=True,
            collate_fn=HighWayDataset.collate_batch,  # todo:may I use default collate_fn?
            drop_last=False,
            sampler=sampler,
            timeout=0
        )

    @staticmethod
    def load_data_to_gpu(data_list):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # online_data_batch = torch.from_numpy(data_list[1]).float().cuda().to(device)
        # map_data_batch = torch.from_numpy(data_list[0]).float().cuda()
        # gt_data_batch = torch.from_numpy(data_list[2]).float().cuda()
        online_data_batch = torch.from_numpy(data_list[1]).float().to(device)
        map_data_batch = torch.from_numpy(data_list[0]).float().to(device)
        gt_data_batch = torch.from_numpy(data_list[2]).float().to(device)
        return online_data_batch, map_data_batch, gt_data_batch

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        # print(len(batch_list))
        dict_frame = batch_list[0]               # dict{array(N*n*64*3), array(N*64*3), array(3,)}\
        array_frame = dict_frame['map_frame']  # array(N*n*64*3)
        N = array_frame.shape[0]
        n = array_frame.shape[1]
        map_feature_batch = np.zeros([len(batch_list), N, n, 64, 3])
        online_feature_batch = np.zeros([len(batch_list), N, 64, 3])
        gt_offset_batch = np.zeros([len(batch_list),3])
        count_batch = 0
        for data_dict in batch_list:
            map_feature_frame = data_dict['map_frame']
            online_feature_frame = data_dict['online_frame']
            gt_offset_frame = data_dict['gt_offset_frame']  # array(3,)
            map_feature_batch[count_batch, :, :, :, :] = map_feature_frame
            online_feature_batch[count_batch, :, :, :] = online_feature_frame
            gt_offset_batch[count_batch, :] = gt_offset_frame
        stacked_batch_list = [map_feature_batch, online_feature_batch, gt_offset_batch]
        return stacked_batch_list
