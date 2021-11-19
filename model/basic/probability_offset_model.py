from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import factory.model_factory as mf
from model.model_base import ModelBase
import numpy as np
import torch.nn.functional as F


class MiniPointNet(nn.Module):
    def __init__(self):
        super(MiniPointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 32, 1)
        self.conv3 = nn.Conv1d(32, 32, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(32)

    def forward(self, x):                    # x: tensor(batchsize, 3, 64)
        x = F.relu(self.bn1(self.conv1(x)))  # tensor(batchsize, 64, 64)
        x = F.relu(self.bn2(self.conv2(x)))  # tensor(batchsize, 32, 64)
        x = self.bn3(self.conv3(x))          # tensor(batchsize, 32, 64)
        x = torch.max(x, 2)[0]               # tensor(batchsize, 32)
        return x


class CNNs(nn.Module):
    def __init__(self, nx):
        super(CNNs, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=nx, out_channels=nx, kernel_size=(3, 3, 16),
                               stride=(1, 1, 3), padding=(1, 1, 1))  # (batchsize, nx, ny, nphi, 7)
        self.conv2 = nn.Conv3d(in_channels=nx, out_channels=nx, kernel_size=(3, 3, 4),
                               stride=(1, 1, 3), padding=(1, 1, 0))  # (batchsize, nx, ny, nphi, 2)
        self.conv3 = nn.Conv3d(in_channels=nx, out_channels=nx, kernel_size=(3, 3, 2),
                               stride=(1, 1, 1), padding=(1, 1, 0))  # (batchsize, nx, ny, nphi, 1)
        self.bn1 = nn.BatchNorm3d(nx)
        self.bn2 = nn.BatchNorm3d(nx)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # shape (batch_size,64,point_nums)
        x = F.relu(self.bn2(self.conv2(x)))  # shape (batch_size,128,point_nums)
        x = self.conv3(x)                    # shape (batch_size,nx,ny,nphi)
        x = torch.abs(x)
        return x


# only if we subclass from ModelBase, can we get model 'ProbabilityOffset' in in model folder
class ProbabilityOffset(ModelBase):
    def __init__(self, config):
        super(ProbabilityOffset, self).__init__()
        self.nx = config['paras']['nx']
        self.ny = config['paras']['ny']
        self.nphi = config['paras']['nphi']
        self.nx_step = config['paras']['nx_step']
        self.ny_step = config['paras']['ny_step']
        self.nphi_step = config['paras']['nphi_step']
        self.cnns = CNNs(self.nx)
        self.mini_PointNet = MiniPointNet()

    def forward(self, online_data_batch, map_data_batch, gt_data_batch):
        batchsize = map_data_batch.shape[0]
        N = map_data_batch.shape[1]
        n = map_data_batch.shape[2]                 # equals to nx*ny*nphi
        cost_volume = torch.zeros([batchsize, N, n, 32])
        cost_volume = cost_volume.cuda()
        cost_overall = torch.zeros([batchsize, self.nx, self.ny, self.nphi])
        cost_overall = cost_overall.cuda()
        # cost_volume = torch.randn(batchsize, N, n, 32)

        # todo: this loop computes the cost volume
        for i in range(N):
            if i % 20 == 0:
                print(i)
            for j in range(n):
                online_data_batch_trans = online_data_batch[:, i, :, :]
                online_data_batch_trans = online_data_batch_trans.permute(0, 2, 1)   # tensor(batchsize, 3, 64)
                feature_vector_map_trans = map_data_batch[:, i, j, :, :]  # tensor(batchsize, 3, 64)
                feature_vector_map_trans = feature_vector_map_trans.permute(0, 2, 1)
                feature_vector_online = self.mini_PointNet(online_data_batch_trans)  # tensor(batchsize, 32)
                feature_vector_map = self.mini_PointNet(feature_vector_map_trans)    # tensor(batchsize, 32)
                metric_distance_cell = (feature_vector_online - feature_vector_map).\
                    mul(feature_vector_online - feature_vector_map)                        # tensor(batchsize, 32)
                # metric_distance_cell_reshaped = metric_distance_cell.reshape(batchsize, 1, 1, 32)
                cost_volume[:, i, j, :] = metric_distance_cell

        # todo: this loop is the regularization part using 3D CNNs
        cost_regularized = torch.zeros([batchsize, N, self.nx, self.ny, self.nphi])
        cost_regularized = cost_regularized.cuda()
        for i in range(N):
            cost_one_point = cost_volume[:, i, :, :]                         # tensor(batchsize, n, 32)
            cost_one_point_reshaped = cost_one_point.reshape(batchsize, n, 32)
            cost_one_point_reshaped = cost_one_point_reshaped.reshape(batchsize, self.nx, self.ny, self.nphi, 32)
            cost_regularized_one_point = self.cnns(cost_one_point_reshaped)  # (batchsize, nx, ny, nphi, 1)
            cost_regularized_one_point = cost_regularized_one_point.reshape(batchsize, self.nx, self.ny, self.nphi)
            cost_regularized[:, i, :, :, :] = cost_regularized_one_point     # (batchsize, 1, nx, ny, nphi)

        # todo: codes below are for discrete space init
        discrete_delta_x = torch.linspace(-((self.nx - 1) / 2 * self.nx_step), (self.nx - 1) / 2 * self.nx_step, self.nx)
        discrete_delta_y = torch.linspace(-((self.ny - 1) / 2 * self.ny_step), (self.ny - 1) / 2 * self.ny_step, self.ny)
        discrete_delta_phi = torch.linspace(-((self.nphi - 1) / 2 * self.nphi_step),
                                            (self.nphi - 1) / 2 * self.nphi_step, self.nphi)
        discrete_delta_x = discrete_delta_x.reshape(1, self.nx)
        discrete_delta_y = discrete_delta_y.reshape(1, self.ny)
        discrete_delta_phi = discrete_delta_phi.reshape(1, self.nphi)
        discrete_delta_x = torch.repeat_interleave(discrete_delta_x, batchsize, dim=0)
        discrete_delta_x = discrete_delta_x.cuda()
        discrete_delta_y = torch.repeat_interleave(discrete_delta_y, batchsize, dim=0)
        discrete_delta_y = discrete_delta_y.cuda()
        discrete_delta_phi = torch.repeat_interleave(discrete_delta_phi, batchsize, dim=0)
        discrete_delta_phi = discrete_delta_phi.cuda()
        # todo: codes below are for Estimated Offset computing
        match_probability_volume = 1/cost_regularized
        # print("match_probability_volume")
        # print(match_probability_volume)
        match_probability_volume = torch.log(match_probability_volume)         # (batchsize, N, nx, ny, nphi)
        # print("match_probability_volume_log")
        # print(match_probability_volume)
        match_probability_volume_ra = torch.sum(match_probability_volume, dim=1) / N    # (batchsize, nx, ny, nphi) todo: reduce average
        match_probability_volume_ra = torch.exp(match_probability_volume_ra)
        softmax_denominator = torch.sum(torch.sum(torch.sum(match_probability_volume_ra, dim=3), dim=2), dim=1)  # (batchsize,)
        for i in range(batchsize):       # todo: softmax in whole volume
            cost_overall[i, :, :, :] = match_probability_volume_ra[i, :, :, :] / softmax_denominator[i]
        # cost_overall = F.softmax(match_probability_volume_frame, dim=1)  # (batchsize, nx, ny, nphi) todo: softmax in x dimension
        # cost_overall = F.softmax(cost_overall, dim=2)  # (batchsize, nx, ny, nphi) todo: softmax in y dimension
        # cost_overall = F.softmax(cost_overall, dim=3)  # (batchsize, nx, ny, nphi) todo: softmax in phi dimension
        # print("cost_overall")
        # print(cost_overall)
        probability_delta_x = torch.sum(torch.sum(cost_overall, dim=2), dim=2)    # (batchsize, nx) todo: reduce sum
        probability_delta_y = torch.sum(torch.sum(cost_overall, dim=1), dim=2)    # (batchsize, ny) todo: reduce sum
        probability_delta_phi = torch.sum(torch.sum(cost_overall, dim=1), dim=1)  # (batchsize, nphi) todo: reduce sum
        # print(probability_delta_x, probability_delta_y, probability_delta_phi)
        pred_delta_x = torch.sum(torch.mul(discrete_delta_x, probability_delta_x), dim=1)        # (batchsize)
        pred_delta_y = torch.sum(torch.mul(discrete_delta_y, probability_delta_y), dim=1)        # (batchsize)
        pred_delta_phi = torch.sum(torch.mul(discrete_delta_phi, probability_delta_phi), dim=1)  # (batchsize)
        pred_offset = torch.cat((pred_delta_x.reshape([batchsize, 1]),
                                 pred_delta_y.reshape([batchsize, 1]),
                                 pred_delta_phi.reshape([batchsize, 1])), dim=1)                 # (batchsize, 3)
        # print(pred_delta_x, pred_delta_y, pred_delta_phi)
        return pred_offset
