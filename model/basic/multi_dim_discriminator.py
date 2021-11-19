import torch
import torch.nn as nn
from model.model_base import ModelBase
import factory.model_factory as mf
from collections import OrderedDict


class MultiDimMLPDiscriminator(ModelBase):
    def __init__(self, config):
        super(MultiDimMLPDiscriminator, self).__init__()
        self.sub_mlp_num = config['paras']['sub_mlp_num']
        self.sub_mlp_name = []
        # todo: optimize the config reading by add Node class
        self.feature_encoding_model = mf.ModelFactory.get_model(config['paras']['feature_encoding_model'])
        self.sub_mlp_input_size = config['paras']['sub_mlp']['config_file']['expanded']['paras']['input_size']
        for i in range(self.sub_mlp_num):
            self.sub_mlp_name.append(str(i))
            self.add_module(str(i), mf.ModelFactory.get_model(config['paras']['sub_mlp']))

    def forward(self, features, x):
        # shape of x: batch size, channel_dim, channel_num

        batch_size, channel_dim = x.shape
        if channel_dim + self.feature_encoding_model.output_shape != self.sub_mlp_input_size:
            raise TypeError(f'input dim of each channel must be the sum of {channel_dim} + {self.feature_encoding_model.output_shape}')
        feature_encoding = self.feature_encoding_model(features)

        sub_input = torch.cat([feature_encoding, x], dim=1)
        sub_output = self._modules['0'](sub_input).unsqueeze(-1)
        return sub_output


