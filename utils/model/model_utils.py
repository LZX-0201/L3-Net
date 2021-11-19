import torch
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as func
from torch.nn.modules.utils import _pair

act_func_dict = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'none': None
}

# todo: parameterize it or move it to specific model if rarely used
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# todo: function for concatenate: shoule be located in model
def Change_data(data):  # 对真值图片和标签进行处理(之前几版代码见pytorch-dcgan-mnist)
    pic_len = data[0].shape[1] * data[0].shape[2] * data[0].shape[3]  # 获取每一张图片压缩后的总像素个数
    img_Din = data[0].numpy().squeeze().reshape((data[0].shape[0], 1, pic_len))  # 变为batch_size*1*16
    label_Din = data[1].unsqueeze(-1).unsqueeze(-1).numpy()  # 获得对应label
    img_Din = torch.from_numpy(np.append(img_Din, label_Din, axis=2))  # 将label加入得到batch_size*1*17，并转为tensor类型
    img_Din = img_Din.to(torch.float32)  # 将double精度转为float适用于全连接层输入类型
    # print(img_Din.shape)

    return img_Din

# todo : no use : delete it
def Combine_data(data, label):  # 直接对tensor类型进行处理，这样可以保存反传的梯度，将处理后图片与经过G得到的类别组合成可以输入D的数据
    pic_len = data.shape[1] * data.shape[2] * data.shape[3]  # 获取每一张图片压缩后的总像素个数
    img_Din = data.squeeze().reshape((data.shape[0], 1, pic_len))  # 变为batch_size*1*len
    # label_Din = label.cpu().unsqueeze(-1).unsqueeze(-1).numpy()   # 获得对应label
    label_Din = label.cpu().unsqueeze(-2)  # 获得对应label,对于增添10各类别概率仅需要加一个维度即可
    img_Din = torch.cat((img_Din, label_Din), 2)  # 将label余图像直接tensor合并，得到batch_size*1*(len+10)，主要为了是的tensor能够用保留反传梯度
    # img_Din = torch.from_numpy(np.append(img_Din, label_Din, axis=2)) # 将label加入得到batch_size*1*(len+10)
    # img_Din = img_Din.to(torch.float32) # 将double精度转为float适用于全连接层输入类型

    return


def cal_conv2d_output_shape(h_in, w_in, conv2d_obj):
    if not isinstance(conv2d_obj, nn.Conv2d):
        raise TypeError
    padding = _pair(conv2d_obj.padding)
    dilation = _pair(conv2d_obj.dilation)
    kernel_size = _pair(conv2d_obj.kernel_size)
    stride = _pair(conv2d_obj.stride)
    h_out = math.floor((h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    w_out = math.floor((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    return h_out, w_out


def cal_max_pool2d_output_shape(h_in, w_in, max_pool2d_obj):
    if not isinstance(max_pool2d_obj, nn.MaxPool2d):
        raise TypeError
    padding = _pair(max_pool2d_obj.padding)
    dilation = _pair(max_pool2d_obj.dilation)
    kernel_size = _pair(max_pool2d_obj.kernel_size)
    stride = _pair(max_pool2d_obj.stride)
    h_out = math.floor((h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    w_out = math.floor((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    return h_out, w_out
