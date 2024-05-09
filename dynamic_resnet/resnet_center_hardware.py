import torch
import torchvision
import torchvision.models
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch
import torchvision.datasets as dset
import torch.nn.functional as F
from PIL import Image
import glob
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Hardtanh, BatchNorm1d as BN
from torch.nn.modules.utils import _single
from torch.autograd import Function
from torch.nn import Parameter
import math
from torch.nn import functional
from torch.nn.modules.utils import _single, _pair, _triple
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl



weight = torch.load('C:\\Users\\zy\\Desktop\\Dynamic\\hardware_new_0.01.pt')
semantic_center = torch.load('C:\\Users\\zy\\Desktop\\Dynamic\\weight and center\\mnist\\semantic_center_mnist.pth')
semantic_center_1 = weight[27][3001]
semantic_center_2 = weight[28][3002]
semantic_center_3 = weight[29][3003]
semantic_center_4 = weight[30][3004]
semantic_center_5 = weight[31][3005]
semantic_center_6 = weight[32][3006]
semantic_center_7 = weight[33][3007]
semantic_center_8 = weight[34][3008]
semantic_center_9 = weight[35][3009]
semantic_center_10 = weight[36][3010]


def binarization(input):
    # Binarize matrices
    # input: 2D matrix to be binarized

    # Column-wise min/max of input
    input_min = np.min(input,axis=0)
    input_max = np.max(input,axis=0)

    n_levels = 2 ** 1 - 1

    # Scaling coefficient (see output)
    a = (input_max - input_min) / n_levels
    b = input_min

    # input_min = input_min.repeat(input.shape[0], 1)
    # input_max = input_max.repeat(input.shape[0], 1)

    # Cast to integers
    # note that, though the first line would result in nan value if max == min,
    # the round function will return 0 for nan values.
    input_int = (input - input_min) / (input_max - input_min) * n_levels
    input_int = np.around(input_int)
    input_int = input_int.astype("int32")
    # Binarize
#     input_b = input_int.unsqueeze(-1).bitwise_and(1).ne(0).byte()
    return input_int

def pad(feature, num):
    b1, c1, m1, n1 = feature.shape
    new_m = m1 + 2 * num
    new_n = n1 + 2 * num
    new_LH = (b1, c1, int(new_m), int(new_n))
    new_map = np.zeros(new_LH)
    new_map[:, num:num + m1, num:num + n1] = feature
    return new_map

class Conv:
    def __init__(self, weight, layer_num, num_filters, in_channel, kernel_h, kernel_w, padding, stride):
        self.num_filters = num_filters
        self.in_channel = in_channel
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.padding = padding
        self.stride = stride
        self.weight = weight[layer_num]

    def filters(self):
        h,w = self.weight.shape
        true_weight = self.weight.reshape(h, self.num_filters, self.in_channel, self.kernel_h ,self.kernel_w)
        return true_weight

    # def iterate_regions(self,image):
    #     image = self.pad(image, self.padding)
    #     h,w = image.shape
    #     for i in range (h-self.kernel_h+1):
    #         for j in range(w-self.kernel_w+1):
    #             im_region = image[i:(i+self.kernel_h),j:(j+self.kernel_w)]
    #             yield im_region, i ,j

    def forward(self, input):
        image = pad(input, self.padding)
        b, c, h, w = image.shape
        new_h = (h - self.kernel_h)//self.stride + 1
        new_w = (w - self.kernel_w)//self.stride + 1
        output = np.zeros((b, self.num_filters, new_h, new_w))

        for i in range(0, h - self.kernel_h+1, self.stride):
            for j in range(0, w - self.kernel_w+1, self.stride):
                im_region = image[:, :, i:(i + self.kernel_h), j:(j + self.kernel_w)]
                # yield im_region, i, j
                one_filter = self.filters()[j]
                output[:, :, i//self.stride, j//self.stride] = np.sum(im_region * one_filter, axis=(1,2,3))
                # output[i//self.stride, j//self.stride] = np.sum(output[i//self.stride, j//self.stride])
        return output

# def Bn_zi(x,):


class BasicBlock1:
    def __init__(self,channels):
        super(BasicBlock1, self).__init__()
        self.channels = channels
        # self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
        #                        kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv1 = Conv(weight, 4, num_filters=channels, in_channel=channels, kernel_h =3, kernel_w=3, padding=1, stride=1).forward()
        # self.bn1 = nn.BatchNorm2d(channels)
        self.relu = hardware_relu
        # self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
        #                        kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv2 = Conv(weight, 5, num_filters=channels, in_channel=channels, kernel_h =3, kernel_w=3, padding=1, stride=1).forward()
        # self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        out_1 = self.conv1 = Conv(weight, 4, num_filters=self.channels, in_channel=self.channels, kernel_h =3, kernel_w=3, padding=1, stride=1).forward(x)
        # out_1 = self.bn1(out_1)
        out_1 = self.relu(out_1)

        out_2 = self.conv2 = Conv(weight, 5, num_filters=self.channels, in_channel=self.channels, kernel_h =3, kernel_w=3, padding=1, stride=1).forward(out_1)
        # out_2 = self.bn2(out_2)
        out = self.relu(out_1 + out_2)
        return out

class BasicBlock2:
    def __init__(self, channels):
        super(BasicBlock2, self).__init__()
        self.channels = channels
        # self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
        #                        kernel_size=3, stride=1, padding=1, bias=False)

        # self.bn1 = nn.BatchNorm2d(channels)
        self.relu = hardware_relu
        # self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
        #                        kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        out_1 = Conv(weight, 6, num_filters=self.channels, in_channel=self.channels, kernel_h=3, kernel_w=3, padding=1,
                          stride=1).forward(x)
        # out_1 = self.bn1(out_1)
        out_1 = self.relu(out_1)

        out_2 = Conv(weight, 7, num_filters=self.channels, in_channel=self.channels, kernel_h=3, kernel_w=3, padding=1,
                          stride=1).forward(out_1)
        # out_2 = self.bn2(out_2)
        out = self.relu(out_1 + out_2)
        return out

class BasicBlock3:
    def __init__(self, channels):
        super(BasicBlock3, self).__init__()
        self.channels = channels
        # self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
        #                        kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.relu = hardware_relu
        # self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
        #                        kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        out_1 = Conv(weight, 8, num_filters=self.channels, in_channel=self.channels, kernel_h=3, kernel_w=3, padding=1,
                          stride=1).forward(x)
        # out_1 = self.bn1(out_1)
        out_1 = self.relu(out_1)

        out_2 = Conv(weight, 9, num_filters=self.channels, in_channel=self.channels, kernel_h=3, kernel_w=3, padding=1,
                          stride=1).forward(out_1)
        # out_2 = self.bn2(out_2)
        out = self.relu(out_1 + out_2)
        return out

class BasicBlock4:
    def __init__(self, channels):
        super(BasicBlock4, self).__init__()
        self.channels = channels
        # self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
        #                        kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.relu = hardware_relu
        # self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
        #                        kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        out_1 = Conv(weight, 10, num_filters=self.channels, in_channel=self.channels, kernel_h=3, kernel_w=3, padding=1,
                          stride=1).forward(x)
        # out_1 = self.bn1(out_1)
        out_1 = self.relu(out_1)

        out_2 = Conv(weight, 11, num_filters=self.channels, in_channel=self.channels, kernel_h=3, kernel_w=3, padding=1,
                          stride=1).forward(out_1)
        # out_2 = self.bn2(out_2)
        out = self.relu(out_1 + out_2)
        return out

class BasicBlock5:
    def __init__(self, channels):
        super(BasicBlock5, self).__init__()
        self.channels = channels
        # self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
        #                        kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.relu = hardware_relu
        # self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
        #                        kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        out_1 = Conv(weight, 12, num_filters=self.channels, in_channel=self.channels, kernel_h=3, kernel_w=3, padding=1,
                          stride=1).forward(x)
        # out_1 = self.bn1(out_1)
        out_1 = self.relu(out_1)

        out_2 = Conv(weight, 13, num_filters=self.channels, in_channel=self.channels, kernel_h=3, kernel_w=3, padding=1,
                          stride=1).forward(out_1)
        # out_2 = self.bn2(out_2)
        out = self.relu(out_1 + out_2)
        return out

class BasicBlock6:
    def __init__(self, channels):
        super(BasicBlock6, self).__init__()
        self.channels = channels
        # self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
        #                        kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.relu = hardware_relu
        # self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
        #                        kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        out_1 = Conv(weight, 14, num_filters=self.channels, in_channel=self.channels, kernel_h=3, kernel_w=3, padding=1,
                          stride=1).forward(x)
        # out_1 = self.bn1(out_1)
        out_1 = self.relu(out_1)

        out_2 = Conv(weight, 15, num_filters=self.channels, in_channel=self.channels, kernel_h=3, kernel_w=3, padding=1,
                          stride=1).forward(out_1)
        # out_2 = self.bn2(out_2)
        out = self.relu(out_1 + out_2)
        return out

class BasicBlock7:
    def __init__(self, channels):
        super(BasicBlock7, self).__init__()
        self.channels = channels
        # self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
        #                        kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.relu = hardware_relu
        # self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
        #                        kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        out_1 = Conv(weight, 16, num_filters=self.channels, in_channel=self.channels, kernel_h=3, kernel_w=3, padding=1,
                          stride=1).forward(x)
        # out_1 = self.bn1(out_1)
        out_1 = self.relu(out_1)

        out_2 = Conv(weight, 17, num_filters=self.channels, in_channel=self.channels, kernel_h=3, kernel_w=3, padding=1,
                          stride=1).forward(out_1)
        # out_2 = self.bn2(out_2)
        out = self.relu(out_1 + out_2)
        return out

class BasicBlock8:
    def __init__(self, channels):
        super(BasicBlock8, self).__init__()
        self.channels = channels
        # self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
        #                        kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.relu = hardware_relu
        # self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
        #                        kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        out_1 = Conv(weight, 18, num_filters=self.channels, in_channel=self.channels, kernel_h=3, kernel_w=3, padding=1,
                          stride=1).forward(x)
        # out_1 = self.bn1(out_1)
        out_1 = self.relu(out_1)

        out_2 = Conv(weight, 19, num_filters=self.channels, in_channel=self.channels, kernel_h=3, kernel_w=3, padding=1,
                          stride=1).forward(out_1)
        # out_2 = self.bn2(out_2)
        out = self.relu(out_1 + out_2)
        return out

class BasicBlock9:
    def __init__(self, channels):
        super(BasicBlock9, self).__init__()
        self.channels = channels
        # self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
        #                        kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.relu = hardware_relu
        # self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
        #                        kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        out_1 = Conv(weight, 20, num_filters=self.channels, in_channel=self.channels, kernel_h=3, kernel_w=3, padding=1,
                          stride=1).forward(x)
        # out_1 = self.bn1(out_1)
        out_1 = self.relu(out_1)

        out_2 = Conv(weight, 21, num_filters=self.channels, in_channel=self.channels, kernel_h=3, kernel_w=3, padding=1,
                          stride=1).forward(out_1)
        # out_2 = self.bn2(out_2)
        out = self.relu(out_1+out_2)
        return out

class BasicBlock10:
    def __init__(self, channels):
        super(BasicBlock10, self).__init__()
        self.channels = channels
        # self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
        #                        kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.relu = hardware_relu
        # self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
        #                        kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        out_1 = Conv(weight, 22, num_filters=self.channels, in_channel=self.channels, kernel_h=3, kernel_w=3, padding=1,
                          stride=1).forward(x)
        # out_1 = self.bn1(out_1)
        out_1 = self.relu(out_1)

        out_2 = Conv(weight, 23, num_filters=self.channels, in_channel=self.channels, kernel_h=3, kernel_w=3, padding=1,
                          stride=1).forward(out_1)
        # out_2 = self.bn2(out_2)
        out = self.relu(out_1+out_2)
        return out

class BasicBlock11:
    def __init__(self, channels):
        super(BasicBlock11, self).__init__()
        self.channels = channels
        # self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
        #                        kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.relu = hardware_relu
        # self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
        #                        kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        out_1 = Conv(weight, 24, num_filters=self.channels, in_channel=self.channels, kernel_h=3, kernel_w=3, padding=1,
                          stride=1).forward(x)
        # out_1 = self.bn1(out_1)
        out_1 = self.relu(out_1)

        out_2 = Conv(weight, 25, num_filters=self.channels, in_channel=self.channels, kernel_h=3, kernel_w=3, padding=1,
                          stride=1).forward(out_1)
        # out_2 = self.bn2(out_2)
        out = self.relu(out_1+out_2)
        return out

class hardware_linear:
    def __init__(self,weight, layer_num, data_in,data_out):
        self.weight = weight[layer_num]
        self.data_in = data_in
        self.data_out = data_out

    def filters(self):
        h,w = self.weight.shape
        true_weight = self.weight.reshape(h, self.data_in, self.data_out)
        return true_weight

    def forward(self,x):
        output = x.dot(self.filters()[2000])
        return output

class maxpool_hardware:
    def __init__(self, kernel_h, kernel_w, padding, stride):
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.padding = padding
        self.stride = stride

    def forward(self, input):
        image = pad(input, self.padding)
        b, c, h, w = image.shape
        new_h = (h - self.kernel_h)//self.stride + 1
        new_w = (w - self.kernel_w)//self.stride + 1
        output = np.zeros((b, c, new_h, new_w))

        for i in range(0, h - self.kernel_h+1, self.stride):
            for j in range(0, w - self.kernel_w+1, self.stride):
                im_region = image[:, :, i:(i + self.kernel_h), j:(j + self.kernel_w)]
                output[:, i//self.stride, j//self.stride] = np.amax(im_region, axis=(2,3))
                # output[i//self.stride, j//self.stride] = np.sum(output[i//self.stride, j//self.stride])
        return output

def hardware_relu(x):
    output = np.where(x>0, x, 0)
    return output

def global_avg_pooling_forward(z):
    """
    全局平均池化前向过程
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :return:
    """
    return np.mean(np.mean(z, axis=-1), axis=-1)

def get_cos_similar(v1: list, v2: list):
    num = float(np.dot(v1, v2))  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0

def AC_calculate(SA):
    SA = sorted(SA)
    SA_h = SA[-1]
    SA_sh = SA[-2]
    AC = (SA_h[0] - SA_sh[0]) / SA_sh[0]
    label = SA_h[1]
    return AC, label

class ResNet:
    def __init__(self):
        # super(ResNet, self).__init__()
        self.in_channel = 3
        # self.conv1 = nn.Conv2d(1, self.in_channel, kernel_size=7, stride=2,
        #                        padding=3, bias=False)
        self.conv1 = Conv(weight, 0, 3, 1, 7, 7, padding=3, stride=2)
        # self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = hardware_relu
        self.maxpool = maxpool_hardware(3, 3, padding=1, stride=2)

        self.conv2 = Conv(weight, 1, 6, 3, 1, 1, padding=0, stride=2)
        # self.bn2 = nn.BatchNorm2d(6)

        self.conv3 = Conv(weight, 2, 12, 6, 1, 1, padding=0, stride=2)
        # self.bn3 = nn.BatchNorm2d(12)

        self.conv4 = Conv(weight, 3, 24, 12, 1, 1, padding=0, stride=2)
        # self.bn4 = nn.BatchNorm2d(24)

        self.layer1 = BasicBlock1(3)
        self.layer2 = BasicBlock2(6)
        self.layer3 = BasicBlock3(12)
        self.layer4 = BasicBlock4(24)
        self.layer5 = BasicBlock5(24)
        self.layer6 = BasicBlock6(24)
        self.layer7 = BasicBlock7(24)
        self.layer8 = BasicBlock8(24)
        self.layer9 = BasicBlock9(24)
        self.layer10 = BasicBlock10(24)
        self.layer11 = BasicBlock11(24)
        self.avgpool = global_avg_pooling_forward  # output size = (1, 1)
        # self.fc = nn.Linear(24 * block.expansion, num_classes)
        self.fc = hardware_linear(weight, 26, 24, 10)
        # self.fc = nn.Linear(128, num_classes)

    def classifier(self, sv, semantic_center, layer_id):
        sim_1 = []
        sv = sv.reshape(-1)
        for i in range(len(semantic_center)):
            semantic_center_j_1 = semantic_center[i][0][layer_id].reshape(-1)
            semantic_center_j_1 = semantic_center_j_1.cpu().numpy()
            similarity = get_cos_similar(semantic_center_j_1, sv)
            # similarity = 2**layer_id * similarity
            label = semantic_center[i][1]
            label = label.cpu().numpy()
            sim_1.append((similarity, label))
        # confidence_label_1 = Similarity_calculate(sim_1)
        # confidence_1 = confidence_label_1[0]
        # label = confidence_label_1[1]
        return sim_1

    def calculate(self, sim_1, sim_2):
        sim_final = []
        for i in range(len(sim_1)):
            sim = sim_1[i][0] + sim_2[i][0]
            label = sim_1[i][1]
            sim_final.append((sim, label))
        return sim_final

    def forward(self, x):
        x = binarization(x)
        # x = self.conv1.forward(x)
        # print(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool.forward(x)
        x1 = self.layer1.forward(binarization(x))
        # x1 = self.layer1.forward(x)
        sv1 = global_avg_pooling_forward(x1)

        x2 = self.conv2.forward(binarization(x1))
        # x2 = self.conv2.forward(x1)
        # x2 = self.bn2(x2)
        x2 = self.layer2.forward(binarization(x2))
        # x2 = self.layer2.forward(x2)
        sv2 = global_avg_pooling_forward(x2)

        x3 = self.conv3.forward(binarization(x2))
        # x3 = self.conv3.forward(x2)
        # x3 = self.bn3(x3)
        x3 = self.layer3.forward(binarization(x3))
        # x3 = self.layer3.forward(x3)
        sv3 = global_avg_pooling_forward(x3)

        x4 = self.conv4.forward(binarization(x3))
        # x4 = self.conv4.forward(x3)
        # x4 = self.bn4(x4)
        x4 = self.layer4.forward(binarization(x4))
        # x4 = self.layer4.forward(x4)
        sv4 = global_avg_pooling_forward(x4)

        x5 = self.layer5.forward(binarization(x4))
        # x5 = self.layer5.forward(x4)
        sv5 = global_avg_pooling_forward(x5)

        x6 = self.layer6.forward(binarization(x5))
        # x6 = self.layer6.forward(x5)
        sv6 = global_avg_pooling_forward(x6)

        x7 = self.layer7.forward(binarization(x6))
        # x7 = self.layer7.forward(x6)
        sv7 = global_avg_pooling_forward(x7)

        x8 = self.layer8.forward(binarization(x7))
        # x8 = self.layer8.forward(x7)
        sv8 = global_avg_pooling_forward(x8)

        x9 = self.layer9.forward(binarization(x8))
        # x9 = self.layer9.forward(x8)
        sv9 = global_avg_pooling_forward(x9)

        x10 = self.layer10.forward(binarization(x9))
        # x10 = self.layer10.forward(x9)
        sv10 = global_avg_pooling_forward(x10)

        x11 = self.layer11.forward(binarization(x10))
        # x11 = self.layer11.forward(x10)
        sv11 = global_avg_pooling_forward(x11)

        output = self.avgpool(binarization(x11))
        # output = self.avgpool(x11)
        output = output.flatten()
        output = self.fc.forward(output)

        return sv1, sv2, sv3, sv4, sv5, sv6, sv7, sv8, sv9, sv10, sv11



device = torch.device("cpu")
print("using {} device.".format(device))



# 测试所保存的模型

# test1 = torch.ones(64, 3, 120, 120)  # 测试一下输出的形状大小 输入一个64,3,120,120的向量
#
# test1 = new_m(test1.to(device))  # 将向量打入神经网络进行测试
# print(test1)  # 查看输出的结果

mytransforms = transforms.Compose([
    transforms.Grayscale(1),
    # transforms.Resize((7, 7)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307),(0.3081))])

# 加载数据集1

dataset = dset.ImageFolder('C:\\Users\\zy\\Desktop\\Dynamic\\mnist\\train',transform=mytransforms)
data_loader = DataLoader(dataset, batch_size=5000, shuffle=False)

semantic_center_final = []

semantic_center_epoch =[]
semantic_center_real =[]
item_1 = []

for data, label in data_loader:
    images = data.to(device)
    images = images.numpy()
    labels = label.to(device)
    labels = labels.numpy()
    sc = ResNet().forward(images)
    # print(sc[1].shape)
    semantic_center = []
    for i, sc_j in enumerate(sc):
        sc_real=np.mean(sc_j,dim=0)
        print(sc_real.shape)
        semantic_center.append(sc_real)
        print(len(semantic_center))

    label = labels[0]
    semantic_center_real.append((semantic_center,label))


torch.save(semantic_center_real, 'ResNet_semantic_center_ternary_mnist_binaryhardware.pth')
print('save sc')

semantic_center = torch.load('ResNet_semantic_center_ternary_mnist_binaryhardware.pth')
print(len(semantic_center))

# dataframe = pd.DataFrame({'semantic_center': semantic_center})
# dataframe.to_csv('semantic_centeer.csv', index=False, sep=',')

# similarity = torch.cosine_similarity(semantic_center[0][0][3],semantic_center[1][0][3],dim=0)
# print(similarity)