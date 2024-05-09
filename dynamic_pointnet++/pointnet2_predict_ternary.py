import argparse
import numpy as np
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.pointnet2_cls import pointnet2_cls_ssg, pointnet2_cls_msg, cls_loss
from data.ModelNet40 import ModelNet40
import tqdm
import torch
import torch.nn as nn
from torchsummary import summary
from torch.nn import Conv2d
from einops.layers.torch import Rearrange, Reduce
# from tensorboardX import SummaryWriter
from torchvision.transforms import transforms
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
import torchvision.models
import torchvision.datasets as dset
import numpy as np # linear algebra
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import numpy as np
import math
import random
import os
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pathlib import Path
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn import Parameter
from torch.autograd import Function
from hyperopt import fmin, tpe, hp,Trials


def ball_query(xyz, new_xyz, radius, K):
    '''

    :param xyz: shape=(B, N, 3)
    :param new_xyz: shape=(B, M, 3)
    :param radius: int
    :param K: int, an upper limit samples
    :return: shape=(B, M, K)
    '''
    device = xyz.device
    B, N, C = xyz.shape
    M = new_xyz.shape[1]
    grouped_inds = torch.arange(0, N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, M, 1)
    dists = get_dists(new_xyz, xyz)
    grouped_inds[dists > radius] = N
    grouped_inds = torch.sort(grouped_inds, dim=-1)[0][:, :, :K]
    grouped_min_inds = grouped_inds[:, :, 0:1].repeat(1, 1, K)
    grouped_inds[grouped_inds == N] = grouped_min_inds[grouped_inds == N]
    return grouped_inds

def get_dists(points1, points2):
    '''
    Calculate dists between two group points
    :param cur_point: shape=(B, M, C)
    :param points: shape=(B, N, C)
    :return:
    '''
    B, M, C = points1.shape
    _, N, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, M, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, N)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists) # Very Important for dist = 0.
    return torch.sqrt(dists).float()


def gather_points(points, inds):
    '''

    :param points: shape=(B, N, C)
    :param inds: shape=(B, M) or shape=(B, M, K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])
    repeat_shape = list(inds.shape)
    repeat_shape[0] = 1
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
    return points[batchlists, inds, :]


def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


import torch
from utils.common import get_dists


def fps(xyz, M):
    '''
    Sample M points from points according to farthest point sampling (FPS) algorithm.
    :param xyz: shape=(B, N, 3)
    :return: inds: shape=(B, M)
    '''
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(size=(B, M), dtype=torch.long).to(device)
    dists = torch.ones(B, N).to(device) * 1e5
    inds = torch.randint(0, N, size=(B, ), dtype=torch.long).to(device)
    batchlists = torch.arange(0, B, dtype=torch.long).to(device)
    for i in range(M):
        centroids[:, i] = inds
        cur_point = xyz[batchlists, inds, :] # (B, 3)
        cur_dist = torch.squeeze(get_dists(torch.unsqueeze(cur_point, 1), xyz), dim=1)
        dists[cur_dist < dists] = cur_dist[cur_dist < dists]
        inds = torch.max(dists, dim=1)[1]
    return centroids

def sample_and_group(xyz, points, M, radius, K, use_xyz=True):
    '''
    :param xyz: shape=(B, N, 3)
    :param points: shape=(B, N, C)
    :param M: int
    :param radius:float
    :param K: int
    :param use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    :return: new_xyz, shape=(B, M, 3); new_points, shape=(B, M, K, C+3);
             group_inds, shape=(B, M, K); grouped_xyz, shape=(B, M, K, 3)
    '''
    new_xyz = gather_points(xyz, fps(xyz, M))
    grouped_inds = ball_query(xyz, new_xyz, radius, K)
    grouped_xyz = gather_points(xyz, grouped_inds)
    grouped_xyz -= torch.unsqueeze(new_xyz, 2).repeat(1, 1, K, 1)
    if points is not None:
        grouped_points = gather_points(points, grouped_inds)
        if use_xyz:
            new_points = torch.cat((grouped_xyz.float(), grouped_points.float()), dim=-1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, grouped_inds, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    '''

    :param xyz: shape=(B, M, 3)
    :param points: shape=(B, M, C)
    :param use_xyz:
    :return: new_xyz, shape=(B, 1, 3); new_points, shape=(B, 1, M, C+3);
             group_inds, shape=(B, 1, M); grouped_xyz, shape=(B, 1, M, 3)
    '''
    B, M, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C)
    grouped_inds = torch.arange(0, M).long().view(1, 1, M).repeat(B, 1, 1)
    grouped_xyz = xyz.view(B, 1, M, C)
    if points is not None:
        if use_xyz:
            new_points = torch.cat([xyz.float(), points.float()], dim=2)
        else:
            new_points = points
        new_points = torch.unsqueeze(new_points, dim=1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, grouped_inds, grouped_xyz


class PointNet_SA_Module(nn.Module):
    def __init__(self, M, radius, K, in_channels, mlp, group_all, bn=True, pooling='max', use_xyz=True):
        super(PointNet_SA_Module, self).__init__()
        self.M = M
        self.radius = radius
        self.K = K
        self.in_channels = in_channels
        self.mlp = mlp
        self.group_all = group_all
        self.bn = bn
        self.pooling = pooling
        self.use_xyz = use_xyz
        self.backbone = nn.Sequential()
        for i, out_channels in enumerate(mlp):
            self.backbone.add_module('Conv{}'.format(i),
                                     nn.Conv2d(in_channels, out_channels, 1,
                                               stride=1, padding=0, bias=False))
            if bn:
                self.backbone.add_module('Bn{}'.format(i),
                                         nn.BatchNorm2d(out_channels))
            self.backbone.add_module('Relu{}'.format(i), nn.ReLU())
            in_channels = out_channels
    def forward(self, xyz, points):
        if self.group_all:
            new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group(xyz=xyz,
                                                                              points=points,
                                                                              M=self.M,
                                                                              radius=self.radius,
                                                                              K=self.K,
                                                                              use_xyz=self.use_xyz)
        new_points = self.backbone(new_points.permute(0, 3, 2, 1).contiguous())
        if self.pooling == 'avg':
            new_points = torch.mean(new_points, dim=2)
        else:
            new_points = torch.max(new_points, dim=2)[0]
        new_points = new_points.permute(0, 2, 1).contiguous()
        return new_xyz, new_points


class PointNet_SA_Module_MSG(nn.Module):
    def __init__(self, M, radiuses, Ks, in_channels, mlps, bn=True, pooling='max', use_xyz=True):
        super(PointNet_SA_Module_MSG, self).__init__()
        self.M = M
        self.radiuses = radiuses
        self.Ks = Ks
        self.in_channels = in_channels
        self.mlps = mlps
        self.bn = bn
        self.pooling = pooling
        self.use_xyz = use_xyz
        self.backbones = nn.ModuleList()
        for j in range(len(mlps)):
            mlp = mlps[j]
            backbone = nn.Sequential()
            in_channels = self.in_channels
            for i, out_channels in enumerate(mlp):
                backbone.add_module('Conv{}_{}'.format(j, i),
                                         nn.Conv2d(in_channels, out_channels, 1,
                                                   stride=1, padding=0, bias=False))
                if bn:
                    backbone.add_module('Bn{}_{}'.format(j, i),
                                             nn.BatchNorm2d(out_channels))
                backbone.add_module('Relu{}_{}'.format(j, i), nn.ReLU())
                in_channels = out_channels
            self.backbones.append(backbone)

    def forward(self, xyz, points):
        new_xyz = gather_points(xyz, fps(xyz, self.M))
        new_points_all = []
        for i in range(len(self.mlps)):
            radius = self.radiuses[i]
            K = self.Ks[i]
            grouped_inds = ball_query(xyz, new_xyz, radius, K)
            grouped_xyz = gather_points(xyz, grouped_inds)
            grouped_xyz -= torch.unsqueeze(new_xyz, 2).repeat(1, 1, K, 1)
            if points is not None:
                grouped_points = gather_points(points, grouped_inds)
                if self.use_xyz:
                    new_points = torch.cat(
                        (grouped_xyz.float(), grouped_points.float()),
                        dim=-1)
                else:
                    new_points = grouped_points
            else:
                new_points = grouped_xyz
            new_points = self.backbones[i](new_points.permute(0, 3, 2, 1).contiguous())
            if self.pooling == 'avg':
                new_points = torch.mean(new_points, dim=2)
            else:
                new_points = torch.max(new_points, dim=2)[0]
            new_points = new_points.permute(0, 2, 1).contiguous()
            new_points_all.append(new_points)
        return new_xyz, torch.cat(new_points_all, dim=-1)


def AC_calculate(SA):
    SA = sorted(SA)
    SA_h = SA[-1]
    SA_sh = SA[-2]
    AC = (SA_h[0] - SA_sh[0]) / SA_sh[0]
    label = SA_h[1]
    return AC, label

# ternary quantization
#----------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------
def gen_noise(weight, noise):
    new_w = weight * noise * torch.randn_like(weight)
    return new_w.to(weight.device)

class TernaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # get output
        ctx_max, ctx_min = torch.max(input), torch.min(input)
        lower_interval = ctx_min + (ctx_max - ctx_min) / 3
        higher_interval = ctx_max - (ctx_max - ctx_min) / 3
        out = torch.where(input < lower_interval, torch.tensor(-1.).to(input.device, input.dtype), input)
        out = torch.where(input > higher_interval, torch.tensor(1.).to(input.device, input.dtype), out)
        out = torch.where((input >= lower_interval) & (input <= higher_interval), torch.tensor(0.).to(input.device, input.dtype), out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output
        grad_input[input[0].gt(1)] = 0
        grad_input[input[0].lt(-1)] = 0
        return grad_input


class TriConv2d(torch.nn.modules.conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(TriConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        # self.weight = Parameter(torch.Tensor(out_channels, in_channels//groups, *kernel_size),requires_grad=True)

    def forward(self, input):

        bw = self.weight
        ba = input
        bw = bw - bw.mean()
        bw = TernaryQuantize().apply(bw)
        self.weight.data = bw
        # ba = TernaryQuantize().apply(ba)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(ba, expanded_padding, mode='circular'),
                            bw, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(ba, bw, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    # def forward(self,input):
    #     return(self.conv2d_forward(input,self.weight))

class TriLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, binary_act=True):
        super(TriLinear, self).__init__(in_features, out_features, bias=bias)
        self.output_ = None
        # self.weight = Parameter(torch.Tensor(out_features, in_features),requires_grad=True)

    def forward(self, input):
        tw = self.weight
        ta = input
        tw = TernaryQuantize().apply(tw)
        output = F.linear(ta, tw, self.bias)
        self.output_ = output
        self.weight.data = tw
        return output


class NoisedTriLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, noise=0):
        super(NoisedTriLinear, self).__init__(in_features, out_features, bias=bias)
        self.output_ = None
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.out_features = out_features
        self.noise = noise

    def forward(self, input):
        tw = self.weight
        ta = input
        tw = TernaryQuantize().apply(tw)
        if not self.noise:
            output = F.linear(ta, tw, self.bias)
        else:
            output = F.linear(ta, tw, self.bias) + self.noised_forward(input, tw)
        self.output_ = output
        return output

    def noised_forward(self, x, w):
        batch_size, n_points = x.size()
        origin_weight = w
        x_new = torch.zeros(batch_size, self.out_features).to(x.device)

        for i in range(batch_size):
            noise_weight = gen_noise(origin_weight, self.noise).detach()
            x_i = F.linear(x[i, :], noise_weight, self.bias)
            x_new[i, :] = x_i.squeeze()
            del noise_weight, x_i

        return x_new
    
class PointNet_SA_Module_ternary(nn.Module):
    def __init__(self, M, radius, K, in_channels, mlp, group_all, bn=True, pooling='max', use_xyz=True):
        super(PointNet_SA_Module_ternary, self).__init__()
        self.M = M
        self.radius = radius
        self.K = K
        self.in_channels = in_channels
        self.mlp = mlp
        self.group_all = group_all
        self.bn = bn
        self.pooling = pooling
        self.use_xyz = use_xyz
        self.backbone = nn.Sequential()
        for i, out_channels in enumerate(mlp):
            self.backbone.add_module('Conv{}'.format(i),
                                     TriConv2d(in_channels, out_channels, 1,
                                               stride=1, padding=0, bias=False))
            if bn:
                self.backbone.add_module('Bn{}'.format(i),
                                         nn.BatchNorm2d(out_channels))
            self.backbone.add_module('Relu{}'.format(i), nn.ReLU())
            in_channels = out_channels
    def forward(self, xyz, points):
        if self.group_all:
            new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group(xyz=xyz,
                                                                              points=points,
                                                                              M=self.M,
                                                                              radius=self.radius,
                                                                              K=self.K,
                                                                              use_xyz=self.use_xyz)
        new_points = self.backbone(new_points.permute(0, 3, 2, 1).contiguous())
        if self.pooling == 'avg':
            new_points = torch.mean(new_points, dim=2)
        else:
            new_points = torch.max(new_points, dim=2)[0]
        new_points = new_points.permute(0, 2, 1).contiguous()
        return new_xyz, new_points
#----------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------


class pointnet2_cls_ssg(nn.Module):
    def __init__(self, in_channels, nclasses,confidence_threshold = [0.03, 0.03, 0.03, 0.03, 0.03, 0.01, 0.1, 0.005]):
        super(pointnet2_cls_ssg, self).__init__()
        self.pt_sa1 = PointNet_SA_Module_ternary(M=512, radius=0.2, K=32, in_channels=in_channels, mlp=[64, 64, 128], group_all=False)
        self.pt_sa2 = PointNet_SA_Module_ternary(M=512, radius=0.2, K=32, in_channels=131, mlp=[64, 64, 128], group_all=False)
        self.pt_sa3 = PointNet_SA_Module_ternary(M=512, radius=0.2, K=32, in_channels=131, mlp=[64, 64, 128], group_all=False)
        self.pt_sa4 = PointNet_SA_Module_ternary(M=512, radius=0.2, K=32, in_channels=131, mlp=[64, 64, 128], group_all=False)
        self.pt_sa5 = PointNet_SA_Module_ternary(M=512, radius=0.2, K=32, in_channels=131, mlp=[128, 128, 256], group_all=False)
        self.pt_sa6 = PointNet_SA_Module_ternary(M=128, radius=0.4, K=64, in_channels=259, mlp=[128, 128, 256], group_all=False)
        self.pt_sa7 = PointNet_SA_Module_ternary(M=128, radius=0.4, K=64, in_channels=259, mlp=[128, 128, 256], group_all=False)
        self.pt_sa8 = PointNet_SA_Module_ternary(M=None, radius=None, K=None, in_channels=259, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = TriLinear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = TriLinear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.cls = TriLinear(256, nclasses)
        self.confidence_threshold = confidence_threshold

    def classifier(self, sv, semantic_center, layer_id):
        sim_1 = []
        sv = sv.reshape(-1)
        for i in range(len(semantic_center)):
            semantic_center_j_1 = semantic_center[i][0][layer_id].reshape(-1)
            similarity = torch.cosine_similarity(semantic_center_j_1, sv, dim=0)
            # similarity = 2**layer_id * similarity
            label = semantic_center[i][1]
            sim_1.append((similarity, label))
        # confidence_label_1 = Similarity_calculate(sim_1)
        # confidence_1 = confidence_label_1[0]
        # label = confidence_label_1[1]
        return sim_1

    def calculate(self, sim_1, sim_2):
        sim_final=[]
        for i in range(len(sim_1)):
            sim = sim_1[i][0]+sim_2[i][0]
            label = sim_1[i][1]
            sim_final.append((sim,label))
        return sim_final

    def forward(self, xyz, points):

        confidence_threshold = self.confidence_threshold

        batchsize = xyz.shape[0]
        new_xyz, new_points = self.pt_sa1(xyz, points)
        # net_1 = new_points.view(batchsize, -1)
        net_1 = new_points.mean(dim=1)
        sim_1 = self.classifier(net_1, semantic_center, 0)
        confidence_1 = AC_calculate(sim_1)
        # if confidence_1[0] >= confidence_threshold:
        if confidence_1[0] >= confidence_threshold[0]:
            is_early = True
            return 0, confidence_1[1]
        else:
            new_xyz, new_points = self.pt_sa2(new_xyz, new_points)
            # net_2 = new_points.view(batchsize, -1)
            net_2 = new_points.mean(dim=1)
            sim_2 = self.classifier(net_2, semantic_center, 1)
            # sim_2 = self.calculate(sim_1, sim_2)
            confidence_2 = AC_calculate(sim_2)
            if confidence_2[0] >= confidence_threshold[1]:
                is_early = True
                return 1, confidence_2[1]
            else:
                new_xyz, new_points = self.pt_sa3(new_xyz, new_points)
                # net_3 = new_points.view(batchsize, -1)
                net_3 = new_points.mean(dim=1)
                sim_3 = self.classifier(net_3, semantic_center, 2)
                # sim_3 = self.calculate(sim_2, sim_3)
                confidence_3 = AC_calculate(sim_3)
                if confidence_3[0] >= confidence_threshold[2]:
                    is_early = True
                    return 2, confidence_3[1]
                else:
                    new_xyz, new_points = self.pt_sa4(new_xyz, new_points)
                    # net_3 = new_points.view(batchsize, -1)
                    net_4 = new_points.mean(dim=1)
                    sim_4 = self.classifier(net_4, semantic_center, 3)
                    # sim_4 = self.calculate(sim_3, sim_4)
                    confidence_4 = AC_calculate(sim_4)
                    if confidence_4[0] >= confidence_threshold[3]:
                        is_early = True
                        return 3, confidence_4[1]
                    else:
                        new_xyz, new_points = self.pt_sa5(new_xyz, new_points)
                        # net_3 = new_points.view(batchsize, -1)
                        net_5 = new_points.mean(dim=1)
                        sim_5 = self.classifier(net_5, semantic_center, 4)
                        # sim_5 = self.calculate(sim_4, sim_5)
                        confidence_5 = AC_calculate(sim_5)
                        if confidence_5[0] >= confidence_threshold[4]:
                            is_early = True
                            return 4, confidence_5[1]
                        else:
                            new_xyz, new_points = self.pt_sa6(new_xyz, new_points)
                            # net_3 = new_points.view(batchsize, -1)
                            net_6 = new_points.mean(dim=1)
                            sim_6 = self.classifier(net_6, semantic_center, 5)
                            # sim_6 = self.calculate(sim_5, sim_6)
                            confidence_6 = AC_calculate(sim_6)
                            if confidence_6[0] >= confidence_threshold[5]:
                                is_early = True
                                return 5, confidence_6[1]
                            else:
                                new_xyz, new_points = self.pt_sa7(new_xyz, new_points)
                                # net_3 = new_points.view(batchsize, -1)
                                net_7 = new_points.mean(dim=1)
                                sim_7 = self.classifier(net_7, semantic_center, 6)
                                # sim_7 = self.calculate(sim_6, sim_7)
                                confidence_7 = AC_calculate(sim_7)
                                if confidence_7[0] >= confidence_threshold[6]:
                                    is_early = True
                                    return 6, confidence_7[1]
                                else:
                                    new_xyz, new_points = self.pt_sa8(new_xyz, new_points)
                                    # net_3 = new_points.view(batchsize, -1)
                                    net_8 = new_points.mean(dim=1)
                                    sim_8 = self.classifier(net_8, semantic_center, 7)
                                    # sim_8 = self.calculate(sim_7, sim_8)
                                    confidence_8 = AC_calculate(sim_8)
                                    if confidence_8[0] >= confidence_threshold[7]:
                                        is_early = True
                                        return 7, confidence_8[1]
                                    else:
                                        net = self.dropout1(F.relu(self.bn1(self.fc1(net_8))))
                                        net = self.dropout2(F.relu(self.bn2(self.fc2(net))))
                                        net = self.cls(net)
                                        output_data = torch.argmax(net, 1)
                                        return 8, output_data

def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, _ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces


class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t - s) * pt2[i] + (1 - t) * pt3[i]
        return (f(0), f(1), f(2))

    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))

        sampled_faces = (random.choices(faces,
                                        weights=areas,
                                        cum_weights=None,
                                        k=self.output_size))

        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))

        return sampled_points


class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return norm_pointcloud


class RandRotation_z(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[math.cos(theta), -math.sin(theta), 0],
                               [math.sin(theta), math.cos(theta), 0],
                               [0, 0, 1]])

        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud


class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))

        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud

class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        return torch.from_numpy(pointcloud)

def default_transforms():
    return transforms.Compose([
                                PointSampler(1024),
                                Normalize(),
                                ToTensor()
                              ])

class PointCloudData(Dataset):
    def __init__(self, root_dir, valid=False, folder="train", transform=default_transforms()):
        self.root_dir = root_dir
        folders = [directory for directory in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/directory)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.files = []
        for category in self.classes.keys():
            new_dir = root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    sample['pcd_path'] = new_dir/file
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file):
        verts, faces = read_off(file)
        if self.transforms:
            pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f)
        return {'pointcloud': pointcloud,
                'category': self.classes[category]}
    
train_transforms = transforms.Compose([
                    PointSampler(1024),
                    Normalize(),
                    RandRotation_z(),
                    RandomNoise(),
                    ToTensor()
                    ])

path = Path("ModelNet10")
folders = [directory for directory in sorted(os.listdir(path)) if os.path.isdir(path/directory)]
classes = {folder: i for i, folder in enumerate(folders)}
train_ds = PointCloudData(path, transform=train_transforms)
valid_ds = PointCloudData(path, valid=True, folder='test', transform=train_transforms)

inv_classes = {i: cat for cat, i in train_ds.classes.items()};

print('Train dataset size: ', len(train_ds))
print('Valid dataset size: ', len(valid_ds))
print(valid_ds)
# print('Number of classes: ', len(train_ds.classes))
print(valid_ds.classes)
# print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())
print( train_ds[1509])
print('Class: ', inv_classes[train_ds[0]['category']])




# train_loader = DataLoader(dataset=train_ds, batch_size=1, shuffle=True)
valid_loader = DataLoader(dataset=valid_ds, batch_size=1)
    
    # 加载数据集1
# train_data = torchvision.datasets.MNIST(root = "./mnist/train" , train = True ,download = True, transform=mytransforms)
# traindata = torch.utils.data.DataLoader(dataset= train_data , batch_size=64, shuffle=True)
# test_data = torchvision.datasets.MNIST(root = "./mnist/test" , train = False ,download = True, transform=mytransforms)
# testdata = torch.utils.data.DataLoader(dataset=test_data, batch_size=64, shuffle=False)
# traindata = dset.ImageFolder('/home/yue/Pointnet2.PyTorch/modelnet10_/train')
# train_data = DataLoader(traindata, batch_size=64, shuffle=False)

# testdata = dset.ImageFolder('/home/yue/Pointnet2.PyTorch/modelnet10_/test')
# test_data = DataLoader(testdata, batch_size=64, shuffle=False)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

semantic_center = torch.load('pointnet2_semantic_center_ternary_0.83_to_ternary_noise10.pth',map_location=device)

def train_model(params):
    epoch = 1  # 迭代次数即训练次数
    test_accur_all = []  # 存放测试集准确率的数组
    for i in range(epoch):  # 开始迭代
        test_accuracy = 0.0
        test_num = 0
        test_is_early_num = 0
        layer_id_amount = [0,1,2,3,4,5,6,7,8]
        confidence_out = []
        confidence_true = []
        model = pointnet2_cls_ssg(in_channels=3, nclasses=10,confidence_threshold=params)
        model.to(device)
        m_state_dict = torch.load('pointnet2_ternary_noise10.pth',map_location=device)
        model.load_state_dict(m_state_dict)
        # semantic_center = torch.load('pointnet2_semantic_center_ternary_0.83.pth')
        model.eval()  # 将模型调整为测试模型
        with torch.no_grad():  # 清空历史梯度，进行测试  与训练最大的区别是测试过程中取消了反向传播
            test_bar = tqdm(valid_loader)
            for data in test_bar:
                inputs, target = data['pointcloud'].to(device).float(), data['category'].to(device)
                xyz, points = inputs[:, :, :3], inputs[:, :, 3:]
                outputs = model(xyz.to(device), points.to(device))

                output_data = outputs[1]
                layer_id = outputs[0]
                layer_id_amount.append(layer_id)
                accuracy = torch.sum(output_data == target.to(device))
                test_accuracy = test_accuracy + accuracy
                test_num += xyz.shape[0]
    print(pd.value_counts(layer_id_amount,sort=False))
    count_num = pd.value_counts(layer_id_amount)
    budget_drop = 908*3877213696 - (221249536*(count_num[0]-1)+576716800*(count_num[1]-1)+932184064*(count_num[2]-1)+1287651328*(count_num[3]-1)+2401239040*(count_num[4]-1)+3092250624*(count_num[5]-1)+3783262208*(count_num[6]-1)+3876552704*(count_num[7]-1)+3877213696*(count_num[8]-1))
    budget_drop = budget_drop/(908*3877213696)
    score = (test_accuracy / test_num)*(budget_drop/0.20)**0.10
    with open('score_1000.txt','a') as file:
        print(score,file=file)
    with open('accuracy_1000.txt','a') as file:
        print(test_accuracy / test_num,file=file)
    with open('budget_drop_1000.txt','a') as file:
        print(budget_drop,file=file)
    return score

def xgb_model(params):
    """used for hyperopt"""
    confidence_threshold_1 = params['confidence_threshold_1']
    confidence_threshold_2 = params['confidence_threshold_2']
    confidence_threshold_3 = params['confidence_threshold_3']
    confidence_threshold_4 = params['confidence_threshold_4']
    confidence_threshold_5 = params['confidence_threshold_5']
    confidence_threshold_6 = params['confidence_threshold_6']
    confidence_threshold_7 = params['confidence_threshold_7']
    confidence_threshold_8 = params['confidence_threshold_8']
    confidence_threshold = [confidence_threshold_1,confidence_threshold_2,confidence_threshold_3,confidence_threshold_4,confidence_threshold_5,confidence_threshold_6,confidence_threshold_7,confidence_threshold_8]
    score = train_model(confidence_threshold)
    with open("threshold_1000.txt", "a") as file:
        print((confidence_threshold), file=file)
    return -score.item()

# #start training
# xgb_params = {'confidence_threshold_1'   : 0,
#               'confidence_threshold_2'   : 0,
#               'confidence_threshold_3'   : 0,
#               'confidence_threshold_4'   : 0,
#               'confidence_threshold_5'   : 0,
#               'confidence_threshold_6'   : 0,
#               'confidence_threshold_7'   : 0,
#               'confidence_threshold_8'   : 0}
# xgb_model(xgb_params)


#Threshold 贝叶斯随机森林优化
#start training with automatic parameter turning
myspace = {'confidence_threshold_1'   : hp.uniform('confidence_threshold_1',0, 1),
           'confidence_threshold_2'   : hp.uniform('confidence_threshold_2',0, 1),
           'confidence_threshold_3'   : hp.uniform('confidence_threshold_3',0, 1),
           'confidence_threshold_4'   : hp.uniform('confidence_threshold_4',0, 1),
           'confidence_threshold_5'   : hp.uniform('confidence_threshold_5',0, 1),
           'confidence_threshold_6'   : hp.uniform('confidence_threshold_6',0, 1),
           'confidence_threshold_7'   : hp.uniform('confidence_threshold_7',0, 1),
           'confidence_threshold_8'   : hp.uniform('confidence_threshold_8',0, 1)}
trials = Trials()
best = fmin(fn=xgb_model,
        space= myspace,
        algo=tpe.suggest,
        max_evals=1000,
        trials = trials)
print(best)




    # print("test-accuracy：{}".format(test_accuracy / test_num))
    # print(pd.value_counts(layer_id_amount))
