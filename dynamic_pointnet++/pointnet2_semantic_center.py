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




class pointnet2_cls_ssg(nn.Module):
    def __init__(self, in_channels, nclasses):
        super(pointnet2_cls_ssg, self).__init__()
        self.pt_sa1 = PointNet_SA_Module(M=512, radius=0.2, K=32, in_channels=in_channels, mlp=[64, 64, 128], group_all=False)
        self.pt_sa2 = PointNet_SA_Module(M=512, radius=0.2, K=32, in_channels=131, mlp=[64, 64, 128], group_all=False)
        self.pt_sa3 = PointNet_SA_Module(M=512, radius=0.2, K=32, in_channels=131, mlp=[64, 64, 128], group_all=False)
        self.pt_sa4 = PointNet_SA_Module(M=512, radius=0.2, K=32, in_channels=131, mlp=[64, 64, 128], group_all=False)
        self.pt_sa5 = PointNet_SA_Module(M=512, radius=0.2, K=32, in_channels=131, mlp=[128, 128, 256], group_all=False)
        self.pt_sa6 = PointNet_SA_Module(M=128, radius=0.4, K=64, in_channels=259, mlp=[128, 128, 256], group_all=False)
        self.pt_sa7 = PointNet_SA_Module(M=128, radius=0.4, K=64, in_channels=259, mlp=[128, 128, 256], group_all=False)
        self.pt_sa8 = PointNet_SA_Module(M=None, radius=None, K=None, in_channels=259, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.cls = nn.Linear(256, nclasses)

    def forward(self, xyz, points):
        batchsize = xyz.shape[0]
        new_xyz, new_points = self.pt_sa1(xyz, points)
        # net_1 = new_points.view(batchsize, -1)
        net_1 = new_points.mean(dim=1)
        print(net_1.shape)
        new_xyz, new_points = self.pt_sa2(new_xyz, new_points)
        # net_2 = new_points.view(batchsize, -1)
        net_2 = new_points.mean(dim=1)
        print(net_2.shape)
        new_xyz, new_points = self.pt_sa3(new_xyz, new_points)
        # net_3 = new_points.view(batchsize, -1)
        net_3 = new_points.mean(dim=1)
        print(net_3.shape)
        new_xyz, new_points = self.pt_sa4(new_xyz, new_points)
        net_4 = new_points.mean(dim=1)
        print(net_4.shape)
        new_xyz, new_points = self.pt_sa5(new_xyz, new_points)
        net_5 = new_points.mean(dim=1)
        print(net_5.shape)
        new_xyz, new_points = self.pt_sa6(new_xyz, new_points)
        net_6 = new_points.mean(dim=1)
        print(net_6.shape)
        new_xyz, new_points = self.pt_sa7(new_xyz, new_points)
        net_7 = new_points.mean(dim=1)
        print(net_7.shape)
        new_xyz, new_points = self.pt_sa8(new_xyz, new_points)
        net_8 = new_points.mean(dim=1)
        print(net_8.shape)
        # net = self.dropout1(F.relu(self.bn1(self.fc1(net))))
        # net = self.dropout2(F.relu(self.bn2(self.fc2(net))))
        # net = self.cls(net)

        return net_1, net_2, net_3, net_4, net_5, net_6, net_7, net_8

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

path = Path("/home/yue/Pointnet2.PyTorch/ModelNet10_center")
folders = [directory for directory in sorted(os.listdir(path)) if os.path.isdir(path/directory)]
classes = {folder: i for i, folder in enumerate(folders)}
train_ds = PointCloudData(path, transform=train_transforms)
valid_ds = PointCloudData(path, valid=True, folder='test', transform=train_transforms)

inv_classes = {i: cat for cat, i in train_ds.classes.items()};

# print('Train dataset size: ', len(train_ds))
# # print('Valid dataset size: ', len(valid_ds))
# # print(valid_ds)
# print('Number of classes: ', len(train_ds.classes))
# print(train_ds.classes)
# # print(valid_ds.classes)
# print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())
# # print( train_ds[1509])
# print('Class: ', inv_classes[train_ds[0]['category']])
# print('Class: ', inv_classes[train_ds[100]['category']])
# print('Class: ', inv_classes[train_ds[200]['category']])
# print('Class: ', inv_classes[train_ds[300]['category']])
# print('Class: ', inv_classes[train_ds[400]['category']])
# print('Class: ', inv_classes[train_ds[500]['category']])
# print('Class: ', inv_classes[train_ds[600]['category']])
# print('Class: ', inv_classes[train_ds[700]['category']])
# print('Class: ', inv_classes[train_ds[800]['category']])
# print('Class: ', inv_classes[train_ds[900]['category']])




train_loader = DataLoader(dataset=train_ds, batch_size=100, shuffle=False)
valid_loader = DataLoader(dataset=valid_ds, batch_size=64)
    
    # 加载数据集1
# train_data = torchvision.datasets.MNIST(root = "./mnist/train" , train = True ,download = True, transform=mytransforms)
# traindata = torch.utils.data.DataLoader(dataset= train_data , batch_size=64, shuffle=True)
# test_data = torchvision.datasets.MNIST(root = "./mnist/test" , train = False ,download = True, transform=mytransforms)
# testdata = torch.utils.data.DataLoader(dataset=test_data, batch_size=64, shuffle=False)
# traindata = dset.ImageFolder('/home/yue/Pointnet2.PyTorch/modelnet10_/train')
# train_data = DataLoader(traindata, batch_size=64, shuffle=False)

# testdata = dset.ImageFolder('/home/yue/Pointnet2.PyTorch/modelnet10_/test')
# test_data = DataLoader(testdata, batch_size=64, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

model = pointnet2_cls_ssg(in_channels=3, nclasses=10)
# load pretrain weights
# download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
# model_weight_path = "./resnet34-pre.pth"
# assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
# net.load_state_dict(torch.load(model_weight_path, map_location=device))
model.to(device)
# 测试所保存的模型
m_state_dict = torch.load('pointnet2_ternary_0.83.pth')
model.load_state_dict(m_state_dict)

epoch = 1
semantic_center_final = []

semantic_center_epoch =[]
semantic_center_real =[]
item_1 = []
for epoch in range(epoch):
    model.eval()
    with torch.no_grad():
        for data in train_loader:
            inputs, target = data['pointcloud'].to(device).float(), data['category'].to(device)
            print(target)
            xyz, points = inputs[:, :, :3], inputs[:, :, 3:]
            sc = model(xyz.to(device), points.to(device))
            semantic_center = []
            for i, sc_j in enumerate(sc):
                sc_real=torch.mean(sc_j,dim=0)
                print(sc_real.shape)
                semantic_center.append(sc_real)
                print(len(semantic_center))

            label = target[0]
            # print(label)
            semantic_center_real.append((semantic_center,label))


# torch.save(semantic_center_real, 'pointnet2_semantic_center_ternary_0.83.pth')
# print('save sc')

# semantic_center = torch.load('pointnet2_semantic_center_ternary_0.83.pth')
# print(semantic_center[0][1])
# print(semantic_center[1][1])
# print(semantic_center[2][1])
# print(semantic_center[3][1])
# print(semantic_center[4][1])
# print(semantic_center[5][1])
# print(semantic_center[6][1])
# print(semantic_center[7][1])
# print(semantic_center[8][1])
# print(semantic_center[9][1])
