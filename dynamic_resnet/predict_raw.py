import torch
import torchvision
import torchvision.models
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.nn import functional
import torchvision.datasets as dset
import numpy as np
from PIL import Image
import pandas as pd
import time
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _single
from torch.autograd import Function
import torchvision.datasets as dset




# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))


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


class TriConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(TriConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)

    def forward(self, input):

        bw = self.weight
        ba = input
        bw = bw - bw.mean()
        bw = TernaryQuantize().apply(bw)
        # ba = TernaryQuantize().apply(ba)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(ba, expanded_padding, mode='circular'),
                            bw, self.bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv2d(ba, bw, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class TriLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, binary_act=True):
        super(TriLinear, self).__init__(in_features, out_features, bias=bias)
        self.output_ = None

    def forward(self, input):
        tw = self.weight
        ta = input
        tw = TernaryQuantize().apply(tw)
        output = F.linear(ta, tw, self.bias)
        self.output_ = output
        return output


class NoisedTriLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, noise=0):
        super(NoisedTriLinear, self).__init__(in_features, out_features, bias=bias)
        self.output_ = None

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

class NoisedTriConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', noise=0):
        super(NoisedTriConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        self.noise = noise

    def forward(self, input):

        tw = self.weight
        ta = input
        tw = tw - tw.mean()
        tw = TernaryQuantize().apply(tw)
        # ba = TernaryQuantize().apply(ba)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            output = F.conv2d(F.pad(ta, expanded_padding, mode='circular'),
                              tw, self.bias, self.stride,
                              _single(0), self.dilation, self.groups)
        else:
            output = F.conv2d(ta, tw, self.bias, self.stride,
                              self.padding, self.dilation, self.groups)

        if self.noise:
            output = output + self.noised_forward(input, tw)

        return output

    def noiseed_forward(self, x, weight):
        x = x.detach()

        batch_size, in_features, nsamples, npoints = x.size()
        x = x.reshape(-1, in_features, 1, 1)

        origin_weight = weight
        x_new = torch.zeros(x.shape[0], self.out_channels, 1, 1)

        for i in range(x.shape[0]):
            noise_weight = gen_noise(weight, self.noise).detach()
            noise_weight = noise_weight.sqeeze()

            x_i =  x[i, :, :, :].unsqueeze(0)
            # x_i = torch.matmul(noise_weight, x_i)
            x_i = F.conv2d(x_i, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            x_new[i, :, :, :] = x_i.squeeze(0)
            del noise_weight, x_i
        x_new = x_new.reshape(batch_size, self.out_channels, nsamples, npoints)
        return x_new.to(x.device).detach()


def AC_calculate(SA):
    SA = sorted(SA)
    SA_h = SA[-1]
    SA_sh = SA[-2]
    AC = (SA_h[0] - SA_sh[0]) / SA_sh[0]
    label = SA_h[1]
    return AC, label

class BasicBlock(nn.Module):
    def __init__(self,channels):
        super(BasicBlock, self).__init__()
        self.channels = channels
        # self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
        #                        kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        # self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
        #                        kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out_1 = self.conv1(x)
        out_1 = self.bn1(out_1)
        out_1 = self.relu(out_1)

        out_2 = self.conv2(out_1)
        out_2 = self.bn2(out_2)
        out = self.relu(out_1+out_2)

        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()
        self.in_channel = 3
        # self.conv1 = nn.Conv2d(1, self.in_channel, kernel_size=7, stride=2,
        #                        padding=3, bias=False)
        self.conv1 = nn.Conv2d(1, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(3, 6, kernel_size=1, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(6)

        self.conv3 = nn.Conv2d(6, 12, kernel_size=1, stride=2, bias=False)
        self.bn3 = nn.BatchNorm2d(12)

        self.conv4 = nn.Conv2d(12, 24, kernel_size=1, stride=2, bias=False)
        self.bn4 = nn.BatchNorm2d(24)

        self.layer1 = BasicBlock(3)
        self.layer2 = BasicBlock(6)
        self.layer3 = BasicBlock(12)
        self.layer4 = BasicBlock(24)
        self.layer5 = BasicBlock(24)
        self.layer6 = BasicBlock(24)
        self.layer7 = BasicBlock(24)
        self.layer8 = BasicBlock(24)
        self.layer9 = BasicBlock(24)
        self.layer10 = BasicBlock(24)
        self.layer11 = BasicBlock(24)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        # self.fc = nn.Linear(24 * block.expansion, num_classes)
        self.fc = nn.Linear(24, 10)
        # self.fc = nn.Linear(128, num_classes)


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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x = self.layer2(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x = self.layer3(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# semantic_center = torch.load('ResNet_semantic_center_0208_thrinary.pth')
# semantic_center = semantic_center.to(device)
net = ResNet()
new_m = net.to(device)
# 测试所保存的模型
m_state_dict = torch.load('C:\\Users\\zy\\Desktop\\Dynamic\\research\\weight\\Resnet_raw_2.pth')
new_m.load_state_dict(m_state_dict)

from torch.utils.data import random_split

mytransforms = transforms.Compose([
    # transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307),(0.3081))])


# test_data = torchvision.datasets.FashionMNIST(root = "./data_fashionmnist/test" , train = False ,download = True, transform=mytransforms)
# testdata = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
test_data = torchvision.datasets.MNIST(root = "./datas/test" , train = False ,download = True, transform=mytransforms)
testdata = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
# data_path = "C:\\Users\\zy\\Desktop\\Dynamic\\test"
# full_data = torchvision.datasets.ImageFolder(root=data_path,transform=mytransforms)
# dataset_size = len(full_data)
# train_dataset, test_dataset = random_split(
#     dataset=full_data,
#     lengths=[0, dataset_size],
#     generator=torch.Generator().manual_seed(0)
# )
# print(list(test_dataset))
# test_loss = 0
epoch = 1
test_accur_all = []
is_early_all = []
# # test_data = torch.utils.data.DataLoader(test_dataset,batch_size=1,num_workers=0,shuffle=False)

for i in range(epoch):
    test_accuracy = 0.0
    test_num = 0
    test_is_early_num = 0
    layer_id_amount = []
    confidence_out = []
    confidence_true = []
    new_m.eval()
    with torch.no_grad():
        test_bar = tqdm(testdata)
        start = time.time()
        for data in test_bar:
            img, target = data

            outputs = new_m(img.to(device))

            # output = outputs[1]
            # layer_id = outputs[0]
            # confidence = outputs[3]
            # is_early = outputs[2]
            # confidence_out.append((confidence,layer_id))
            # layer_id_amount.append(layer_id)
            accuracy = torch.sum(outputs == target.to(device))
            # is_early_num = np.sum(is_early != 0)
            test_accuracy = test_accuracy + accuracy
            # test_is_early_num = test_is_early_num + is_early_num
            test_num += img.size(0)

            # if outputs==target.to(device):
            #     confidence_true.append((confidence,layer_id))


        # confidence_threshold = 0.1
        # f = open(r'.\Resnet_1020_time.txt', 'a')
        # print('Running time: {} Seconds, threshold: {}'.format(end - start, confidence_threshold), file=f)
        # f.close()

        print("epoch:{}, test-accuracy: {}".format(i+1, test_accuracy / test_num))
        # print(pd.value_counts(layer_id_amount))
        # exit_layer = pd.value_counts(layer_id)

