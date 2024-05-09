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


def binarization(input):
    # Binarize matrices
    # input: 2D matrix to be binarized

    # Column-wise min/max of input
    input_min = torch.min(input, dim=0)[0]
    input_max = torch.max(input, dim=0)[0]

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
    input_int = torch.round(input_int).to(torch.int)
    # Binarize
    input_b = input_int.unsqueeze(-1).bitwise_and(1).ne(0).byte()
    return input_int

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
        self.conv1 = NoisedTriConv2d(in_channels=channels, out_channels=channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        # self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
        #                        kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = NoisedTriConv2d(in_channels=channels, out_channels=channels,
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
        self.conv1 = NoisedTriConv2d(1, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = NoisedTriConv2d(3, 6, kernel_size=1, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(6)

        self.conv3 = NoisedTriConv2d(6, 12, kernel_size=1, stride=2, bias=False)
        self.bn3 = nn.BatchNorm2d(12)

        self.conv4 = NoisedTriConv2d(12, 24, kernel_size=1, stride=2, bias=False)
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
        self.fc = NoisedTriLinear(24, 10)
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


    # def forward(self, x):
    #     # x = self.conv1(x)
    #     # x = self.bn1(x)
    #     # x = self.relu(x)
    #     # x = self.maxpool(x)
    #     #
    #     # x = self.layer1(x)
    #     # x = self.conv2(x)
    #     # x = self.bn2(x)
    #     #
    #     # x = self.layer2(x)
    #     # x = self.conv3(x)
    #     # x = self.bn3(x)
    #     #
    #     # x = self.layer3(x)
    #     # x = self.conv4(x)
    #     # x = self.bn4(x)
    #     #
    #     # x = self.layer4(x)
    #     # x = self.layer5(x)
    #     # x = self.layer6(x)
    #     # x = self.layer7(x)
    #     # x = self.layer8(x)
    #     # x = self.layer9(x)
    #     # x = self.layer10(x)
    #     # x = self.layer11(x)
    #     #
    #     # x = self.avgpool(x)
    #     # x = torch.flatten(x, 1)
    #     # x = self.fc(x)
    #
    #     return x
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        confidence_threshold = [0.0200, 0.0300, 0.0900, 0.0100, 0.0900, 0.0400, 0.0200, 0.0160, 0.0150, 0.0140, 0.0050]

        x1 = self.layer1(x)
        sv1 = nn.functional.adaptive_avg_pool2d(x1, (1, 1))
        # sv1 = sv1.expand([2,-1,-1,-1,-1])
        sim_1 = self.classifier(sv1, semantic_center, 0)
        confidence_1 = AC_calculate(sim_1)
        # if confidence_1[0] >= confidence_threshold:
        if confidence_1[0] >= confidence_threshold[0]:
            is_early = True
            return 0, confidence_1[1], is_early, confidence_1[0],x1

        else:
            x2 = self.conv2(x1)
            x2 = self.bn2(x2)
            x2 = self.layer2(x2)
            sv2 = nn.functional.adaptive_avg_pool2d(x2, (1, 1))
            # sv2= sv2.expand([2,-1,-1,-1,-1])
            sim_2 = self.classifier(sv2, semantic_center, 1)
            # sim_2 = self.calculate(sim_1, sim_2)
            confidence_2 = AC_calculate(sim_2)
            if confidence_2[0] >= confidence_threshold[1]:
                is_early = True
                return 1, confidence_2[1], is_early, confidence_2[0],x2

            else:
                x3 = self.conv3(x2)
                x3 = self.bn3(x3)
                x3 = self.layer3(x3)
                sv3 = nn.functional.adaptive_avg_pool2d(x3, (1, 1))
                # sv3 = sv3.expand([2,-1,-1,-1,-1])
                sim_3 = self.classifier(sv3, semantic_center, 2)
                # sim_3 = self.calculate(sim_2, sim_3)
                confidence_3 = AC_calculate(sim_3)
                if confidence_3[0] >= confidence_threshold[2]:
                    is_early = True
                    return 2, confidence_3[1], is_early, confidence_3[0],x3

                else:
                    x4 = self.conv4(x3)
                    x4 = self.bn4(x4)
                    x4 = self.layer4(x4)
                    sv4 = nn.functional.adaptive_avg_pool2d(x4, (1, 1))
                    # sv4 = sv4.expand([2,-1,-1,-1,-1])
                    sim_4 = self.classifier(sv4, semantic_center, 3)
                    # sim_4 = self.calculate(sim_3, sim_4)
                    confidence_4 = AC_calculate(sim_4)
                    if confidence_4[0] >= confidence_threshold[3]:
                        is_early = True
                        return 3, confidence_4[1], is_early, confidence_4[0],x4
                    else:
                        x5 = self.layer5(x4)
                        sv5 = nn.functional.adaptive_avg_pool2d(x5, (1, 1))
                        # sv5 = sv5.expand([2,-1,-1,-1,-1])
                        sim_5 = self.classifier(sv5, semantic_center, 4)
                        # sim_5 = self.calculate(sim_4, sim_5)
                        confidence_5 = AC_calculate(sim_5)
                        if confidence_5[0] >= confidence_threshold[4]:
                            is_early = True
                            return 4, confidence_5[1], is_early,confidence_5[0],x5
                        else:
                            x6 = self.layer6(x5)
                            sv6 = nn.functional.adaptive_avg_pool2d(x6, (1, 1))
                            # sv6 = sv6.expand([2,-1,-1,-1,-1])
                            sim_6 = self.classifier(sv6, semantic_center, 5)
                            # sim_6 = self.calculate(sim_5, sim_6)
                            confidence_6 = AC_calculate(sim_6)
                            if confidence_6[0] >= confidence_threshold[5]:
                                is_early = True
                                return 5, confidence_6[1], is_early, confidence_6[0],x6
                            else:
                                x7 = self.layer7(x6)
                                sv7 = nn.functional.adaptive_avg_pool2d(x7, (1, 1))
                                # sv7 = sv7.expand([2,-1,-1,-1,-1])
                                sim_7 = self.classifier(sv7, semantic_center, 6)
                                # sim_7 = self.calculate(sim_6, sim_7)
                                confidence_7 = AC_calculate(sim_7)
                                if confidence_7[0] >= confidence_threshold[6]:
                                    is_early = True
                                    return 6, confidence_7[1], is_early, confidence_7[0],x7
                                else:
                                    x8 = self.layer8(x7)
                                    sv8 = nn.functional.adaptive_avg_pool2d(x8, (1, 1))
                                    # sv8 =sv8.expand([2,-1,-1,-1,-1])
                                    sim_8 = self.classifier(sv8, semantic_center, 7)
                                    # sim_8 = self.calculate(sim_7, sim_8)
                                    confidence_8 = AC_calculate(sim_8)
                                    if confidence_8[0] >= confidence_threshold[7]:
                                        is_early = True
                                        return 7, confidence_8[1], is_early,confidence_8[0],x8
                                    else:
                                        x9 = self.layer9(x8)
                                        sv9 = nn.functional.adaptive_avg_pool2d(x9, (1, 1))
                                        # sv9 = sv9.expand([2,-1,-1,-1,-1])
                                        sim_9 = self.classifier(sv9, semantic_center, 8)
                                        # sim_9 = self.calculate(sim_8, sim_9)
                                        confidence_9 = AC_calculate(sim_9)
                                        if confidence_9[0] >= confidence_threshold[8]:
                                            is_early = True
                                            return 8, confidence_9[1], is_early,confidence_9[0],x9
                                        else:
                                            x10 = self.layer10(x9)
                                            sv10 = nn.functional.adaptive_avg_pool2d(x10, (1, 1))
                                            # sv10 = sv10.expand([2,-1,-1,-1,-1])
                                            sim_10 = self.classifier(sv10, semantic_center, 9)
                                            # sim_10 = self.calculate(sim_9, sim_10)
                                            confidence_10 = AC_calculate(sim_10)
                                            if confidence_10[0] >= confidence_threshold[9]:
                                                is_early = True
                                                return 9, confidence_10[1], is_early,confidence_10[0],x10
                                            else:
                                                x11 = self.layer11(x10)
                                                sv11 = nn.functional.adaptive_avg_pool2d(x11, (1, 1))
                                                # sv11 = sv11.expand([2,-1,-1,-1,-1])
                                                sim_11 = self.classifier(sv11, semantic_center, 10)
                                                # sim_11 = self.calculate(sim_10, sim_11)
                                                confidence_11 = AC_calculate(sim_11)
                                                if confidence_11[0] >= confidence_threshold[10]:
                                                    is_early = True
                                                    return 10, confidence_11[1], is_early,confidence_11[0],x11

                                                else:
                                                    layer_id = 11
                                                    is_early = False
                                                    output = self.avgpool(x11)
                                                    output = torch.flatten(output, 1)
                                                    output = self.fc(output)
                                                    ans = (output.argmax(1)).item()
                                                    return layer_id, ans, is_early, output.argmax(1),output



semantic_center = torch.load('C:\\Users\\zy\\Desktop\\Dynamic\\weight and center\\mnist\\semantic_center_mnist.pth')

net = ResNet()
new_m = net.to(device)
# 测试所保存的模型
m_state_dict = torch.load('C:\\Users\\zy\\Desktop\\Dynamic\\weight and center\\mnist\\weight_mnist.pth',map_location='cuda:0')
new_m.load_state_dict(m_state_dict)

from torch.utils.data import random_split

mytransforms = transforms.Compose([
    # transforms.Resize((7, 7)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307),(0.3081))])

img_path = 'C:\\Users\\zy\\Desktop\\Dynamic\\research\\data_test'
dataset = dset.ImageFolder(img_path,transform=mytransforms)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

test_accuracy = 0.0
test_num = 0
test_is_early_num = 0
layer_id_amount = []
confidence_out = []
confidence_true = []
new_m.eval()
with torch.no_grad():
    for data, label in data_loader:
        images = data.to(device)
        labels = label.to(device)
        images_binary = binarization(images)/1
        outputs = new_m(images)
        outputs_binary = new_m(images_binary)
        output = outputs[1]
        layer_id = outputs[0]
        confidence = outputs[3]
        is_early = outputs[2]
        feature_map = outputs[4]
        output_binary = outputs_binary[1]
        layer_id_binary = outputs_binary[0]
        confidence_binary = outputs_binary[3]
        is_early_binary = outputs_binary[2]
        feature_map_binary = outputs_binary[4]
        confidence_out.append(confidence)
        print(feature_map-feature_map_binary)
        layer_id_amount.append(layer_id)
        accuracy = torch.sum(output == label.to(device))
        is_early_num = np.sum(is_early != 0)
        test_accuracy = test_accuracy + accuracy
        test_is_early_num = test_is_early_num + is_early_num
        test_num += images.size(0)

        if output == label.to(device):
            confidence_true.append((confidence, layer_id))
        #获取输出列表这是一个列表，里面每个代表了每层的输出
        #通过一个迭代器来遍历每个特征图
        # [N, C, H, W] -> [C, H, W]
        feature_map = feature_map.permute(0, 3, 1, 2)
        im = np.squeeze(feature_map.detach().cpu().numpy())#把tensor变成numpy
        # show top 12 feature maps
        f1_map_num = im.shape[0]
        # 绘制图像
        plt.figure()
        # 通过遍历的方式，将通道的tensor拿出
        # for index in range(1, f1_map_num + 1):
        #     plt.subplot(3, 5, index)
        plt.imshow(im, cmap='gray')
        plt.axis('off')

        plt.show()

    print("test-accuracy: {},  test_is_early_num:{}".format(test_accuracy / test_num,
                                                                 test_is_early_num / test_num))
    print(pd.value_counts(layer_id_amount))
    output = pd.DataFrame({'confidence': confidence_out, 'layer_id': layer_id_amount})
    output.to_csv('submission.csv', index=False)



