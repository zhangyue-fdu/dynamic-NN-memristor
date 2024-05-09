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
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Hardtanh, BatchNorm1d as BN
from torch.nn.modules.utils import _single
from torch.autograd import Function
from torch.nn import Parameter
import math
from torch.nn import functional
from torch.nn.modules.utils import _single, _pair, _triple
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torchvision.datasets as dset

def Similarity_calculate(SA):

    SA = sorted(SA)
    SA_h = SA[-1]
    SA_sh = SA[-2]
    AC = (SA_h[0] - SA_sh[0]) / SA_sh[0]
    label = SA_h[1]
    return AC, label

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

# class NoisedTriConv2d(torch.nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=1, dilation=1, groups=1,
#                  bias=True, noise=0.05):
#         super(NoisedTriConv2d, self).__init__(
#             in_channels, out_channels, kernel_size, stride, padding, dilation,
#             groups, bias)
#         self.noise = noise

#     def forward(self, input):

#         tw = self.weight
#         ta = input
#         tw = tw - tw.mean()
#         tw = TernaryQuantize().apply(tw)
#         # ba = TernaryQuantize().apply(ba)

#         if not self.noise:
#             output = F.conv2d(ta, tw, self.bias, self.stride,
#                               self.padding, self.dilation, self.groups)

#         else:
#             output = F.conv2d(ta, tw, self.bias, self.stride,
#                               self.padding, self.dilation, self.groups) + self.noised_forward(input, tw)

#         return output

#     def noised_forward(self, x, weight):
#         x = x.detach()

#         batch_size, in_features, nsamples, npoints = x.size()
#         nsamples = int(nsamples/self.stride[0])
#         npoints = int(npoints/self.stride[0])
#         x = x.reshape(-1, in_features, 1, 1)

#         origin_weight = weight
#         nvectors = int(batch_size * nsamples * npoints)
#         x_new = torch.zeros(batch_size, self.out_channels, nsamples, npoints)

#         for i in range(batch_size):
#             noise_weight = gen_noise(weight, self.noise).detach()
#             # noise_weight = noise_weight.squeeze()

#             # x_i =  x[i, :, :, :].unsqueeze(0)
#             # x_i = torch.matmul(noise_weight, x_i)
#             x_i = F.conv2d(x[i, :, :, :], noise_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
#             x_new[i, :, :, :] = x_i.squeeze()
#             del noise_weight, x_i
#         return x_new.to(x.device).detach()




class NoisedTriConv2d(torch.nn.modules.conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', noise=0):
        super(NoisedTriConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        self.noise = noise
        self.weight = Parameter(torch.Tensor(out_channels, in_channels//groups, *kernel_size))

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

    def noised_forward(self, x, weight):
        x = x.detach()

        batch_size, in_features, nsamples, npoints = x.size()
        nsamples = round(nsamples/self.stride[0])
        npoints = round(npoints/self.stride[0])
        x = x.reshape(-1, in_features, 1, 1)

        origin_weight = weight
        nvectors = int(batch_size * nsamples * npoints)
        x_new = torch.zeros(nvectors, self.out_channels, 1, 1)

        for i in range(x_new.shape[0]):
            noise_weight = gen_noise(weight, self.noise).detach()
            noise_weight = noise_weight.squeeze()

            x_i =  x[i, :, :, :].unsqueeze(0)
            # x_i = torch.matmul(noise_weight, x_i)
            x_i = F.conv2d(x_i, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            x_new[i, :, :, :] = x_i.squeeze(0)
            del noise_weight, x_i
        x_new = x_new.reshape(batch_size, self.out_channels, nsamples, npoints)
        return x_new.to(x.device).detach()



class BasicBlock(nn.Module):
    def __init__(self,channels):
        super(BasicBlock, self).__init__()
        self.channels = channels
        # self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
        #                        kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = TriConv2d(in_channels=channels, out_channels=channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        # self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
        #                        kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = TriConv2d(in_channels=channels, out_channels=channels,
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
        self.conv1 = TriConv2d(1, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = TriConv2d(3, 6, kernel_size=1, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(6)

        self.conv3 = TriConv2d(6, 12, kernel_size=1, stride=2, bias=False)
        self.bn3 = nn.BatchNorm2d(12)

        self.conv4 = TriConv2d(12, 24, kernel_size=1, stride=2, bias=False)
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
        for sc_j_1 in semantic_center:
            semantic_center_j_1 = sc_j_1[0][layer_id].reshape(-1)
            similarity = torch.cosine_similarity(semantic_center_j_1, sv, dim=0)
            label = sc_j_1[1]
            sim_1.append((similarity, label))

        confidence_label_1 = Similarity_calculate(sim_1)
        confidence_1 = confidence_label_1[0]
        label = confidence_label_1[1]
        return confidence_1, label, layer_id

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        sv1 = nn.functional.adaptive_avg_pool2d(x1, (1, 1))
        # sv1 = sv1.expand([2,-1,-1,-1,-1])
        # sv1 = Toternary(sv1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.layer2(x2)
        sv2 = nn.functional.adaptive_avg_pool2d(x2, (1, 1))
        # sv2 = sv2.expand([2,-1,-1,-1,-1])
        # sv2 = Toternary(sv2)

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        x3 = self.layer3(x3)
        sv3 = nn.functional.adaptive_avg_pool2d(x3, (1, 1))
        # sv3 = sv3.expand([2,-1,-1,-1,-1])
        # sv3 = Toternary(sv3)

        x4 = self.conv4(x3)
        x4 = self.bn4(x4)
        x4 = self.layer4(x4)
        sv4 = nn.functional.adaptive_avg_pool2d(x4, (1, 1))
        # sv4 = sv4.expand([2,-1,-1,-1,-1])
        # sv4 = Toternary(sv4)

        x5 = self.layer5(x4)
        sv5 = nn.functional.adaptive_avg_pool2d(x5, (1, 1))
        # sv5.expand([2,-1,-1,-1,-1])
        # sv5 = Toternary(sv5)
        x6 = self.layer6(x5)
        sv6 = nn.functional.adaptive_avg_pool2d(x6, (1, 1))
        # sv6.expand([2,-1,-1,-1,-1])
        # sv6 = Toternary(sv6)
        x7 = self.layer7(x6)
        sv7 = nn.functional.adaptive_avg_pool2d(x7, (1, 1))
        # sv7.expand([2,-1,-1,-1,-1])
        # sv7 = Toternary(sv7)
        x8= self.layer8(x7)
        sv8 = nn.functional.adaptive_avg_pool2d(x8, (1, 1))
        # sv8 = sv8.expand([2,-1,-1,-1,-1])
        # sv8 = Toternary(sv8)
        x9 = self.layer9(x8)
        sv9 = nn.functional.adaptive_avg_pool2d(x9, (1, 1))
        # sv9 = sv9.expand([2,-1,-1,-1,-1])
        # sv9 = Toternary(sv9)
        x10 = self.layer10(x9)
        sv10 = nn.functional.adaptive_avg_pool2d(x10, (1, 1))
        # sv10.expand([2,-1,-1,-1,-1])
        # sv10 = Toternary(sv10)
        x11 = self.layer11(x10)
        sv11 = nn.functional.adaptive_avg_pool2d(x11, (1, 1))
        # sv11.expand([2,-1,-1,-1,-1])
        # sv11 = Toternary(sv11)

        x12 = self.avgpool(x11)
        x12 = torch.flatten(x12, 1)
        x12 = self.fc(x12)

        return sv1, sv2, sv3, sv4, sv5, sv6, sv7, sv8, sv9, sv10, sv11, x12


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))


net = ResNet()
new_m = net.to(device)
print(new_m)  # 输出模型结构
# 测试所保存的模型
m_state_dict = torch.load('Resnet_0208_threnary.pth',map_location='cuda:0')
new_m.load_state_dict(m_state_dict)

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

epoch = 1
semantic_center_final = []

semantic_center_epoch =[]
semantic_center_real =[]
item_1 = []
for epoch in range(epoch):
    new_m.eval()
    with torch.no_grad():
        for data, label in data_loader:
            images = data.to(device)
            labels = label.to(device)

            sc = new_m(images)
            # print(sc[1].shape)
            semantic_center = []
            for i, sc_j in enumerate(sc):
                sc_real=torch.mean(sc_j,dim=0)
                print(sc_real.shape)
                semantic_center.append(sc_real)
                print(len(semantic_center))

            label = labels[0]
            semantic_center_real.append((semantic_center,label))


torch.save(semantic_center_real, 'ResNet_semantic_center_1116_noised_0208.pth')
print('save sc')

semantic_center = torch.load('ResNet_semantic_center_1116_noised_0208.pth')
print(len(semantic_center))

# dataframe = pd.DataFrame({'semantic_center': semantic_center})
# dataframe.to_csv('semantic_centeer.csv', index=False, sep=',')

# similarity = torch.cosine_similarity(semantic_center[0][0][3],semantic_center[1][0][3],dim=0)
# print(similarity)
