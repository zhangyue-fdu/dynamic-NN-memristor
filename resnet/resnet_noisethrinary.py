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
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def Toternary(semantic_center):
    ctx_max, ctx_min = torch.max(semantic_center), torch.min(semantic_center)
    lower_interval = ctx_min + (ctx_max - ctx_min) / 3
    higher_interval = ctx_max - (ctx_max - ctx_min) / 3
    out = torch.where(semantic_center < lower_interval,
                      torch.tensor(-1.).to(semantic_center.device, semantic_center.dtype), semantic_center)
    out = torch.where(semantic_center > higher_interval,
                      torch.tensor(1.).to(semantic_center.device, semantic_center.dtype), out)
    out = torch.where((semantic_center >= lower_interval) & (semantic_center <= higher_interval),
                      torch.tensor(0.).to(semantic_center.device, semantic_center.dtype), out)

    return out

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
    def __init__(self, in_features, out_features, bias=True, noise=0.05):
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




class NoisedTriConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', noise=0.05):
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


mytransforms = transforms.Compose([
    # transforms.Resize((16, 16)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307),(0.3081))])

# 加载数据集1
train_data = torchvision.datasets.MNIST(root = "./datas/train" , train = True ,download = True, transform=mytransforms)
traindata = torch.utils.data.DataLoader(dataset= train_data , batch_size=64, shuffle=True)
test_data = torchvision.datasets.MNIST(root = "./datas/test" , train = False ,download = True, transform=mytransforms)
testdata = torch.utils.data.DataLoader(dataset=test_data, batch_size=64, shuffle=False)

train_size = len(train_data)
test_size = len(test_data)
print(train_size)
print(test_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))


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

net = ResNet()
# load pretrain weights
# download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
# model_weight_path = "./resnet34-pre.pth"
# assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
# net.load_state_dict(torch.load(model_weight_path, map_location=device))
net.to(device)
# print(net.to(device))  # 输出模型结构


epoch = 300  # 迭代次数即训练次数
learning = 0.001  # 学习率
optimizer = torch.optim.Adam(net.parameters(), lr=learning)  # 使用Adam优化器-写论文的话可以具体查一下这个优化器的原理
loss = nn.CrossEntropyLoss()  # 损失计算方式，交叉熵损失函数

train_loss_all = []  # 存放训练集损失的数组
train_accur_all = []  # 存放训练集准确率的数组
test_loss_all = []  # 存放测试集损失的数组
test_accur_all = []  # 存放测试集准确率的数组
for i in range(epoch):  # 开始迭代
    train_loss = 0  # 训练集的损失初始设为0
    train_num = 0.0  #
    train_accuracy = 0.0  # 训练集的准确率初始设为0
    net.train()  # 将模型设置成 训练模式
    train_bar = tqdm(traindata)  # 用于进度条显示，没啥实际用处
    for step, data in enumerate(train_bar):  # 开始迭代跑， enumerate这个函数不懂可以查查，将训练集分为 data是序号，data是数据
        img, target = data  # 将data 分位 img图片，target标签
        optimizer.zero_grad()  # 清空历史梯度
        outputs = net(img.to(device))  # 将图片打入网络进行训练,outputs是输出的结果

        loss1 = loss(outputs, target.to(device))  # 计算神经网络输出的结果outputs与图片真实标签target的差别-这就是我们通常情况下称为的损失
        outputs = torch.argmax(outputs, 1)  # 会输出10个值，最大的值就是我们预测的结果 求最大值
        loss1.backward()  # 神经网络反向传播
        optimizer.step()  # 梯度优化 用上面的abam优化
        train_loss += abs(loss1.item()) * img.size(0)  # 将所有损失的绝对值加起来
        accuracy = torch.sum(outputs == target.to(device))  # outputs == target的 即使预测正确的，统计预测正确的个数,从而计算准确率
        train_accuracy = train_accuracy + accuracy  # 求训练集的准确率
        train_num += img.size(0)  #

    print("epoch：{} ， train-Loss：{} , train-accuracy：{}".format(i + 1, train_loss / train_num,  # 输出训练情况
                                                                train_accuracy / train_num))
    train_loss_all.append(train_loss / train_num)  # 将训练的损失放到一个列表里 方便后续画图
    train_accur_all.append(train_accuracy.double().item() / train_num)  # 训练集的准确率
    test_loss = 0  # 同上 测试损失
    test_accuracy = 0.0  # 测试准确率
    test_num = 0
    net.eval()  # 将模型调整为测试模型
    with torch.no_grad():  # 清空历史梯度，进行测试  与训练最大的区别是测试过程中取消了反向传播
        test_bar = tqdm(testdata) 
        for data in test_bar:
            img, target = data

            outputs = net(img.to(device))
            loss2 = loss(outputs, target.to(device))
            outputs = torch.argmax(outputs, 1)
            test_loss = test_loss + abs(loss2.item()) * img.size(0)
            accuracy = torch.sum(outputs == target.to(device))
            test_accuracy = test_accuracy + accuracy
            test_num += img.size(0)

    print("test-Loss：{} , test-accuracy：{}".format(test_loss / test_num, test_accuracy / test_num))
    test_loss_all.append(test_loss / test_num)
    test_accur_all.append(test_accuracy.double().item() / test_num)

# 下面的是画图过程，将上述存放的列表  画出来即可
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(epoch), train_loss_all,
         "ro-", label="Train loss")
plt.plot(range(epoch), test_loss_all,
         "bs-", label="test loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.subplot(1, 2, 2)
plt.plot(range(epoch), train_accur_all,
         "ro-", label="Train accur")
plt.plot(range(epoch), test_accur_all,
         "bs-", label="test accur")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.legend()
plt.show()

torch.save(net.state_dict(), "Resnet_1221_noised.pth")
print("模型已保存")




