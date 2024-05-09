import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.nn import functional
from thop import profile




# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))


# def get_cos(target, behaviored):
#     attention_distribution = []
#
#     for i in range(behaviored.size(0)):
#         attention_score = torch.cosine_similarity(target, behaviored[i].view(1, -1))  # 计算每一个元素与给定元素的余弦相似度
#         attention_distribution.append(attention_score)
#     attention_distribution = torch.Tensor(attention_distribution)
#
#     return attention_distribution / torch.sum(attention_distribution, 0)  # 标准化


def AC_calculate(SA):
    SA = sorted(SA)
    SA_h = SA[-1]
    SA_sh = SA[-2]
    AC = (SA_h[0] - SA_sh[0]) / SA_sh[0]
    label = SA_h[1]
    return AC, label

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        # sv = nn.functional.adaptive_avg_pool2d(out, (1, 1))

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=10,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 4

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(1, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 4, blocks_num[0])
        self.layer2 = self._make_layer(block, 8, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 16, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 32, blocks_num[2], stride=1)
        self.layer5 = self._make_layer(block, 32, blocks_num[2], stride=1)
        self.layer6 = self._make_layer(block, 32, blocks_num[2], stride=1)
        self.layer7 = self._make_layer(block, 32, blocks_num[2], stride=1)
        self.layer8 = self._make_layer(block, 32, blocks_num[2], stride=1)
        self.layer9 = self._make_layer(block, 32, blocks_num[2], stride=1)
        self.layer10 = self._make_layer(block, 32, blocks_num[2], stride=1)
        self.layer11 = self._make_layer(block, 32, blocks_num[2], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(32 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

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

        confidence_threshold = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

        x1 = self.layer1(x)
        sv1 = nn.functional.adaptive_avg_pool2d(x1, (1, 1))
        sim_1 = self.classifier(sv1, semantic_center, 0)
        confidence_1 = AC_calculate(sim_1)
        # if confidence_1[0] >= confidence_threshold:
        if confidence_1[0] >= confidence_threshold[0]:
            is_early = True
            return 0, confidence_1[1], is_early, confidence_1[0]

        else:
            x2 = self.layer2(x1)
            sv2 = nn.functional.adaptive_avg_pool2d(x2, (1, 1))
            sim_2 = self.classifier(sv2, semantic_center, 1)
            # sim_2 = self.calculate(sim_1, sim_2)
            confidence_2 = AC_calculate(sim_2)
            if confidence_2[0] >= confidence_threshold[1]:
                is_early = True
                return 1, confidence_2[1], is_early, confidence_2[0]

            else:
                x3 = self.layer3(x2)
                sv3 = nn.functional.adaptive_avg_pool2d(x3, (1, 1))
                sim_3 = self.classifier(sv3, semantic_center, 2)
                # sim_3 = self.calculate(sim_2, sim_3)
                confidence_3 = AC_calculate(sim_3)
                if confidence_3[0] >= confidence_threshold[2]:
                    is_early = True
                    return 2, confidence_3[1], is_early, confidence_3[0]

                else:
                    x4 = self.layer4(x3)
                    sv4 = nn.functional.adaptive_avg_pool2d(x4, (1, 1))
                    sim_4 = self.classifier(sv4, semantic_center, 3)
                    # sim_4 = self.calculate(sim_3, sim_4)
                    confidence_4 = AC_calculate(sim_4)
                    if confidence_4[0] >= confidence_threshold[3]:
                        is_early = True
                        return 3, confidence_4[1], is_early, confidence_4[0]
                    else:
                        x5 = self.layer5(x4)
                        sv5 = nn.functional.adaptive_avg_pool2d(x5, (1, 1))
                        sim_5 = self.classifier(sv5, semantic_center, 4)
                        # sim_5 = self.calculate(sim_4, sim_5)
                        confidence_5 = AC_calculate(sim_5)
                        if confidence_5[0] >= confidence_threshold[4]:
                            is_early = True
                            return 4, confidence_5[1], is_early,confidence_5[0]
                        else:
                            x6 = self.layer6(x5)
                            sv6 = nn.functional.adaptive_avg_pool2d(x6, (1, 1))
                            sim_6 = self.classifier(sv6, semantic_center, 5)
                            # sim_6 = self.calculate(sim_5, sim_6)
                            confidence_6 = AC_calculate(sim_6)
                            if confidence_6[0] >= confidence_threshold[5]:
                                is_early = True
                                return 5, confidence_6[1], is_early, confidence_6[0]
                            else:
                                x7 = self.layer7(x6)
                                sv7 = nn.functional.adaptive_avg_pool2d(x7, (1, 1))
                                sim_7 = self.classifier(sv7, semantic_center, 6)
                                # sim_7 = self.calculate(sim_6, sim_7)
                                confidence_7 = AC_calculate(sim_7)
                                if confidence_7[0] >= confidence_threshold[6]:
                                    is_early = True
                                    return 6, confidence_7[1], is_early, confidence_7[0]
                                else:
                                    x8 = self.layer8(x7)
                                    sv8 = nn.functional.adaptive_avg_pool2d(x8, (1, 1))
                                    sim_8 = self.classifier(sv8, semantic_center, 7)
                                    # sim_8 = self.calculate(sim_7, sim_8)
                                    confidence_8 = AC_calculate(sim_8)
                                    if confidence_8[0] >= confidence_threshold[7]:
                                        is_early = True
                                        return 7, confidence_8[1], is_early,confidence_8[0]
                                    else:
                                        x9 = self.layer9(x8)
                                        sv9 = nn.functional.adaptive_avg_pool2d(x9, (1, 1))
                                        sim_9 = self.classifier(sv9, semantic_center, 8)
                                        # sim_9 = self.calculate(sim_8, sim_9)
                                        confidence_9 = AC_calculate(sim_9)
                                        if confidence_9[0] >= confidence_threshold[8]:
                                            is_early = True
                                            return 8, confidence_9[1], is_early,confidence_9[0]
                                        else:
                                            x10 = self.layer10(x9)
                                            sv10 = nn.functional.adaptive_avg_pool2d(x10, (1, 1))
                                            sim_10 = self.classifier(sv10, semantic_center, 9)
                                            # sim_10 = self.calculate(sim_9, sim_10)
                                            confidence_10 = AC_calculate(sim_10)
                                            if confidence_10[0] >= confidence_threshold[9]:
                                                is_early = True
                                                return 9, confidence_10[1], is_early,confidence_10[0]
                                            else:
                                                x11 = self.layer11(x10)
                                                sv11 = nn.functional.adaptive_avg_pool2d(x11, (1, 1))
                                                sim_11 = self.classifier(sv11, semantic_center, 10)
                                                # sim_11 = self.calculate(sim_10, sim_11)
                                                confidence_11 = AC_calculate(sim_11)
                                                if confidence_11[0] >= confidence_threshold[10]:
                                                    is_early = True
                                                    return 10, confidence_11[1], is_early,confidence_11[0]

                                                else:
                                                    layer_id = 11
                                                    is_early = False
                                                    if self.include_top:
                                                        output = self.avgpool(x11)
                                                        output = torch.flatten(output, 1)
                                                        output = self.fc(output)
                                                    ans = (output.argmax(1)).item()
                                                    return layer_id, ans, is_early,output.argmax(1)


def resnet34(num_classes=10, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=10, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=10, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=10, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=10, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


# data_transform = {
#     "train": transforms.Compose([transforms.RandomResizedCrop(120),
#                                  transforms.RandomHorizontalFlip(),
#                                  transforms.ToTensor(),
#                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
#     "val": transforms.Compose([transforms.Resize((120, 120)),  # cannot 224, must (224, 224)
#                                transforms.ToTensor(),
#                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
#
# train_data = torchvision.datasets.ImageFolder(root="./train", transform=data_transform["train"])
#
# traindata = DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=0)  # 将训练数据以每次32张图片的形式抽出进行训练
#
# test_data = torchvision.datasets.ImageFolder(root="./test", transform=data_transform["val"])
#
# train_size = len(train_data)  # 训练集的长度
# test_size = len(test_data)  # 测试集的长度
# print(train_size)  # 输出训练集长度看一下，相当于看看有几张图片
# print(test_size)  # 输出测试集长度看一下，相当于看看有几张图片
# testdata = DataLoader(dataset=test_data, batch_size=128, shuffle=True, num_workers=0)  # 将训练数据以每次32张图片的形式抽出进行测试
semantic_center = torch.load('ResNet_semantic_center_1112.pth')

net = resnet34()
new_m = net.to(device)
# 测试所保存的模型
m_state_dict = torch.load('Resnet_1110.pth')
new_m.load_state_dict(m_state_dict)


mytransforms = transforms.Compose([
    transforms.Resize((7, 7)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307),(0.3081))])


test1 = torch.ones(1, 1, 7, 7)  # 测试一下输出的形状大小 输入一个64,3,120,120的向量
test1.to(device)
macs, params = profile(new_m, inputs=(test1.to(device),))

print('macs:',macs,'params:',params)

f = open(r'calculate_resnet11_1112_1.txt', 'a')
print("macs:{}, params: {}".format(macs, params), file=f)
f.close()






