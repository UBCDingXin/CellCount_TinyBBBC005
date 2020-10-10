'''

codes are based on
@article{
zhang2018mixup,
title={mixup: Beyond Empirical Risk Minimization},
author={Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz},
journal={International Conference on Learning Representations},
year={2018},
url={https://openreview.net/forum?id=r1Ddp1-Rb},
}
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


img_size = (300, 300)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) #if stride=1, h=h; if stride=2, h=floor(h/2+1/2)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False) #h=h
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),#h=(h-1)/s+1
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=24, nc=1):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes

        self.main = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1, bias=False),  # h=h
            nn.BatchNorm2d(64),
            nn.ReLU(),
            self._make_layer(block, 64, num_blocks[0], stride=1),  # h=h
            nn.MaxPool2d(kernel_size=2, stride=2), #h=h/2; 128; does not exist in original ResNet model
            self._make_layer(block, 128, num_blocks[1], stride=2),  # ResNet34: h=floor(h/2+1/2); 64
            nn.MaxPool2d(kernel_size=2, stride=2), #h=h/2; 32; does not exist in original ResNet model
            self._make_layer(block, 256, num_blocks[2], stride=2),  # ResNet34: h=floor(h/2+1/2); 16
            self._make_layer(block, 512, num_blocks[3], stride=2),  # ResNet34: h=floor(h/2+1/2);8
            nn.AvgPool2d(kernel_size=4) #h/4; 2
        )

        in_dim = 512 * math.floor(img_size[0]/2**7)*math.floor(img_size[1]/2**7) #in original ResNet model, in_dim=512*block.expansion
        if num_classes > 1: #classification
            # self.dense = nn.Sequential(
            #     nn.Linear(in_dim, 512),
            #     nn.BatchNorm1d(512),
            #     nn.LeakyReLU(0.1),
            #     nn.Linear(512, num_classes),
            # )
            self.dense = nn.Linear(in_dim, num_classes)
        else: #regression
            self.dense = nn.Sequential(
                nn.Linear(in_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.1),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.1),
                nn.Linear(512, 1),
                nn.ReLU()
            )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        features = self.main(x)
        features = features.view(features.size(0), -1)
        out = self.dense(features)
        return out

def ResNet18(num_classes=24):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)

def ResNet34(num_classes=24):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)

def ResNet50(num_classes=24):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes)

def ResNet101(num_classes=24):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes)

def ResNet152(num_classes=24):
    return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes)


if __name__ == "__main__":
    net = ResNet34(num_classes=24).cuda()
    net = nn.DataParallel(net)
    img = torch.randn(32,1,img_size[0],img_size[1]).cuda()
    out = net(img)
    print(out.size())
