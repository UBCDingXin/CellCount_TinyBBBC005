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

'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import math


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

img_size = (256, 256)

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=24):
        super(VGG, self).__init__()
        self.convs = self._make_layers(cfg[vgg_name])

        in_dim = 512 * math.floor(img_size[0]/2**5)*math.floor(img_size[1]/2**5)
        if num_classes > 1: #classification
            self.dense = nn.Linear(in_dim, num_classes)
        else: #regression
            self.dense = nn.Sequential(
                nn.Linear(in_dim, num_classes),
                nn.ReLU()
            )
        #end num_classes

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        flag = 0
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)] # h=h/2
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, stride = 1, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
                flag += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), -1)
        out = self.dense(out)
        return out

if __name__ == "__main__":
    net = VGG('VGG19').cuda()
    net = nn.DataParallel(net)
    img = torch.randn(64, 1, img_size[0], img_size[1]).cuda()
    out = net(img)
    print(out.size())
