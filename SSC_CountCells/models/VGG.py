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

img_size = (520, 696)
#img_size = (300, 300)

class VGG(nn.Module):
    def __init__(self, vgg_name, ngpu=1, num_classes=24):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.ngpu = ngpu
        self.num_classes = num_classes

        # in_dim = math.floor(img_size[0]/2**4)*math.floor(img_size[1]/2**4)
        in_dim = 512
        if num_classes > 1: #for classification
            self.classifier = nn.Linear(in_dim, num_classes)
        else:#for regression
            self.classifier = nn.Sequential(
                nn.Linear(in_dim, num_classes),
                nn.ReLU()
#                nn.Sigmoid()
            )

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        flag = 0
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)] # h=h/2
            else:
                if flag < 4:
                    stride_conv2d = 2
                else:
                    stride_conv2d = 1
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, stride = stride_conv2d, padding=1), # if stride=1 (default), h=h; if stride=2, h=floor(h/2+1/2)
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
                flag += 1
        #if all stride_conv2d = 1, then state size is about 512*16*21 which is too large; let first 4 conv layers do dimension reduction
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)] #Used in the original VGG, so h=h
        #layers += [nn.AdaptiveAvgPool2d((1,1))]
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.is_cuda and self.ngpu > 1:
            features = nn.parallel.data_parallel(self.features, x, range(self.ngpu))
            features = features.view(features.size(0), -1)
            out = nn.parallel.data_parallel(self.classifier, features, range(self.ngpu))
        else:
            features = self.features(x)
            features = features.view(features.size(0), -1)
            out = self.classifier(features)
#        if self.num_classes == 1: #only for sigmoid()
#            out = out*100 #make sure the prediction is between (0,100)
        return out, features

if __name__ == "__main__":
    net = VGG('VGG16', torch.cuda.device_count()).cuda()
    img = Variable(torch.randn(2, 1, img_size[0], img_size[1])).cuda()
    out, features = net(img)
    print(out.size())
    print(features.size())
