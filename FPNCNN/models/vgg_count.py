'''

based on https://github.com/cxhernandez/cellcount/blob/master/cellcount/models.py

'''

import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        # "flatten" the C * H * W values into a single vector per image
        return x.view(N, -1)



class Counter(nn.Module):
    def __init__(self, h, w, model_class='VGG19', c=1):
        super(Counter, self).__init__()

        self.cfg = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [
                64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512,
                'M'
            ],
            'VGG19': [
                64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512,
                512, 512, 512, 'M'
            ],
        }

        in_channels = c

        arch = []
        for v in self.cfg[model_class]:
            if v == 'M':
                arch += [nn.MaxPool2d(kernel_size=2, stride=2)]
                h = (h - 2) // 2 + 1
                w = (w - 2) // 2 + 1
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                arch += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(inplace=True)]
                in_channels = v

        self.vgg = nn.Sequential(*arch)

        self.fc_mean = nn.Sequential(Flatten(), nn.Linear(h * w * in_channels, 1024),
                                     nn.BatchNorm1d(1024), nn.LeakyReLU(inplace=True),
                                     nn.Linear(1024, 512), nn.BatchNorm1d(512),
                                     nn.LeakyReLU(inplace=True), nn.Linear(512, 1),
                                     nn.ReLU(inplace=True))

        self.fc_lvar = nn.Sequential(Flatten(), nn.Linear(h * w * in_channels, 1024),
                                     nn.BatchNorm1d(1024), nn.LeakyReLU(inplace=True),
                                     nn.Linear(1024, 512), nn.BatchNorm1d(512),
                                     nn.LeakyReLU(inplace=True), nn.Linear(512, 1))

    def forward(self, x):
        means, lv = x
        out = self.vgg(means[-1])
        # return self.fc_mean(out), self.fc_lvar(out)
        return self.fc_mean(out)

