'''

based on https://github.com/cxhernandez/cellcount/blob/master/cellcount/models.py

'''

import torch
import torch.nn as nn


def ConvBNReLUPool(i,
                   o,
                   bn=True,
                   kernel_size=(3, 3),
                   stride=1,
                   padding=0,
                   p=0.5,
                   pool=False,
                   leaky=True):
    model = [nn.Conv2d(i, o, kernel_size=kernel_size, stride=stride, padding=padding)]

    if bn:
        model += [nn.BatchNorm2d(o)]

    if leaky:
        model += [nn.LeakyReLU(inplace=True)]
    else:
        model += [nn.ReLU(inplace=True)]

    if p > 0.:
        model += [nn.Dropout2d(p)]

    if pool:
        model += [nn.MaxPool2d(2, ceil_mode=True)]

    return nn.Sequential(*model)


class FPN(torch.nn.Module):
    def __init__(self, height, width, h=4, ratio=2, d=128):
        """
        Initialize Feature Pyramid Network (FPN)
        """
        super(FPN, self).__init__()

        # self.conv_1 = ConvBNReLUPool(3, 1, padding=1, p=0.)
        self.h = h
        self.d = d

        heights, widths = [], []
        for i in range(self.h):
            height, width = height // ratio, width // ratio
            heights.append(height)
            widths.append(width)
            setattr(self, 'down%s' % i, nn.AdaptiveAvgPool2d(output_size=(height, width)))

        for i in range(1, self.h):
            setattr(self, 'across%s' % i,
                    ConvBNReLUPool(1, self.d, p=0., padding=1, kernel_size=(3, 3)))
            height, width = heights[-(i + 1)], widths[-(i + 1)]
            setattr(self, 'up%s' % i, nn.UpsamplingBilinear2d(size=(height, width)))

        for i in range(self.h):
            setattr(
                self, 'conv_2_%s' % i,
                nn.Sequential(
                    ConvBNReLUPool(self.d, 2 * self.d, p=0., padding=1, kernel_size=(3, 3)),
                    ConvBNReLUPool(2 * self.d, 2 * self.d, p=0., padding=1, kernel_size=(3, 3)),
                    ConvBNReLUPool(2 * self.d,
                                   1,
                                   leaky=False,
                                   bn=False,
                                   p=0.,
                                   padding=1,
                                   kernel_size=(3, 3))))

        for i in range(self.h):
            setattr(
                self, 'conv_3_%s' % i,
                nn.Sequential(
                    ConvBNReLUPool(self.d, 2 * self.d, p=0., padding=1, kernel_size=(3, 3)),
                    ConvBNReLUPool(2 * self.d, 2 * self.d, p=0., padding=1, kernel_size=(3, 3)),
                    ConvBNReLUPool(2 * self.d, 1, bn=False, p=0., padding=1, kernel_size=(3, 3))))

    def forward(self, x):
        """
        Foward Pass through FPN
        """
        down_sampled = [x]
        for i in range(self.h):
            down_sampled.append(getattr(self, 'down%s' % i)(down_sampled[-1]))

        up_sampled = [down_sampled[-1].repeat(1, self.d, 1, 1)]
        for i in range(1, self.h):
            up_2 = getattr(self, 'across%s' % i)(down_sampled[-(i + 1)])
            up_1 = getattr(self, 'up%s' % i)(up_sampled[-1])
            up_sampled.append(up_1 + up_2)

        up_sampled_mean = []
        up_sampled_log_var = []
        for i, item in enumerate(up_sampled):
            up_sampled_mean.append(getattr(self, 'conv_2_%s' % i)(item))
            up_sampled_log_var.append(getattr(self, 'conv_3_%s' % i)(item))

        return up_sampled_mean, up_sampled_log_var
