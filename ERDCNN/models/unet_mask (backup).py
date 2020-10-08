'''

Based on the FCRN-A in "Xie, Weidi, J. Alison Noble, and Andrew Zisserman. "Microscopy cell counting and detection with fully convolutional regression networks." Computer methods in biomechanics and biomedical engineering: Imaging & Visualization 6.3 (2018): 283-292.".

Actually not a U-Net like https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

The arch in the Figure 3 of "Liu, Qian, et al. "A novel convolutional regression network for cell counting." 2019 IEEE 7th International Conference on Bioinformatics and Computational Biology (ICBCB). IEEE, 2019." does not work.

'''


import torch
import torch.nn as nn


class UNet_mask(nn.Module):

    def __init__(self, nc_in=1, nc_out=1):
        super().__init__()

        self.main = nn.Sequential(

            # conv 1: down
            nn.Conv2d(nc_in, 32, kernel_size=3, stride=1, padding=1), #h=h; 256
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), #128

            # conv 2: down
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), #h=h; 128
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), #64

            # conv 3: down
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), #h=h; 64
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), #32

            # conv 4
            nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1), #h=h; 32
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # conv 5: up
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # h=h*2; 64
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1), #h=h; 64
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # conv 6: up
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # h=h*2; 128
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), #h=h; 128
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # conv 7: up
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # h=h*2; 256
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1), #h=h; 256
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # output
            nn.Conv2d(32, nc_out, kernel_size=3, stride=1, padding=1),
            # nn.Sigmoid()
        )


    def forward(self, x):

        out = self.main(x)

        return out




if __name__ == "__main__":
    unet_mask = UNet_mask().cuda()
    unet_mask = nn.DataParallel(unet_mask)
    N=64
    IMG_SIZE=256
    x = torch.randn(N,1,IMG_SIZE,IMG_SIZE).cuda()
    o = unet_mask(x)
    print(o.shape)
