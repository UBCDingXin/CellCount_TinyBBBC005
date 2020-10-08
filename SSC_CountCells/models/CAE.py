"""

Convolutional Auto-Encoder

https://github.com/rasbt/deep-learning-book/blob/master/code/model_zoo/pytorch_ipynb/autoencoder-conv-2.ipynb

"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 21:01:47 2018

@author: xin
"""

import torch.nn as nn
import torch


#questions
#should we use batch norm??

# Reinitialize weights using He initialization


class ConvAE(nn.Module):
    def __init__(self, latent_size=100, in_channels=1, dec_channels=32, ngpu=1):
        #num_input is the number of images fed into autoencoder
        super(ConvAE, self).__init__()
        self.ngpu=ngpu
        self.latent_size=latent_size
        self.in_channels=in_channels
        self.dec_channels=dec_channels
        
        self.encoder_conv = nn.Sequential(
                #conv 1
                nn.Conv2d(in_channels, dec_channels, 4, stride=2, padding=1), #h=h/2; 150
                nn.BatchNorm2d(dec_channels),
                nn.LeakyReLU(0.2,inplace=True),
                #conv 2
                nn.Conv2d(dec_channels, dec_channels*2, 4, stride=2, padding=1), #h=h/2; 75
                nn.BatchNorm2d(dec_channels*2),
                nn.LeakyReLU(0.2,inplace=True),
                #conv 3
                nn.Conv2d(dec_channels*2, dec_channels*4, 4, stride=2, padding=1), #h=floor(h/2); 37
                nn.BatchNorm2d(dec_channels*4),
                nn.LeakyReLU(0.2,inplace=True),
                #conv 4
                nn.Conv2d(dec_channels*4, dec_channels*8, 4, stride=2, padding=1), #h=floor(h/2); 18
                nn.BatchNorm2d(dec_channels*8),
                nn.LeakyReLU(0.2,inplace=True),
                #conv 5
                nn.Conv2d(dec_channels*8, dec_channels*16, 4, stride=2, padding=1), #h=floor(h/2); 9
                nn.BatchNorm2d(dec_channels*16),
                nn.LeakyReLU(0.2,inplace=True),
                #conv 6
                nn.Conv2d(dec_channels*16, dec_channels*16, 4, stride=2, padding=1), #h=floor(h/2); 4
                nn.BatchNorm2d(dec_channels*16),
                nn.LeakyReLU(0.2,inplace=True),
        )
        self.encoder_linear = nn.Sequential(
                nn.Linear((dec_channels*16)*4*4, 2048),
                nn.BatchNorm1d(2048),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Linear(2048, latent_size),
        )
#        self.encoder_linear = nn.Sequential(
#                nn.Linear((dec_channels*16)*4*4, latent_size),
#        )
        self.decoder_linear = nn.Sequential(
                nn.Linear(latent_size, 4*4*(dec_channels*16)),
        )
        self.decoder_conv = nn.Sequential(
                #TransConv 1
                nn.ConvTranspose2d(dec_channels*16, dec_channels*8, kernel_size = 4, stride = 2, padding = 1),
                #output: (h-1)*2 - 2*1 + 4 = 2*h; 8
                nn.BatchNorm2d(dec_channels*8),
                nn.LeakyReLU(0.2,inplace=True),
                #TransConv 2
                nn.ConvTranspose2d(dec_channels*8, dec_channels*4, kernel_size = 4, stride = 2, padding = 1),
                #output: (h-1)*2 - 2*1 + 4 = 2*h; 16
                nn.BatchNorm2d(dec_channels*4),
                nn.LeakyReLU(0.2,inplace=True),
                #TransConv 3
                nn.ConvTranspose2d(dec_channels*4, dec_channels*2, kernel_size = 5, stride = 2, padding = 0),
                #output: (h-1)*2 - 2*0 + 5; 35
                nn.BatchNorm2d(dec_channels*2),
                nn.LeakyReLU(0.2,inplace=True),
                #TransConv 4
                nn.ConvTranspose2d(dec_channels*2, dec_channels, kernel_size = 7, stride = 2, padding = 0),
                #output: (h-1)*2 - 2*0 + 7; 75
                nn.BatchNorm2d(dec_channels),
                nn.LeakyReLU(0.2,inplace=True),
                #TransConv 5: 
                nn.ConvTranspose2d(dec_channels, dec_channels, kernel_size = 4, stride = 2, padding = 1),
                #output: (h-1)*2 - 2*1 + 4 = 2*h; 150
                nn.BatchNorm2d(dec_channels),
                nn.LeakyReLU(0.2,inplace=True),
                #TransConv 6
                nn.ConvTranspose2d(dec_channels, 1, kernel_size = 4, stride = 2, padding = 1),
                #output: (h-1)*2 - 2*1 + 4 = 2*h; 300
                nn.Tanh()
        )
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.detach())
                m.bias.detach().zero_()
            elif isinstance(m, torch.nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.detach())
                m.bias.detach().zero_()
            elif isinstance(m, torch.nn.Linear):
                nn.init.kaiming_normal_(m.weight.detach())
                m.bias.detach().zero_()

    def forward(self, imgs):
        if imgs.is_cuda and self.ngpu > 1:
            features = nn.parallel.data_parallel(self.encoder_conv, imgs, range(self.ngpu))
            features = features.view(features.size(0), -1)
            features = nn.parallel.data_parallel(self.encoder_linear, features, range(self.ngpu))
            features = nn.parallel.data_parallel(self.decoder_linear, features, range(self.ngpu))
            features = features.view(features.size(0), self.dec_channels*16, 4, 4)
            reconst_imgs = nn.parallel.data_parallel(self.decoder_conv, features, range(self.ngpu))
        else:
            features = self.encoder_conv(imgs)
            features = features.view(features.size(0), -1)
            features = self.encoder_linear(features)
            features = self.decoder_linear(features)
            features = features.view(features.size(0), self.dec_channels*16, 4, 4)
            reconst_imgs = self.decoder_conv(features)
        return reconst_imgs
    
#    def Encoder(self, imgs):
#        if imgs.is_cuda and self.ngpu > 1:
#            features = nn.parallel.data_parallel(self.encoder_conv, imgs, range(self.ngpu))
#            features = features.view(features.size(0), -1)
#            features = nn.parallel.data_parallel(self.encoder_linear, features, range(self.ngpu))
#        else:
#            features = self.encoder_conv(imgs)
#            features = features.view(features.size(0), -1)
#            features = self.encoder_linear(features)
#        return features
#    
#    def Decoder(self, features):
#        if features.is_cuda and self.ngpu > 1:
#            features = nn.parallel.data_parallel(self.decoder_linear, features, range(self.ngpu))
#            features = features.view(features.size(0), self.dec_channels, 4, 4)
#            reconst_imgs = nn.parallel.data_parallel(self.decoder_conv, features, range(self.ngpu))
#        else:
#            features = self.decoder_linear(features)
#            features = features.view(features.size(0), 16, 4, 4)
#            reconst_imgs = self.decoder_conv(features)
#        return reconst_imgs

if __name__=="__main__":
    n = 5
    net = ConvAE(latent_size=100, ngpu=1).cuda()
    x = torch.randn(n,1, 300, 300).cuda()
    print(net(x).size())


