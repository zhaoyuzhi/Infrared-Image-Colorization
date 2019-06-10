# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 12:04:52 2018

@author: yzzhao2
"""

import torch
import torch.nn as nn
from network_module import *

#----------------------------------------------------------------------------
#                                  Generator
#----------------------------------------------------------------------------

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 3):
        super(GeneratorUNet, self).__init__()

        self.down1 = DownBlock(in_channels, 64, normalize = False)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)
        self.down5 = DownBlock(512, 512)
        self.down6 = DownBlock(512, 512)
        self.down7 = DownBlock(512, 512)
        self.down8 = DownBlock(512, 512, normalize = False)
        self.up1 = UpBlock(512, 512)
        self.up2 = UpBlock(1024, 512)
        self.up3 = FusionUp(1024, 512)
        self.up4 = FusionUp(1024, 512)
        self.up5 = FusionUp(1024, 256)
        self.up6 = UpBlock(512, 128)
        self.up7 = UpBlock(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1, bias = False),
            nn.Tanh()
        )

    def forward(self, x):

        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)                                      # out: batch * 64 * 128 * 128
        d2 = self.down2(d1)                                     # out: batch * 128 * 64 * 64
        d3 = self.down3(d2)                                     # out: batch * 256 * 32 * 32
        d4 = self.down4(d3)                                     # out: batch * 512 * 16 * 16
        d5 = self.down5(d4)                                     # out: batch * 512 * 8 * 8
        d6 = self.down6(d5)                                     # out: batch * 512 * 4 * 4
        d7 = self.down7(d6)                                     # out: batch * 512 * 2 * 2
        d8 = self.down8(d7)                                     # out: batch * 512 * 1 * 1
        u1 = self.up1(d8)                                       # out: batch * 512 * 2 * 2
        u1 = torch.cat((u1, d7), 1)                             # out: batch * (1024 = 512 + 512) * 2 * 2
        u2 = self.up2(u1)                                       # out: batch * 512 * 4 * 4
        u2 = torch.cat((u2, d6), 1)                             # out: batch * (1024 = 512 + 512) * 4 * 4
        u3 = self.up3(u2, d5)                                   # out: batch * (1024 = 512 + 512) * 8 * 8
        u4 = self.up4(u3, d4)                                   # out: batch * (1024 = 512 + 512) * 16 * 16
        u5 = self.up5(u4, d3)                                   # out: batch * (512 = 256 + 256) * 32 * 32
        u6 = self.up6(u5)                                       # out: batch * 128 * 64 * 64
        u6 = torch.cat((u6, d2), 1)                             # out: batch * (256 = 128 + 128) * 64 * 64
        u7 = self.up7(u6)                                       # out: batch * 64 * 128 * 128
        u7 = torch.cat((u7, d1), 1)                             # out: batch * (128 = 64 + 64) * 128 * 128
        # final output
        x = self.final(u7)                                      # out: batch * 3 * 256 * 256

        return x

class GeneratorUNetPix2Pix(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 3):
        super(GeneratorUNetPix2Pix, self).__init__()

        self.down1 = DownBlock(in_channels, 64, normalize = False)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)
        self.down5 = DownBlock(512, 512)
        self.down6 = DownBlock(512, 512)
        self.down7 = DownBlock(512, 512)
        self.down8 = DownBlock(512, 512, normalize = False)
        self.up1 = UpBlock(512, 512)
        self.up2 = UpBlock(1024, 512)
        self.up3 = UpBlock(1024, 512)
        self.up4 = UpBlock(1024, 512)
        self.up5 = UpBlock(1024, 256)
        self.up6 = UpBlock(512, 128)
        self.up7 = UpBlock(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1, bias = False),
            nn.Tanh()
        )

    def forward(self, x):

        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)                                      # out: batch * 64 * 128 * 128
        d2 = self.down2(d1)                                     # out: batch * 128 * 64 * 64
        d3 = self.down3(d2)                                     # out: batch * 256 * 32 * 32
        d4 = self.down4(d3)                                     # out: batch * 512 * 16 * 16
        d5 = self.down5(d4)                                     # out: batch * 512 * 8 * 8
        d6 = self.down6(d5)                                     # out: batch * 512 * 4 * 4
        d7 = self.down7(d6)                                     # out: batch * 512 * 2 * 2
        d8 = self.down8(d7)                                     # out: batch * 512 * 1 * 1
        u1 = self.up1(d8)                                       # out: batch * 512 * 2 * 2
        u1 = torch.cat((u1, d7), 1)                             # out: batch * (1024 = 512 + 512) * 2 * 2
        u2 = self.up2(u1)                                       # out: batch * 512 * 4 * 4
        u2 = torch.cat((u2, d6), 1)                             # out: batch * (1024 = 512 + 512) * 4 * 4
        u3 = self.up3(u2)                                       # out: batch * 512 * 8 * 8
        u3 = torch.cat((u3, d5), 1)                             # out: batch * (1024 = 512 + 512) * 8 * 8
        u4 = self.up4(u3)                                       # out: batch * 512 * 16 * 16
        u4 = torch.cat((u4, d4), 1)                             # out: batch * (1024 = 512 + 512) * 16 * 16
        u5 = self.up5(u4)                                       # out: batch * 256 * 32 * 32
        u5 = torch.cat((u5, d3), 1)                             # out: batch * (512 = 256 + 256) * 32 * 32
        u6 = self.up6(u5)                                       # out: batch * 128 * 64 * 64
        u6 = torch.cat((u6, d2), 1)                             # out: batch * (256 = 128 + 128) * 64 * 64
        u7 = self.up7(u6)                                       # out: batch * 64 * 128 * 128
        u7 = torch.cat((u7, d1), 1)                             # out: batch * (128 = 64 + 64) * 128 * 128
        # final output
        x = self.final(u7)                                      # out: batch * 3 * 256 * 256

        return x

##############################
#        Discriminator
##############################

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_size, out_size, normalize = True, bn = True, dropout = 0.0):
        super(DiscriminatorBlock, self).__init__()
        layers = []
        if normalize:
            if bn:
                layers.append(nn.Conv2d(in_size, out_size, 4, 2, 1, bias = False))
                layers.append(nn.BatchNorm2d(out_size))
            else:
                layers.append(nn.Conv2d(in_size, out_size, 4, 2, 1, bias = False))
                layers.append(nn.InstanceNorm2d(out_size))
        else:
            layers.append(nn.Conv2d(in_size, out_size, 4, 2, 1, bias = False))
        layers.append(nn.LeakyReLU(0.2, inplace = True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# This is a kind of PatchGAN. Patch is implied in the output. This is 70 * 70 PatchGAN
class Discriminator70(nn.Module):
    def __init__(self, G_input = 1, G_output = 2):
        super(Discriminator70, self).__init__()

        self.block1 = DiscriminatorBlock(G_input + G_output, 64, normalize = False)
        self.block2 = DiscriminatorBlock(64, 128)
        self.block3 = DiscriminatorBlock(128, 256)

        self.final = nn.Sequential(
            nn.Conv2d(256, 512, 4, padding = 1, bias = False),  # out: batch * 512 * 31 * 31
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(512, 1, 4, padding = 1, bias = False),    # out: batch * 1 * 30 * 30
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        x = torch.cat((img_A, img_B), 1)                        # out: batch * 3 * 256 * 256
        x = self.block1(x)                                      # out: batch * 64 * 128 * 128
        x = self.block2(x)                                      # out: batch * 128 * 64 * 64
        x = self.block3(x)                                      # out: batch * 256 * 32 * 32
        x = self.final(x)                                       # out: batch * 1 * 30 * 30
        return x

class Discriminator70IN(nn.Module):
    def __init__(self, G_input = 1, G_output = 3):
        super(Discriminator70IN, self).__init__()

        self.block1 = DiscriminatorBlock(G_input + G_output, 64, normalize = False)
        self.block2 = DiscriminatorBlock(64, 128, bn = False)
        self.block3 = DiscriminatorBlock(128, 256, bn = False)

        self.final = nn.Sequential(
            nn.Conv2d(256, 512, 4, padding = 1, bias = False),  # out: batch * 512 * 31 * 31
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(512, 1, 4, padding = 1, bias = False),    # out: batch * 1 * 30 * 30
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        x = torch.cat((img_A, img_B), 1)                        # out: batch * 4 * 256 * 256
        x = self.block1(x)                                      # out: batch * 64 * 128 * 128
        x = self.block2(x)                                      # out: batch * 128 * 64 * 64
        x = self.block3(x)                                      # out: batch * 256 * 32 * 32
        x = self.final(x)                                       # out: batch * 1 * 30 * 30
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels = 3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization = True):
            # Returns downsampling layers of each discriminator block
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride = 2, padding = 1, bias = False)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.ReLU(inplace = True))
            return nn.Sequential(*layers)

        self.block1 = discriminator_block(in_channels, 64)
        self.block2 = discriminator_block(64, 128)
        self.block3 = discriminator_block(128, 256)
        self.block4 = discriminator_block(256, 512)
        self.block5 = discriminator_block(512, 512)
        self.final = nn.Sequential(
            nn.Linear(25088, 1000),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(1000, 1)
        )

    def forward(self, x):
        # Concatenate image and condition image by channels to produce input
        x = self.block1(x)                                  # out: batch * 64 * 112 * 112
        x = self.block2(x)                                  # out: batch * 128 * 56 * 56
        x = self.block3(x)                                  # out: batch * 256 * 28 * 28
        x = self.block4(x)                                  # out: batch * 512 * 14 * 14
        x = self.block5(x)                                  # out: batch * 512 * 7 * 7
        x = x.view(-1, 25088)                               # out: batch * 25088
        x = self.final(x)                                   # out: batch * 1
        
        return x
