# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 12:04:52 2018

@author: yzzhao2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from spectralnorm import SpectralNorm

#----------------------------------------------------------------------------
#                              Generator blocks
#----------------------------------------------------------------------------

class GammaBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(GammaBlock, self).__init__()

        self.base_conv = SpectralNorm(nn.Conv2d(in_size, out_size, 1, 1, 0))
        self.power_conv = SpectralNorm(nn.Conv2d(in_size, out_size, 1, 1, 0))
        # self.coeff_conv = SpectralNorm(nn.Conv2d(in_size, out_size, 1, 1, 0))
        self.sigmoid = nn.Sigmoid()
        self.avgpoolsig = nn.Sigmoid()

    def forward(self, x):
        
        # Convolution operations
        base_fea = self.base_conv(x)
        power_fea = self.power_conv(x)
        # coeff_fea = self.coeff_conv(x)

        # Get the attention representation
        base_fea = self.sigmoid(base_fea)
        power_fea = self.avgpoolsig(power_fea).expand_as(base_fea)
        # coeff_fea = self.avgpoolsig(coeff_fea).expand_as(base_fea)

        # Gamma operations
        gamma_fea = torch.pow(base_fea, power_fea)
        # gamma_fea = gamma_fea.mul(coeff_fea)

        return gamma_fea

class LogBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(LogBlock, self).__init__()

        self.base_conv = SpectralNorm(nn.Conv2d(in_size, out_size, 1, 1, 0))
        self.logarithm_conv = SpectralNorm(nn.Conv2d(in_size, out_size, 1, 1, 0))
        # self.coeff_conv = SpectralNorm(nn.Conv2d(in_size, out_size, 1, 1, 0))
        self.sigmoid = nn.Sigmoid()
        self.avgpoolsig = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        # Convolution operations
        base_fea = self.base_conv(x)
        logarithm_fea = self.logarithm_conv(x)
        # coeff_fea = self.coeff_conv(x)

        # Get the attention representation
        base_fea = self.sigmoid(base_fea)
        logarithm_fea = self.avgpoolsig(logarithm_fea).expand_as(base_fea)
        # coeff_fea = self.avgpoolsig(coeff_fea).expand_as(base_fea)

        # Log operations
        log_base_fea = torch.log1p(logarithm_fea * base_fea)
        log_logarithm_fea = torch.log1p(logarithm_fea)
        log_fea = log_base_fea / log_logarithm_fea
        # log_fea = log_fea.mul(coeff_fea)

        return log_fea

class LaplaceBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(LaplaceBlock, self).__init__()

        self.lap1 = nn.Parameter(torch.FloatTensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]).expand(out_size, out_size, 3, 3), requires_grad = False)
        self.lap2 = nn.Parameter(torch.FloatTensor([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]).expand(out_size, out_size, 3, 3), requires_grad = False)
        self.lap1_conv = SpectralNorm(nn.Conv2d(in_size, out_size, 3, 1, 1))
        self.lap2_conv = SpectralNorm(nn.Conv2d(in_size, out_size, 3, 1, 1))
        self.coeff_conv = SpectralNorm(nn.Conv2d(out_size * 2, out_size, 3, 1, 1))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        # Convolution operations
        lap1_fea = self.lap1_conv(x)
        lap2_fea = self.lap2_conv(x)

        # Laplace operations
        lap1_fea = self.sigmoid(lap1_fea)
        lap1_fea = F.conv2d(lap1_fea, self.lap1, stride = 1, padding = 1)
        lap2_fea = self.sigmoid(lap2_fea)
        lap2_fea = F.conv2d(lap2_fea, self.lap2, stride = 1, padding = 1)

        # Combination
        lap_fea = torch.cat((lap1_fea, lap2_fea), 1)
        lap_fea = self.coeff_conv(lap_fea)

        return lap_fea

class DownBlock(nn.Module):
    def __init__(self, in_size, out_size, normalize = False):
        super(DownBlock, self).__init__()

        layers = []
        layers.append(SpectralNorm(nn.Conv2d(in_size, out_size, 4, 2, 1, bias = True)))
        layers.append(nn.LeakyReLU(0.2, inplace = True))
        if normalize:
            layers.append(SpectralNorm(nn.Conv2d(out_size, out_size, 3, 1, 1, bias = False)))
            layers.append(nn.InstanceNorm2d(out_size))
        else:
            layers.append(SpectralNorm(nn.Conv2d(out_size, out_size, 3, 1, 1, bias = True)))
        layers.append(nn.LeakyReLU(0.2, inplace = True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UpBlock(nn.Module):
    def __init__(self, in_size, out_size, normalize = False):
        super(UpBlock, self).__init__()

        layers = []
        layers.append(SpectralNorm(nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias = True)))
        layers.append(nn.LeakyReLU(0.2, inplace = True))
        if normalize:
            layers.append(SpectralNorm(nn.Conv2d(out_size, out_size, 3, 1, 1, bias = False)))
            layers.append(nn.InstanceNorm2d(out_size))
        else:
            layers.append(SpectralNorm(nn.Conv2d(out_size, out_size, 3, 1, 1, bias = True)))
        layers.append(nn.LeakyReLU(0.2, inplace = True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class FusionUp(nn.Module):
    def __init__(self, in_size, out_size, normalize = False):
        super(FusionUp, self).__init__()

        self.gammablock = GammaBlock(in_size, in_size // 4)         # GammaBlock
        self.logblock = GammaBlock(in_size, in_size // 4)             # LogBlock
        self.up_conv = UpBlock(in_size, in_size // 4)
        self.up_sample = SpectralNorm(nn.Conv2d(in_size, in_size // 4, 1, 1, 0))
        self.fusion_conv = SpectralNorm(nn.Conv2d(in_size, out_size, 3, 1, 1))

    def forward(self, x, skip_x):

        # 4 kinds of operations: gamma, log, upsample, upconv
        gamma = self.gammablock(x)
        gamma = F.interpolate(gamma, scale_factor = 2, mode = 'nearest')
        log = self.logblock(x)
        log = F.interpolate(log, scale_factor = 2, mode = 'nearest')
        upsample = self.up_sample(x)
        upsample = F.interpolate(upsample, scale_factor = 2, mode = 'nearest')
        up_conv = self.up_conv(x)
        # Fusion convolution ensuring the dimension equals to out_size
        fusion = torch.cat((gamma, log, upsample, up_conv), 1)
        fusion = self.fusion_conv(fusion)
        x = torch.cat((fusion, skip_x), 1)
        
        return x

'''
a = torch.randn(1, 3, 64, 64)
net = FusionUp(3, 64)
b = net(a)
print(b)
print(net)
'''