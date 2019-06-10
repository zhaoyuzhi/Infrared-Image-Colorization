# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 19:36:00 2018

@author: yzzhao2
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

def read_img(root, transform):
    img = Image.open(root).convert('L')
    img = img.resize((256, 256), Image.ANTIALIAS)
    img = transform(img)
    img = img.reshape([1, 1, 256, 256])
    return img

def test_by_img(root, transform, model):
    img = read_img(root, transform)
    img = img.cuda()
    model = model.cuda()
    output = model(img)                                     # out: 1 * 3 * 256 * 256
    output = output.cpu().detach().numpy().reshape([3, 256, 256])
    output = output.transpose(1, 2, 0)
    output = (output * 0.5 + 0.5) * 255
    output = np.array(output, dtype = np.uint8)
    return output

if __name__ == '__main__':

    imgroot = '/media/ztt/6864FEA364FE72E4/zhaoyuzhi/KAIST_spectral_zip/set00/V000/lwir/I00011.jpg'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    model = torch.load('Pre_Infrared_epoch30_batchsize16_2gammas.pth')
    print(model)

    output = test_by_img(imgroot, transform, model)
    outimg = Image.fromarray(output)
    outimg.show()
