# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 12:03:52 2018

@author: yzzhao2
"""

import time
import datetime
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import network_infrared
import dataset_infrared
import utils

# This code is not run
def Trainer(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # cudnn benchmark accelerates the network
    if opt.cudnn_benchmark == True:
        cudnn.benchmark = True
    else:
        cudnn.benchmark = False

    # define network
    if opt.pre_train == True:
        net = network_infrared.GeneratorUNet()
        utils.weights_init(net)
    else:
        modelname = opt.modelname
        print('Load the model:', modelname)
        net = torch.load(modelname)

    # To device
    if opt.multi_gpu == True:
        net = nn.DataParallel(net)
        net = net.cuda()
    else:
        net = net.cuda()

    # Loss functions
    criterion = torch.nn.L1Loss().cuda()

    # Optimizers
    optimizer = torch.optim.Adam(net.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2))

    # Learning rate decrease
    def adjust_learning_rate(optimizer, epoch, opt):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        if opt.pre_train == False:
            addition_epoch = int(opt.load_epoch)
        else:
            addition_epoch = 0
        if opt.multi_gpu == True:
            lr = opt.lr * (opt.lr_decrease_factor ** ((epoch + addition_epoch) // opt.lr_decrease_epoch))
            for param_group in optimizer.module.param_groups:
                param_group['lr'] = lr
        else:
            lr = opt.lr * (opt.lr_decrease_factor ** ((epoch + addition_epoch) // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.pre_train == False:
            addition_epoch = int(opt.load_epoch)
        else:
            addition_epoch = 0
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module, 'Pre_Infrared_epoch%d_batchsize%d.pth' % ((epoch + addition_epoch), opt.batch_size))
                print('The trained model is successfully saved at epoch %d' % epoch)
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net, 'Pre_Infrared_epoch%d_batchsize%d_2gammas.pth' % ((epoch + addition_epoch), opt.batch_size))
                print('The trained model is successfully saved at epoch %d' % epoch)
    
    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Network dataset txt files
    # imglist = utils.text_readlines("KAIST_train.txt")
    lwirlist = utils.text_readlines("KAIST_train_lwir.txt")
    visiblelist = utils.text_readlines("KAIST_train_visible.txt")

    # Define the dataset
    trainset = dataset_infrared.InfraredDataset(opt.baseroot, lwirlist, visiblelist)
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #            Training and Testing
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()

    # Training loop
    for epoch in range(opt.epochs):
        for batch_idx, (data, target) in enumerate(dataloader):

            # Load data and put it to cuda
            data = data.cuda()
            target = target.cuda()

            # Train one iteration
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [L1 Loss: %.5f] time_left: %s" %
                ((epoch + 1), opt.epochs, batch_idx, len(dataloader), loss.item(), time_left))

        # Learning rate decrease
        adjust_learning_rate(optimizer, (epoch + 1), opt)

        # Save the model
        save_model(net, (epoch + 1), opt)