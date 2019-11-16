#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import argparse
import os
import numpy as np
import math
import sys
import pdb

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.datasets.mnist import MNIST
from lenet import LeNet5Half
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
import resnet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST','cifar10','cifar100'])
parser.add_argument('--data', type=str, default='/cache/data/')
parser.add_argument('--teacher_dir', type=str, default='/cache/models/')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')
parser.add_argument('--lr_G', type=float, default=0.2, help='learning rate')
parser.add_argument('--lr_S', type=float, default=2e-3, help='learning rate')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--oh', type=float, default=1, help='one hot loss')
parser.add_argument('--ie', type=float, default=5, help='information entropy loss')
parser.add_argument('--a', type=float, default=0.1, help='activation loss')
parser.add_argument('--output_dir', type=str, default='/cache/models/')

opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True 

accr = 0
accr_best = 0

teacher = torch.load(opt.teacher_dir + 'teacher').cuda()
teacher.eval()
criterion = torch.nn.CrossEntropyLoss().cuda()

teacher = nn.DataParallel(teacher).cuda()


def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, reduction= 'sum') / y.shape[0]
    return l_kl


if opt.dataset == 'MNIST':    
    # Configure data loader   
    net = LeNet5Half()
    net = nn.DataParallel(net).cuda()
    data_test = MNIST(opt.data,
                      train=False,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                          ]))           
    data_test_loader = DataLoader(data_test, batch_size=64, shuffle=False)
    optimizer_S = torch.optim.Adam(net.parameters(), lr=opt.lr_S)

if opt.dataset != 'MNIST':  
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if opt.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        net = resnet.ResNet18()
        net = nn.DataParallel(net).cuda()
        data_train = CIFAR10(opt.data,
                             train=True,
                             transforms=transform_train)
        data_test = CIFAR10(opt.data,
                          train=False,
                          transform=transform_test)
    if opt.dataset == 'cifar100': 
        net = resnet.ResNet18(num_classes=100)
        net = nn.DataParallel(net).cuda()
        data_test = CIFAR100(opt.data,
                          train=False,
                          transform=transform_test)
    data_test_loader = DataLoader(data_test, batch_size=opt.batch_size)
    data_train_loader = DataLoader(data_train, batch_size=128)

    optimizer_S = torch.optim.SGD(net.parameters(), lr=opt.lr_S, momentum=0.9, weight_decay=5e-4)


def adjust_learning_rate(optimizer, epoch, learing_rate):
    if epoch < 800:
        lr = learing_rate
    elif epoch < 1600:
        lr = 0.1*learing_rate
    else:
        lr = 0.01*learing_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):

    total_correct = 0
    avg_loss = 0.0
    if opt.dataset != 'MNIST':
        adjust_learning_rate(optimizer_S, epoch, opt.lr_S)

    for i, (images, labels) in enumerate(data_train_loader):
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        net.train()
        optimizer_S.zero_grad()        

        outputs_T = teacher(images)

        loss_kd = kdloss(net(images.detach()), outputs_T.detach())
        loss += loss_kd       
        loss.backward()
        optimizer_S.step() 
        if i == 1:
            print("[Epoch %d/%d] [loss_kd: %f]" % (epoch, loss_kd.item()))
            
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images = images.cuda()
            labels = labels.cuda()
            net.eval()
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), float(total_correct) / len(data_test)))
    accr = round(float(total_correct) / len(data_test), 4)
    if accr > accr_best:
        torch.save(net,opt.output_dir + 'student')
        accr_best = accr
