"""
    Orientation Recognition Network training
    {input:3 模态,

    TODO:
        先搭建一个baseline
"""

import torch
import torchvision
from torchvision import transforms
from torch import nn
import torch.utils.data
import sys

import cv2
import scipy
import glob
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from model.densenet import DenseNet
from util.dataloader import LoadDataset
sys.path.append('/Users/liujiyao/Desktop/MSCMR/1 MSCMR orient/code/')
from d2l import torch as d2l


root = '/Users/liujiyao/Desktop/MSCMR/1 MSCMR orient/code/data_transform/C0'
num_workers=0

net = DenseNet()
dataset=LoadDataset(root=root, mode='train', truncation = True)
train_dataset=DataLoader(dataset, batch_size=32, shuffle=True,
                            num_workers=num_workers)
dataset=LoadDataset(root=root, mode='valid', truncation = True)
valid_dataset=DataLoader(dataset, batch_size=32, shuffle=False,
                            num_workers=num_workers)
dataset=LoadDataset(root=root, mode='test', truncation = True)
test_dataset=DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=num_workers)

lr, num_epochs = 0.1, 40
d2l.train_ch6(net, train_dataset, valid_dataset, num_epochs, lr, d2l.try_gpu())