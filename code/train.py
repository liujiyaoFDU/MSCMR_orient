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
from tqdm import tqdm
from util.dataloader import LoadDataset
sys.path.append('/Users/liujiyao/Desktop/MSCMR/1 MSCMR orient/code/')
from d2l import torch as d2l

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

os.environ['CUDA_VISIBLE_DEVICES']='2'
root = 'data_transform/C0'
num_workers=8

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
makedir('./checkpoints/C0')

# net = DenseNet()

# from torchvision.models import resnet18, densenet121
# net = resnet18()
# num_ftrs = net.fc.in_features
# net.fc = nn.Linear(num_ftrs, 8)

# net = densenet121()
# num_ftrs = net.fc.in_features
# net.fc = nn.Linear(num_ftrs, 8)

net = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(64 * 16 * 16, 64), nn.Sigmoid(),
    nn.Linear(64, 8)
)
train_dataset=LoadDataset(root=root, mode='train', truncation = True)
train_iter=DataLoader(train_dataset, batch_size=32, shuffle=True,
                            num_workers=num_workers)
valid_dataset=LoadDataset(root=root, mode='valid', truncation = True)
valid_iter=DataLoader(valid_dataset, batch_size=32, shuffle=False,
                            num_workers=num_workers)
# test_dataset=LoadDataset(root=root, mode='test', truncation = True)
# test_iter=DataLoader(test_dataset, batch_size=1, shuffle=False,
#                             num_workers=num_workers)

lr, num_epochs = 0.01, 40
"""Train a model with a GPU (defined in Chapter 6).

Defined in :numref:`sec_lenet`"""


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


net.apply(init_weights)
device = d2l.try_gpu(2)
print('training on', device)
net.to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()
# animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
#                         legend=['train loss', 'train acc', 'test acc'])
timer, num_batches = d2l.Timer(), len(train_iter)
best_acc = 0
for epoch in range(num_epochs):
    print(f'epoch:{epoch}')
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = d2l.Accumulator(3)
    net.train()
    for i, (X, y) in tqdm(enumerate(train_iter)):
        timer.start()
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
        with torch.no_grad():
            metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
        timer.stop()
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            writer.add_scalar('train loss', train_l, epoch + (i + 1) / num_batches)
            writer.add_scalar('train acc', train_acc, epoch + (i + 1) / num_batches)
    valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
    writer.add_scalar('valid acc', valid_acc, epoch + 1)

    if valid_acc>best_acc:
        torch.save(net.state_dict(), "./checkpoints/C0/model-best.pth")
        best_acc = valid_acc
    torch.save(net.state_dict(), "./checkpoints/C0/model-latest.pth")

print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
      f'test acc {valid_acc:.3f}')
print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
      f'on {str(device)}')