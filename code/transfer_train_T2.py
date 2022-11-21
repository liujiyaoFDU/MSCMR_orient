"""
    Orientation Recognition Network training
    {input:3 模态

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
root = 'data_transform/T2'  # 数据路径
num_workers=8
PATH = '/home3/HWGroup/zhengyx/JY_file/1 MSCMR orient/code/checkpoints/C0/model-best.pth'  # 加载模型路径
# 冻结阶段训练参数，learning_rate和batch_size可以设置大一点
Init_Epoch          = 0
Freeze_Epoch        = 20
Freeze_batch_size   = 32
Freeze_lr           = 1e-3
# 解冻阶段训练参数，learning_rate和batch_size设置小一点
UnFreeze_Epoch      = 40
Unfreeze_batch_size = 16
Unfreeze_lr         = 1e-4

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
makedir('./checkpoints/T2')

# net = DenseNet()
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

net.load_state_dict(torch.load(PATH))

device = d2l.try_gpu(2)
print('training on', device)
net.to(device)
loss = nn.CrossEntropyLoss()


'''冻结backbone训练'''
Freeze_Train        = True
batch_size  = Freeze_batch_size
lr          = Freeze_lr
start_epoch = Init_Epoch
end_epoch   = Freeze_Epoch

dataset=LoadDataset(root=root, mode='train', truncation = True)
train_iter=DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers)
dataset=LoadDataset(root=root, mode='valid', truncation = True)
valid_iter=DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
dataset=LoadDataset(root=root, mode='test', truncation = True)
test_iter=DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=num_workers)
timer, num_batches = d2l.Timer(), len(train_iter)
if Freeze_Train:
    # 冻结全连接层之前的
    # for param in net[:12].parameters():
    for param in net[:9].parameters():
        param.requires_grad = False
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

best_acc = 0
valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
writer.add_scalar('valid acc', valid_acc, 0)
for epoch in range(start_epoch, end_epoch):
    print(f'epoch:{epoch}')
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = d2l.Accumulator(3)
    net.train()
    for i, (X, y) in tqdm(enumerate(train_iter)):
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
        with torch.no_grad():
            metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            writer.add_scalar('train loss', train_l, epoch + (i + 1) / num_batches)
            writer.add_scalar('train acc', train_acc, epoch + (i + 1) / num_batches)
    valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
    writer.add_scalar('valid acc', valid_acc, epoch + 1)

    if valid_acc>best_acc:
        torch.save(net.state_dict(), "./checkpoints/T2/model-best.pth")
        best_acc = valid_acc
    torch.save(net.state_dict(), "./checkpoints/T2/model-latest.pth")



"""finetune："""
# 解冻后训练
Freeze_Train        =False
batch_size  = Unfreeze_batch_size
lr          = Unfreeze_lr
start_epoch = Freeze_Epoch
end_epoch   = UnFreeze_Epoch

dataset=LoadDataset(root=root, mode='train', truncation = True)
train_iter=DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers)
dataset=LoadDataset(root=root, mode='valid', truncation = True)
valid_iter=DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
dataset=LoadDataset(root=root, mode='test', truncation = True)
test_iter=DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=num_workers)
timer, num_batches = d2l.Timer(), len(train_iter)
if not Freeze_Train:
    # for param in net[:12].parameters():
    for param in net[:9].parameters():
        param.requires_grad = True
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

for epoch in range(start_epoch,end_epoch):
    print(f'epoch:{epoch}')
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = d2l.Accumulator(3)
    net.train()
    for i, (X, y) in tqdm(enumerate(train_iter)):
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
        with torch.no_grad():
            metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            writer.add_scalar('train loss', train_l, epoch + (i + 1) / num_batches)
            writer.add_scalar('train acc', train_acc, epoch + (i + 1) / num_batches)
    valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
    writer.add_scalar('valid acc', valid_acc, epoch + 1)

    if valid_acc>best_acc:
        torch.save(net.state_dict(), "./checkpoints/T2/model-best.pth")
        best_acc = valid_acc
    torch.save(net.state_dict(), "./checkpoints/T2/model-latest.pth")


# print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
#       f'test acc {valid_acc:.3f}')
# print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
#       f'on {str(device)}')