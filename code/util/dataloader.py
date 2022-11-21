"""
    假设给定的图像标签对(Xt,Yt)，对于每一对Xt。
    我们将最大灰度值表示为G。在阈值60%G, 80%G, G处对Xt进行三次截断操作，
    分别生成X1t, X2t, X3t。截断操作将灰度值高于阈值的像素映射到阈值灰度值。
    设置不同的阈值强制图像在不同灰度值窗口宽度下的特征，以避免极端灰度值的影响。
    对X1t、X2t、X3t进行灰度直方图均衡化，得到X1t、X2t、X3t。
    我们发现对灰度直方图进行均衡预处理可以使模型在训练过程中收敛更稳定。
    我们将连接的3通道图像[x1t, x2t, x3t]表示为X'。
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
import random

sys.path.append('/Users/liujiyao/Desktop/MSCMR/1 MSCMR orient/code/')
from d2l import torch as d2l



class LoadDataset(Dataset):

    #获取图像路径与数据增广
    def __init__(self, root=None, mode='train', truncation = False):

        '''
        root:data根目录, 
        mode=train/valid/test,
        truncation: 是否截断concat操作
        '''

        super(LoadDataset, self).__init__()
        self.root=root
        self.mode = mode
        self.truncation = truncation

        print(f"load data in : {self.root} as {self.mode}")

        "1.加载获取图像-mask路径列表"
        # self.folders = glob.glob(self.root+'/*/*') #多个文件路径
        # self.folders = sorted(self.folders)
        # self.folders = random.shuffle(self.folders, )
        # self.folders = os.listdir(self.root+'/0')  # 只获取0类别路径
        # self.folders = sorted(self.folders)


        # assert self.folders!=[] #是否成功读入

        "======================================="

        "2.数据规整与增广"

        self.data=[]
        if self.mode == 'train':
            for i in range(1,36):
                self.data+=glob.glob(self.root+f'/*/patient{i}*')
        elif self.mode == 'valid':
            for i in range(36,46):
                self.data+=glob.glob(self.root+f'/*/patient{i}*')
        else:
            self.data += [self.root]  # 直接输入测试图片路径
        print(self.data)
        print(f'len of data = {len(self.data)}')
        #定义数据增广
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.ToPILImage(),
                # transforms.Grayscale(num_output_channels=1),
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop((256, 256), scale=(0.7, 1), ratio=(0.8, 1.2)),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            ]),
            'valid': transforms.Compose([
                transforms.ToPILImage(),
                # transforms.Grayscale(num_output_channels=1),
                transforms.RandomResizedCrop((256, 256), scale=(0.7, 1), ratio=(0.8, 1.2)),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            ]),
            'test': transforms.Compose([
                transforms.ToPILImage(),
                # transforms.Grayscale(num_output_channels=1),
                # transforms.RandomResizedCrop((256, 256), scale=(0.7, 1), ratio=(0.8, 1.2)),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            ])
        }

    #获取数据集大小
    def __len__(self):
            return len(self.data)


    #获取数据集
    def __getitem__(self, item):

        #加载数据
        #获取原图和mask并转为tensor
        img_fname = self.data[item]
        img = cv2.imread(img_fname, 0)  # H，W

        if self.truncation:
            # 截断扩展到3通道
            img100, img80, img60 = img.copy(), img.copy(), img.copy()
            rec1, img80 = cv2.threshold(img,0.8*np.max(img),0.8*np.max(img),cv2.THRESH_TRUNC)  # 超过阈值取阈值
            rec1, img60 = cv2.threshold(img,0.6*np.max(img),0.6*np.max(img),cv2.THRESH_TRUNC)
            # 直方图均衡化
            img100 = cv2.equalizeHist(img100)
            img80 = cv2.equalizeHist(img80)
            img60 = cv2.equalizeHist(img60) 

            img = cv2.merge([img100,img80,img60])
            #维度变换：(H,W,T)-->(T,H,W)
            img = torch.tensor(img).float()
            img = img.permute(2, 0, 1)
        else:
            img = torch.tensor(img).float()

        if self.mode != 'test':
            label = int(img_fname.split('/')[-2])
            label = torch.tensor(label).long()
            inp= self.data_transforms[self.mode](img)
            return inp, label
        else:
            inp = self.data_transforms[self.mode](img)
            return inp

# if __name__=='__main__':
#
#
#     '''========参数设置========'''
#     num_workers=2
#     '''===================='''
#
#
#     root = '/Users/liujiyao/Desktop/MSCMR/1 MSCMR orient/code/data_transform/C0'
#
#     dataset=LoadDataset(root=root, mode='valid', truncation = True)
#     load_dataset=DataLoader(dataset, batch_size=32, shuffle=True,
#                                 num_workers=num_workers)
#     for i, (i1, i2) in enumerate(load_dataset):
#         print(i1.shape)
#         break

