import torch
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
import SimpleITK as  sitk
import nibabel as nib # nii格式一般都会用到这个包
import imageio # 转换成图像
sys.path.append('/Users/liujiyao/Desktop/MSCMR/1 MSCMR orient/code/')
from d2l import torch as d2l
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import copy
import os
import sys

class auto:
    def __init__(self, modality):
        '''

        :param modality: 'C0','LGE','T2';
        '''
        self.slices = 6
        self.imgSize = 256
        self.channels = [0.6, 0.8, 1.0]
        self.saveName = "correct"
        self.directs = [0,1,2,3,4,5,6,7]
        self.direct = ""
        self.paths = {'C0': '/checkpoints/C0/model-best.pth',
                      'LGE': '/checkpoints/LGE/model-best.pth',
                      'T2': '/checkpoints/T2/model-best.pth',
        }
        self.modality = modality

        model = nn.Sequential(
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

        self.model = model
        self.model.load_state_dict(torch.load(self.paths[self.modality]))

    def predict(self, nii_path):
        self.preProcess(nii_path)
        test_dataset = LoadDataset(root=nii_path, mode='test', truncation=True)
        test_iter = DataLoader(test_dataset, batch_size=1, shuffle=False,
                               num_workers=0)
        device = d2l.try_gpu(2)
        with torch.no_grad():
            for X in tqdm(test_iter):
                X = X.to(device)
                y_hat = self.model(X)

        predictions = np.array(y_hat).tolist()
        resultDic = dict()
        for i in range(self.slices):
            prediction = predictions[i]
            key = prediction.index(max(prediction))
            direct = self.directs[key]
            if direct in resultDic.keys():
                resultDic[direct] += 1
            else:
                resultDic[direct] = 1
        self.direct, count = sorted(list(resultDic.items()), key=lambda x: x[1], reverse=True)[0]
        return self.direct




    def adjust(self, nii_path):
        image = sitk.ReadImage(nii_path)
        self.derict = self.predict((nii_path))
        if self.direct == "":
            return 0, False
        os.remove("./"+self.saveName)
        if self.direct == 0:
            target = img  # 000 Target[x,y,z]=Source[x,y,z]
        if self.direct == 1:
            target = np.fliplr(img)  # 001 Target[x,y,z]=Source[sx-x,y,z]
        if self.direct == 2:
            target = np.flipud(img)  # 010 Target[x,y,z]=Source[x,sy-y,z]
        if self.direct == 3:
            target = np.flipud(np.fliplr(img))  # 011 Target[x,y,z]=Source[sx-x,sy-y,z]
        if self.direct == 4:
            target = img.transpose((1, 0, 2))  # 100 Target[x,y,z]=Source[y,x,z]
        if self.direct == 5:
            # 101 Target[x,y,z]=Source[sx-y,x,z] 110 Target[x,y,z]=Source[y,sy-x,z]
            # target = np.fliplr(img.transpose((1, 0, 2)))
            target = np.flipud(img.transpose((1, 0, 2)))
        if self.direct == 6:
            # 110 Target[x,y,z]=Source[y,sy-x,z] 101 Target[x,y,z]=Source[sx-y,x,z]
            # target = np.flipud(img.transpose((1, 0, 2)))
            target = np.fliplr(img.transpose((1, 0, 2)))
        if self.direct == 7:
            target = np.flipud(np.fliplr(img.transpose((1, 0, 2))))  # 111 Target[x,y,z]=Source[sx-y,sy-x,z]
        return target, True

    def preProcess(self, nii_path):
        '''
        slice nii into pngs
        :param nii_path: path of nii
        :return: png
        '''

        img = nib.load(nii_path)  # 读取nii
        img_fdata = img.get_fdata()
        fname = nii_path.split('/')[-1].replace('.nii.gz', '')  # 去掉nii的后缀名

        # 开始转换为图像
        (x, y, z) = img.shape  # 获得数据shape信息：（长，宽，维度-切片数量）
        for i in range(z):  # z是图像的序列
            silce = img_fdata[:, :, i]  #
            imageio.imwrite(os.path.join('output_slice', str(fname) + '{}.png'.format(i)), silce)


    def _bytes_feature(self, value):
        pass

    def create_features(self, image_string):
        pass

    def read_tfrecord(self, example):
        pass

    def get_batched_dataset(self):
        pass

if __name__=='__main__':
    adjust = auto(modality='C0')
    nii_path = ''
    adjust(nii_path)


