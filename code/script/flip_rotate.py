"""
    功能: 切片、旋转、翻转变换，生成训练、测试数据集

    原始方向
        [[1,2],
        [3,4]]
    变换方向
        No	Operation	Representation	Description
        000	initial state	(1,2,3,4)	Target[x,y,z]=Source[x,y,z]
        001	horizontal flip	(2,1,4,3)	Target[x,y,z]=Source[sx-x,y,z]
        010	vertical flip	(3,4,1,2)	Target[x,y,z]=Source[x,sy-y,z]
        011	Rotate 180° clockwise	(4,3,2,1)	Target[x,y,z]=Source[sx-x,sy-y,z]
        100	Flip along the upper left-lower right corner	(1,3,2,4)	Target[x,y,z]=Source[y,x,z]
        101	Rotate 90° clockwise	(3,1,4,2)	Target[x,y,z]=Source[sx-y,x,z]
        110	Rotate 270° clockwise	(2,4,1,3)	Target[x,y,z]=Source[y,sy-x,z]
        110	Flip along the bottom left-top right corner	(4,2,3,1)	Target[x,y,z]=Source[sx-y,sy-x,z]
    How to use:
        1.修改root和target路径
"""

import argparse
import time
import cv2
import os
import datetime
import numpy as np
from multiprocessing import Process
from tqdm import tqdm

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def Fn_flip_rotate(file:str, typeit:int):
    """图像变换"""
    savepath= os.path.join(target,str(typeit))
    makedir(savepath)
    img = cv2.imread(os.path.join(root, file),cv2.IMREAD_GRAYSCALE)

    if typeit == 0:
        # 加载图像并变换
        return
    elif typeit == 1:
        new= cv2.flip(img, 1)  # 左右翻转
    elif typeit == 2:
        new= cv2.flip(img, 0)  # 上下翻转
    elif typeit == 3:
        new = cv2.rotate(img, cv2.ROTATE_180)  # 180度旋转
    elif typeit == 4:
        new = filp1324(img)  # Flip along the upper left-lower right corner	(1,3,2,4)
    elif typeit == 5:
        new = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # Rotate 90° clockwise	(3,1,4,2)
    elif typeit == 6:
        new = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Rotate 270° clockwise	(2,4,1,3)
    elif typeit == 7:
        new = filp4231(img)    # Flip along the bottom left-top right corner	(4,2,3,1)
    cv2.imwrite(os.path.join(savepath, file), new)


			

def filp4231(img):
    "左上角右下角调换图像"

    # 获取图片基本信息
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    # mode = imgInfo[2]
    
    # 获取原图的第一个镜像为了下面以横轴为主进行调换操作
    dst1 = np.zeros([width, height],np.uint8)
    # np.copyto(dst1,img)
    
    # 以横轴为主进行图像对角调换
    for i in range(0,width):
        # 此处的分界点是竖直方向的一半
        for j in range(0,int(height)):
            dst1[i][j] = img[height-1-j][width-1-i]

    return dst1

def filp1324(img):
    "左下角右上角调换图像"

    # 获取图片基本信息
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    # mode = imgInfo[2]
    
    # 获取原图的第一个镜像为了下面以横轴为主进行调换操作
    dst1 = np.zeros([width, height], np.uint8)
    # np.copyto(dst1,img)
    
    # 以横轴为主进行图像对角调换
    for i in range(0,width):
        # 此处的分界点是竖直方向的一半
        for j in range(0,int(height)):
            dst1[width-1-i][j] = img[j][width-1-i]

    return dst1



if __name__=="__main__":
    root = '/Users/liujiyao/Desktop/MSCMR/1 MSCMR orient/code/data_transform/T2/0'
    target = '/Users/liujiyao/Desktop/MSCMR/1 MSCMR orient/code/data_transform/T2'      
    datalist = os.listdir(root)
    for file in tqdm(datalist):
        for i in range(1,8):
            Fn_flip_rotate(file, typeit=i)

