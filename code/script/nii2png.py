"""
    功能: 拆分nii,转化为单张的png
    
    disadvantage: Lossy conversion from float64 to uint8. 
                  eg. Range [0.0, 1535.0]. Convert image to uint8 prior to saving to suppress this warning.
"""

import numpy as np
import os # 遍历文件夹
import nibabel as nib # nii格式一般都会用到这个包
import imageio # 转换成图像

def nii_to_image(niifile):
    filenames = os.listdir(niifile) # 读取nii文件夹
    slice_trans = []

    for f in filenames:
        # 开始读取nii文件
        img_path = os.path.join(filepath, f)
        img = nib.load(img_path) # 读取nii
        img_fdata = img.get_fdata()
        fname = f.replace('.nii.gz', '') # 去掉nii的后缀名
        # img_f_path = os.path.join(imgfile, fname)
        img_f_path = os.path.join(imgfile)
        # 创建nii对应的图像的文件夹
        # if not os.path.exists(img_f_path):
        # os.mkdir(img_f_path)  # 新建文件夹

        # 开始转换为图像
        (x, y, z) = img.shape #获得数据shape信息：（长，宽，维度-切片数量）（511，511,3）
        for i in range(z): # z是图像的序列,去掉部分腹部
            silce = img_fdata[:, :, i] # 
            imageio.imwrite(os.path.join(img_f_path, str(fname)+'{}.png'.format(i)), silce)
            # print('已完成'+str(i))
            # 保存图像

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

if __name__ == '__main__':
    filepath = '/Users/liujiyao/Desktop/MSCMR/1 MSCMR orient/data_adjusted/T2'
    imgfile = '/Users/liujiyao/Desktop/MSCMR/1 MSCMR orient/code/data_transform/T2/0'
    makedir(imgfile)
    nii_to_image(filepath) 