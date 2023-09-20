import os
import glob
import random
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import make_grid
import torchvision.transforms as tt
import albumentations as A
from sklearn.model_selection import train_test_split

dataset_path = './input/lgg-mri-segmentation/kaggle_3m/'
# dataset_path  = './input/lgg-mri-segmentation/lmysb'


mask_files = glob.glob(dataset_path + '*/*_mask*')
image_files = [file.replace('_mask', '') for file in mask_files]

def diagnosis(mask_path):
    return 1 if np.max(cv2.imread(mask_path)) > 0 else 0

files_df = pd.DataFrame({"image_path": image_files,
                  "mask_path": mask_files,
                  "diagnosis": [diagnosis(x) for x in mask_files]})


num = 1
print('Number of images: ', files_df.shape[0])
# for file in image_files:
#     print(file)
#     img = cv2.imread(file)
#     cv2.imwrite(image_save_path+str(i)+os.path.splitext(file)[1],img)
#     num = num + 1
# print('\n\n\n')


#划分3集

trainval_df, test_df = train_test_split(files_df, stratify=files_df['diagnosis'], test_size=0.2, random_state=0)
trainval_df = trainval_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print('Size of train + validation sets: ', trainval_df.shape[0])
print('Size of test set: ', test_df.shape[0])
# train_df, val_df = train_test_split(files_df, stratify=files_df['diagnosis'], test_size=0.1, random_state=0)
# train_df = train_df.reset_index(drop=True)
# val_df = val_df.reset_index(drop=True)

def Mask_to_Label(mask):
    x, y, z = mask.shape
    for k in range(z):
        for i in range(x):
            for j in range(y):
                if mask[i, j, k] > 0:
                    mask[i, j, k] = 1

# 数据增强函数
def DataAugmentation(image_path, mask_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)

    compose = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),        
    ])

    composed = compose(image=image, mask=mask)
    new_image = composed['image']
    new_mask = composed['mask']

    return new_image, new_mask


 # A.OneOf([
        #     A.MotionBlur(p=0.2),   # 使用随机大小的内核将运动模糊应用于输入图像。
        #     A.MedianBlur(blur_limit=3, p=0.1),    # 中值滤波
        #     A.Blur(blur_limit=3, p=0.1),   # 使用随机大小的内核模糊输入图像。
        # ], p=0.2),
        # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        # 随机应用仿射变换：平移，缩放和旋转输入
        # A.ColorJitter(p=0.5), # 随机明亮对比度
        
trainval_image_path = './lgg-mri-segmentation/kaggle_3m/Images/'
trainval_mask_path = './lgg-mri-segmentation/kaggle_3m/Masks/'

#保存训练集+验证集
trainval_size = trainval_df.shape[0]
for num in range(trainval_size):
    image_path = trainval_df['image_path'].iloc[num]
    image = cv2.imread(image_path)
    saved_image_path = trainval_image_path + str(num+1) + os.path.splitext(image_path)[1]
    cv2.imwrite(saved_image_path, image)

    mask_path = trainval_df['mask_path'].iloc[num]
    mask = cv2.imread(mask_path)
    # 白色区域像素值归1化
    Mask_to_Label(mask)

    # 存新图
    saved_mask_path = trainval_mask_path + str(num+1) + os.path.splitext(mask_path)[1]
    cv2.imwrite(saved_mask_path, mask)

    # added_image, added_mask = DataAugmentation(saved_image_path, saved_mask_path)
    # added_image_path = trainval_image_path + str(num+1+trainval_size) + os.path.splitext(image_path)[1]
    # added_mask_path = trainval_mask_path + str(num+1+trainval_size) + os.path.splitext(mask_path)[1]
    # cv2.imwrite(added_image_path, added_image)
    # cv2.imwrite(added_mask_path, added_mask)

    # 验证白色区域归1化是否成功
    # b=1
    # mask = cv2.imread(mask_path)
    # mask1 = cv2.imread('./lgg-mri-segmentation/kaggle_3m/Masks/'+str(num+1)+'_mask'+os.path.splitext(mask_path)[1])
    # for k in range(z):
    #     for i in range(x):
    #         for j in range(y):
    #             # if (mask[i,j,k] == 0 and mask1[i,j,k] == 0) or (mask[i,j,k] == 255 and mask1[i,j,k] == 1):
    #             if mask1[i,j,k] == 1:
    #                 b=0
    #             # else:
    #             #     b=0
    # print(b,end='')

test_image_path = './test_dataset/images/'
test_mask_path = './test_dataset/labels_visualize/'
test_label_path = './test_dataset/labels/'
#保存测试集
test_size = test_df.shape[0]
for num in range(test_size):
    image_path = test_df['image_path'].iloc[num]
    image = cv2.imread(image_path)
    saved_image_path = test_image_path + str(num + 1) + os.path.splitext(image_path)[1]
    cv2.imwrite(saved_image_path, image)

    mask_path = test_df['mask_path'].iloc[num]
    mask = cv2.imread(mask_path)

    # saved_mask_path = test_mask_path + str(num + 1) + '_mask' + os.path.splitext(mask_path)[1]
    # saved_label_path = test_label_path + str(num + 1) + '_mask' + os.path.splitext(mask_path)[1]
    saved_mask_path = test_mask_path + str(num + 1) + os.path.splitext(mask_path)[1]
    saved_label_path = test_label_path + str(num + 1) + os.path.splitext(mask_path)[1]
    cv2.imwrite(saved_mask_path, mask)
    Mask_to_Label(mask)
    cv2.imwrite(saved_label_path, mask)


    added_image, added_mask = DataAugmentation(saved_image_path, saved_mask_path)
    added_image_path = test_image_path + str(num + 1 + test_size) + os.path.splitext(image_path)[1]
    # added_mask_path = test_mask_path + str(num + 1 + test_size) + '_mask' + os.path.splitext(mask_path)[1]
    # added_label_path = test_label_path + str(num + 1 + test_size) + '_mask' + os.path.splitext(mask_path)[1]
    added_mask_path = test_mask_path + str(num + 1 + test_size) + os.path.splitext(mask_path)[1]
    added_label_path = test_label_path + str(num + 1 + test_size) + os.path.splitext(mask_path)[1]
    cv2.imwrite(added_image_path, added_image)
    cv2.imwrite(added_mask_path, added_mask)
    Mask_to_Label(added_mask)
    cv2.imwrite(added_label_path, added_mask)

