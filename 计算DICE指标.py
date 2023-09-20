
import os

import numpy as np
import cv2

label_path=r"test_dataset/labels_visualize"
pre_path=r"test_dataset_output"

label_files=os.listdir(label_path)
pre_files=os.listdir(pre_path)

def Dice(inp, target, eps=1):
    overlap = np.sum(inp * target)
    # print(overlap)
    # print((2. * overlap) / (np.sum(inp) + np.sum(target) + eps))
    return np.clip(((2. * overlap) / (np.sum(inp) + np.sum(target) + eps)), 1e-4, 0.9999)

dice_score=[]
morethan=1
less=1
for i ,o in zip(label_files,pre_files):
    aa=cv2.imread(os.path.join(label_path,i), cv2.THRESH_BINARY)
    bb=cv2.imread(os.path.join(pre_path,o), cv2.THRESH_BINARY)

    res = cv2.imread(os.path.join(label_path,i), cv2.THRESH_BINARY).clip(0, 1)
    label = cv2.imread(os.path.join(pre_path,o), cv2.THRESH_BINARY).clip(0, 1)
    single_dice=Dice(res,label)
    if single_dice>0.0001:
        dice_score.append(single_dice)
        less+=1
    if single_dice>0.9:
        morethan+=1
    print(single_dice,i)

print(np.mean(dice_score))
print(morethan)
print(less)
#0.9231996734751797