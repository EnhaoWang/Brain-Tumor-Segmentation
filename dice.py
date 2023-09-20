import numpy as np
import os
import cv2

def only_image(inp):
    outp=[]
    for x in inp:
        if x.endswith('.tif'):
            outp.append(x)
    return outp

def Dice(inp, target, eps=1):
    overlap = np.sum(inp * target)
    return np.clip(((2. * overlap) / (np.sum(inp) + np.sum(target) + eps)), 1e-4, 0.9999)


def acc_cal(epoch, lr):
    res = os.listdir('./test_dataset_output/')
    res_path = only_image(res)
    label = os.listdir('./test_dataset/labels_visualize/')
    label_path = only_image(label)
    num = [x.split('_')[0] for x in label_path]
    acc_total = []
    acc = []
    temp = []
    each_max_filename = ""
    each_max = 0
    for idx, filename in enumerate(res_path):
        if filename == '.ipynb_checkpoints':
            continue
        if idx == len(num)-1:
            break
        elif num[idx] != num[idx+1]:
            acc.append('Heart {}, Mean Acc: {}, Max Acc: {}, Max file: {}'.format(num[idx], np.mean(temp), np.max(temp), each_max_filename))
            temp = []
            each_max = 0
            each_max_filename = ""
        res = cv2.imread('./test_dataset_output/' + filename, cv2.THRESH_BINARY).clip(0, 1)
        label = cv2.imread('./test_dataset/labels_visualize/' + label_path[idx], cv2.THRESH_BINARY).clip(0, 1)
        dice = Dice(res, label)
        if dice > 0.0001:
            acc_total.append(dice)
            temp.append(dice)
            if dice > each_max:
                each_max = dice
                each_max_filename = filename
#         acc_total.append(dice)
#         temp.append(dice)
    print(len(temp))
    print(len(acc_total))
    acc.append('Heart {}, Mean Acc: {}, Max Acc: {}, Max file: {}'.format(num[-1], np.mean(temp), np.max(temp), each_max_filename))
    acc.append('Total Mean Acc: {}, Max Acc: {}'.format(np.mean(acc_total), np.max(acc_total)))
    
#     f = open('acc_u2netd1_no_normal_0.0002_b8_originalData.txt', 'a')
#     f.write(str(epoch) + '\n')
#     f.write(str(lr) + '\n')
    for accuracy in acc:
#         f.write(str(accuracy) + '\n')
        print('\n'+accuracy)
#         print(np.mean(acc_total))
#         print(len(acc_total))
#     f.write('\n')
#     f.close()
    return np.mean(acc_total)


acc_cal(1, 1)