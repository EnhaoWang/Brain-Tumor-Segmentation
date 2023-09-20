import os
import random 

#数据已经划分好，看是否再进行划分
segfilepath=r'./VOCdevkit/VOC2007/SegmentationClass'
saveBasePath=r"./VOCdevkit/VOC2007/ImageSets/Segmentation/"
 
trainval_percent=1
train_percent=0.9

temp_seg = os.listdir(segfilepath)
total_seg = []
for seg in temp_seg:
    print(seg)
    if seg.endswith(".png"):
        total_seg.append(seg)

num=len(total_seg)
print(num)
list=range(num)  
tv=int(num*trainval_percent)  
tr=int(tv*train_percent)  
trainval= random.sample(list,tv)  
train=random.sample(trainval,tr)  

print(total_seg[4][:-4])
print(list)
print("train and val size",tv)
print("train size",tr)
ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')
ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')
ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')
fval = open(os.path.join(saveBasePath,'val.txt'), 'w')

for i  in list:
    name=total_seg[i][:-4]+'\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest .close()
