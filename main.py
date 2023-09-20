import cv2

if __name__ == '__main__':
    with open(r"lgg-mri-segmentation/kaggle_3m/ImageSets/Segmentation/train.txt", "r") as f:
        train_lines = f.readlines()

    print(train_lines)
    print(len(train_lines))



