# -------------------------------------#
#       对单张图片进行预测
# -------------------------------------#
from unet import uNet
from PIL import Image
import os
import time

uNet = uNet()
imgs = os.listdir("./test_dataset/images")

a = 1
for jpg in imgs:
    img = Image.open("./test_dataset/images/" + jpg)
    start_time = time.time()
    image = uNet.detect_image(img)
    duration = time.time() - start_time
    print("第%s / %s张，预测时间" % (a, len(imgs)), duration)
    image.save("./test_dataset_output/" + jpg)

    a += 1

# while True:
#     img = input('Input image filename:')
#     try:
#         image = Image.open(img)
#     except:
#         print('Open Error! Try again!')
#         continue
#     else:
#         r_image = pspnet.detect_image(image)
#         r_image.show()
