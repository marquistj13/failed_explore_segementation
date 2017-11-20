## -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import voc_utils
from skimage.io import imread
from skimage.io import imshow
import matplotlib.pyplot as plt

# imgPath = r"D:\dataset\VOCtrainval_11-May-2012\VOCdevkit\VOC2012"
# imgSegIndex = imgPath + r"\ImageSets\Segmentation"
# imgSetPath = imgPath + r"\JPEGImages"
#
# f = open(imgSegIndex + r"\train.txt", 'r')
# tmp = f.read()
# imgSegIndexList = tmp.split('\n')
# # print imgSegIndexList
#
# img=imread(imgSetPath+'\\'+imgSegIndexList[0]+'.jpg')
# imshow(img)
# plt.show()
#

# print  voc_utils.list_image_sets()
# ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
#  'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
#  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

image_url_list = voc_utils.get_image_url_list('aeroplane', data_type='train')
# imgs = []
# for url in image_url_list:
#     imgs.append(voc_utils.load_img(url))
#     pass
img = voc_utils.load_img(image_url_list[0])
imshow(img)
plt.show()