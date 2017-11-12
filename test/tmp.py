## -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import voc_utils
from skimage.io import imread
from skimage.io import imshow
import matplotlib.pyplot as plt


imgPath = r"E:\VOCtrainval_11-May-2012\VOCdevkit\VOC2012"
imgSegIndex = imgPath + r"\ImageSets\Segmentation"
imgSetPath = imgPath + r"\JPEGImages"
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

print  voc_utils.list_image_sets()
