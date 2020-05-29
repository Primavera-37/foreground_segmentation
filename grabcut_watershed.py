import os
import cv2
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from operator import itemgetter
from collections import OrderedDict


def watershed(img, img_gt):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)  # OTSU 二值化

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, kernel, iterations=2)  # 去噪

    sure_bg = cv2.dilate(opening, kernel, iterations=3)  # 形态学求sure_bg

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)  # 求sure_fg
    ret, sure_fg = cv2.threshold(
        dist_transform, 0.7*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)  # unknown区域

    ret, markers = cv2.connectedComponents(sure_fg)  # 根据前景区域生成markers
    markers = markers + 1  # 背景是被标记为0的，所以需要进行+1
    markers[unknown == 255] = 0  # 将未知区域设置为0

    ##### 不能直接用，-1代表边界 #####
    markers = cv2.watershed(img, markers)
    return markers


def mask2bbox(mask):  # 通过truth求bbox
    lbl = label(mask)
    props = regionprops(lbl)
    bbox = [(prop.bbox[1], prop.bbox[0], prop.bbox[3] - prop.bbox[1], prop.bbox[2] - prop.bbox[0])
            for prop in props if prop.bbox[3] - prop.bbox[1] > 3 and prop.bbox[2] - prop.bbox[0] > 3]
    return bbox


def grabcut(img, img_gt):
    mask = img_gt[:, :, 0]
    mask[mask >= 1] = 1
    bbox = mask2bbox(mask)
    new_img = np.zeros(img.shape[:2]).astype(np.uint8)
    for i in range(len(bbox)):
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        mask, bgdModel, fgdModel = cv2.grabCut(
            img, mask, bbox[i], bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        index = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
        new_img[index] = 1

    plt.subplot(131)
    plt.imshow(img)
    plt.subplot(132)
    plt.imshow(mask, cmap="gray")
    plt.subplot(133)
    plt.imshow(new_img,cmap="gray")
    plt.show()  


    return new_img


img = cv2.imread("./orderedImages/30001.jpg")
img_gt = cv2.imread("./orderedTruths/30001_gt.bmp")
grabcut(img, img_gt)
