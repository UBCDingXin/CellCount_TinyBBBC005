# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 22:36:02 2019

@author: xin
"""

import os
# wd = 'your path/SSC_CountCells'
wd = '/home/xin/Working directory/Counting_Cells/SSC_CountCells'
# wd = '/media/qiong/icecream/SSC_CountCells'
wd = 'C:/Users/xin/Desktop/Github/SSC_CountCells'
os.chdir(wd)
import h5py
import torch
import numpy as np
import math
import random
import gc
import timeit
import scipy.misc
import PIL
#from DataAugmentation import *
from DataAugmentation2 import *
import matplotlib.pyplot as plt


Setting = "Exp2"
do_aug = False
do_filter = True
knowMC = True

fontsize = 15
xtick_labelsize = 15

# data augmentation
#Exp2
if Setting == "Exp2":
    nfake = 25
    nround = 0
    formula_dir = wd + '/data/Exp2_Formulae/'
    AugClass=[ 2,  3,  4,  6,  7,  8,  9, 11, 12, 13, 15, 16, 17, 19, 20, 21, 22,
              24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 39, 41, 42, 43, 45,
              46, 47, 49, 50, 51, 52, 54, 55, 56, 58, 59, 60, 62, 63, 64, 65, 67,
              68, 69, 71, 72, 73, 75, 76, 77, 79, 80, 81, 82, 84, 85, 86, 88, 89,
              90, 92, 93, 94, 95, 97, 98, 99]

#Exp3
if Setting == "Exp3":
    nfake = 25
    nround = 1
    formula_dir = wd + '/data/Exp3_Formulae/Round' + str(nround+1) + '/'
    deleted_classes = np.array([[14, 35, 57, 66, 83],
                                [10, 31, 70, 83, 91],
                                [18, 27, 44, 53, 91]])
    AugClass = list(deleted_classes[nround])

#Exp4
if Setting == "Exp4":
    nfake = 25
    nfake_extra = 200
    nround = 1
    formula_dir = wd + '/data/Exp4_Formulae/Round' + str(nround+1) + '/'
    deleted_classes = np.array([[61, 66, 70, 74, 78],
                                [70, 74, 78, 83, 87],
                                [83, 87, 91, 96, 100]])
    if knowMC:
        AugClass = list(deleted_classes[nround])
    else:
        AugClass = list(np.arange(deleted_classes[nround][0], deleted_classes[nround][-1]+1))

hf = h5py.File('./data/CellCount_resized_dataset.h5', 'r')
IMGs_train_raw = hf['IMGs_resized_train'].value
Blur_train_raw = hf['Blur_train'].value
Stain_train_raw = hf['Stain_train'].value
CellCount_train_raw = hf['CellCount_train'].value
IMGs_test_raw = hf['IMGs_resized_test'].value
Blur_test_raw = hf['Blur_test'].value
Stain_test_raw = hf['Stain_test'].value
CellCount_test_raw = hf['CellCount_test'].value
hf.close()

IMGs_train_raw = np.concatenate((IMGs_train_raw, IMGs_test_raw), axis=0)
Blur_train_raw = np.concatenate((Blur_train_raw, Blur_test_raw))
Stain_train_raw = np.concatenate((Stain_train_raw, Stain_test_raw))
CellCount_train_raw = np.concatenate((CellCount_train_raw, CellCount_test_raw))

#IMGs_train_raw[IMGs_train_raw<=5] = 0


if Setting in ["Exp3", "Exp4"]:
    unique_cell_count = np.array(list(set(CellCount_train_raw)))
    removed_cell_count = deleted_classes[nround]
    indx_train_removed = np.array([], dtype=np.int)
    for i in range(len(removed_cell_count)):
        indx_train_removed = np.concatenate((indx_train_removed, np.where(CellCount_train_raw==removed_cell_count[i])[0]))
    indx_train_all = np.arange(len(IMGs_train_raw))
    indx_train_left = np.array(list(set(indx_train_all).difference(set(indx_train_removed))))
    IMGs_train_raw = IMGs_train_raw[indx_train_left]
    Blur_train_raw = Blur_train_raw[indx_train_left]
    Stain_train_raw = Stain_train_raw[indx_train_left]
    CellCount_train_raw = CellCount_train_raw[indx_train_left]


if do_aug:
    IMGs_train, CellCount_train, Blur_train, Stain_train = AugmentData(IMGs_train_raw, CellCount_train_raw, Blur_train_raw, Stain_train_raw, AugClass, formula_dir, nfake=nfake, nfake_extra=nfake_extra, flipping = True, one_formula_per_class=False, show_sample_img=False, dump_fake=False, fakeImg_dir=None, do_filter=do_filter)
else:
    IMGs_train = IMGs_train_raw
    CellCount_train = CellCount_train_raw
    Blur_train = Blur_train_raw
    Stain_train = Stain_train_raw



Intensity_train = np.mean(IMGs_train, axis=(1,2,3))
Intensity_train = Intensity_train.reshape(-1,1)


plt.figure(figsize=(20, 10))
plt.subplot(231)
indx_stain_blur = np.where((Stain_train==1)*(Blur_train==1)==True)[0]
x=Intensity_train[indx_stain_blur]; y = CellCount_train[indx_stain_blur]
plt.scatter(x, y)
plt.xlim(0,100)
plt.rc('xtick',labelsize=xtick_labelsize)
plt.rc('ytick',labelsize=xtick_labelsize)
plt.xlabel("Avg. intensity", fontsize=fontsize)
plt.ylabel("Cell count", fontsize=fontsize)
plt.text(100+2, np.max(y)/2, 'stain=1', horizontalalignment='center', verticalalignment='center',rotation=-90, fontsize=fontsize)
plt.text(100/2, np.max(y)+8, 'blur=1', horizontalalignment='center', verticalalignment='center', fontsize=fontsize)

plt.subplot(232)
indx_stain_blur = np.where((Stain_train==1)*(Blur_train==23)==True)[0]
x=Intensity_train[indx_stain_blur]; y = CellCount_train[indx_stain_blur]
plt.scatter(x, y)
plt.xlim(0,100)
plt.rc('xtick',labelsize=xtick_labelsize)
plt.rc('ytick',labelsize=xtick_labelsize)
plt.xlabel("Avg. intensity", fontsize=fontsize)
plt.ylabel("Cell count", fontsize=fontsize)
plt.text(100+2, np.max(y)/2, 'stain=1', horizontalalignment='center', verticalalignment='center',rotation=-90, fontsize=fontsize)
plt.text(100/2, np.max(y)+8, 'blur=23', horizontalalignment='center', verticalalignment='center', fontsize=fontsize)

plt.subplot(233)
indx_stain_blur = np.where((Stain_train==1)*(Blur_train==48)==True)[0]
x=Intensity_train[indx_stain_blur]; y = CellCount_train[indx_stain_blur]
plt.scatter(x, y)
plt.xlim(0,100)
plt.rc('xtick',labelsize=xtick_labelsize)
plt.rc('ytick',labelsize=xtick_labelsize)
plt.xlabel("Avg. intensity", fontsize=fontsize)
plt.ylabel("Cell count", fontsize=fontsize)
plt.text(100+2, np.max(y)/2, 'stain=1', horizontalalignment='center', verticalalignment='center',rotation=-90, fontsize=fontsize)
plt.text(100/2, np.max(y)+8, 'blur=48', horizontalalignment='center', verticalalignment='center', fontsize=fontsize)

plt.subplot(234)
indx_stain_blur = np.where((Stain_train==2)*(Blur_train==1)==True)[0]
x=Intensity_train[indx_stain_blur]; y = CellCount_train[indx_stain_blur]
plt.scatter(x, y)
plt.rc('xtick',labelsize=xtick_labelsize)
plt.rc('ytick',labelsize=xtick_labelsize)
plt.xlabel("Avg. intensity", fontsize=fontsize)
plt.ylabel("Cell count", fontsize=fontsize)
plt.text(100+8, np.max(y)/2, 'stain=2', horizontalalignment='center', verticalalignment='center',rotation=-90, fontsize=fontsize)
plt.text(100/2, np.max(y)+8, 'blur=1', horizontalalignment='center', verticalalignment='center', fontsize=fontsize)

plt.subplot(235)
indx_stain_blur = np.where((Stain_train==2)*(Blur_train==23)==True)[0]
x=Intensity_train[indx_stain_blur]; y = CellCount_train[indx_stain_blur]
plt.scatter(x, y)
plt.rc('xtick',labelsize=xtick_labelsize)
plt.rc('ytick',labelsize=xtick_labelsize)
plt.xlabel("Avg. intensity", fontsize=fontsize)
plt.ylabel("Cell count", fontsize=fontsize)
plt.text(100+8, np.max(y)/2, 'stain=2', horizontalalignment='center', verticalalignment='center',rotation=-90, fontsize=fontsize)
plt.text(100/2, np.max(y)+8, 'blur=23', horizontalalignment='center', verticalalignment='center', fontsize=fontsize)

plt.subplot(236)
indx_stain_blur = np.where((Stain_train==2)*(Blur_train==48)==True)[0]
x=Intensity_train[indx_stain_blur]; y = CellCount_train[indx_stain_blur]
plt.scatter(x, y)
plt.rc('xtick',labelsize=xtick_labelsize)
plt.rc('ytick',labelsize=xtick_labelsize)
plt.xlabel("Avg. intensity", fontsize=fontsize)
plt.ylabel("Cell count", fontsize=fontsize)
plt.text(100+8, np.max(y)/2, 'stain=2', horizontalalignment='center', verticalalignment='center',rotation=-90, fontsize=fontsize)
plt.text(100/2, np.max(y)+8, 'blur=48', horizontalalignment='center', verticalalignment='center', fontsize=fontsize)

plt.tight_layout()
plt.savefig(Setting + '_round_' + str(nround) +'_cellcount_vs_intensity_doAug_' + str(do_aug) + '_filtering_' + str(do_filter) + '_knowMC_' + str(knowMC) + '.png', format="png")
plt.show()



#plt.figure(figsize=(20, 10))
#plt.subplot(121)
#indx_stain_blur = np.where((Stain_train==1)*(Blur_train==1)==True)[0]
#x=Intensity_train[indx_stain_blur]; y = CellCount_train[indx_stain_blur]
#plt.scatter(x, y)
#plt.xlim(0,100)
#plt.xlabel("intensity")
#plt.ylabel("cell count")
#plt.text(100+2, np.max(y)/2, 'stain=1', horizontalalignment='center', verticalalignment='center',rotation=-90)
#plt.text(100/2, np.max(y)+8, 'blur=1', horizontalalignment='center', verticalalignment='center')
#
#
#
#plt.subplot(121)
#indx_stain_blur = np.where((Stain_train==1)*(Blur_train==2)==True)[0]
#x=Intensity_train[indx_stain_blur]; y = CellCount_train[indx_stain_blur]
#plt.scatter(x, y)
#plt.xlim(0,100)
#plt.xlabel("intensity")
#plt.ylabel("cell count")
#plt.text(100+2, np.max(y)/2, 'stain=1', horizontalalignment='center', verticalalignment='center',rotation=-90)
#plt.text(100/2, np.max(y)+8, 'blur=1', horizontalalignment='center', verticalalignment='center')
