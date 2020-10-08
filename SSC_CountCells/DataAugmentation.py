#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions for data augmentation

input the raw dataset (a numpy array) and output a dataset (another numpy array) which contains the original one and fake images

LQReg Bound filtering


"""


import h5py
import numpy as np
import random
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import gc
import torchvision
from tqdm import tqdm

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


#import os
#wd = '/home/xin/Working directory/Counting_Cells/SSC_CountCells'
##wd = 'C:/Users/xin/Desktop/Github/SSC_CountCells'
#os.chdir(wd)
## only consider resized images
#h5py_file = './data/CellCount_resized_dataset.h5'
#hf = h5py.File(h5py_file, 'r')
#IMGs_real = hf['IMGs_resized_train'].value
#Blur_real = hf['Blur_train'].value
#Stain_real = hf['Stain_train'].value
#CellCount_real = hf['CellCount_train'].value
#hf.close()
#
#
#AugClass=[ 2,  3,  4,  6,  7,  8,  9, 11, 12, 13, 15, 16, 17, 19, 20, 21, 22,
#       24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 39, 41, 42, 43, 45,
#       46, 47, 49, 50, 51, 52, 54, 55, 56, 58, 59, 60, 62, 63, 64, 65, 67,
#       68, 69, 71, 72, 73, 75, 76, 77, 79, 80, 81, 82, 84, 85, 86, 88, 89,
#       90, 92, 93, 94, 95, 97, 98, 99]
#formula_dir = wd + '/data/Exp2_Formulae/'
#fakeImg_dir = wd + '/Output/saved_images/Exp2_fake_images/'
#if not os.path.exists(fakeImg_dir):
#    os.makedirs(fakeImg_dir)
#
#nfake=25; flipping=False; one_formula_per_class=False; do_filter=True; nfake_extra=200; filter_method='QReg'



TransHFlip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
TransVFlip = torchvision.transforms.RandomVerticalFlip(p=0.5)

threshold_for_skipping_filtering =1
multiple_left_bound = 1
multiple_right_bound = 1.5

def AugmentData(IMGs_real, CellCount_real, Blur_real, Stain_real, AugClass,
                formula_dir, nfake=25, flipping = True, one_formula_per_class=False,
                show_sample_img=False, dump_fake=False, fakeImg_dir=None, do_filter = True, verbose=True):
    #AugClass: a list of classes we want to create
    #one_formula_per_class: one formula for each class?
    
    Intensity_real = (np.mean(IMGs_real, axis=(1,2,3))).reshape(-1,1)

    ###########################################################################
    # load formula for creating fake classes
    formula_fake_list = []
    for i in range(len(AugClass)): #load formulae for each class
        formula_tmp = np.loadtxt(formula_dir+'C'+str(AugClass[i])+'.csv', dtype=np.int, delimiter=',')
        basis_cellcounts = formula_tmp[0,1:]
        cellcount_filename = AugClass[i] #cell count in filename
        cellcount_in_csv = formula_tmp[1,0] #cell count in csv
        assert cellcount_filename == cellcount_in_csv #they should be consistent
        formula_tmp = formula_tmp[1:,1:]
        for j in range(len(formula_tmp)): #check the correctness of each formula
            if np.sum(formula_tmp[j]*basis_cellcounts)!=cellcount_in_csv:
                print((AugClass[i], formula_tmp[j]))
            assert np.sum(formula_tmp[j]*basis_cellcounts)==cellcount_in_csv
        if one_formula_per_class:
            formula_tmp = formula_tmp[0] #one formula for each class
            formula_tmp = formula_tmp.reshape(1,len(basis_cellcounts))
        #the number of rows in formula_tmp represents the number of formulae for each cell count
        #store formula in the list
        formula_fake_list.append(formula_tmp)

    #unique cell count in the input dataset
    unique_cell_count = list(np.sort(np.array(list(set(CellCount_real)))))
#    unique_cell_count = list(set(CellCount_real))

    ###########################################################################
    # create fake classes by using existing classes
    #--------------------------------------------------------------------------
    ### 1. Blur 1 and Stain 1
    idx_per_class_1 = [] #indices of this combination if cell count is fixed
    for i in range(len(unique_cell_count)):
        idx_per_class_1.append(np.where((Blur_real==1) * (Stain_real==1) * (CellCount_real==unique_cell_count[i]) == True)[0])
        
        
    # filtering by a fitted linear regression;
    # rank intensities for each cell count by square error, choose first nfake images
    intensity_real_mins = []
    intensity_real_maxs = []
    for i in range(len(unique_cell_count)):
        current_cellcount = unique_cell_count[i]
        indx_stain_blur_count = np.where(((Blur_real==1) * (Stain_real==1) * (CellCount_real==current_cellcount))==True)[0]
        intensity_real_mins.append(np.min(Intensity_real[indx_stain_blur_count]))
        intensity_real_maxs.append(np.max(Intensity_real[indx_stain_blur_count]))
    y_min = np.array(intensity_real_mins).reshape(-1,1)
    y_max = np.array(intensity_real_maxs).reshape(-1,1)
    x = np.array(list(unique_cell_count)).reshape(-1,1)
    regr_min = linear_model.LinearRegression()
    regr_min.fit(x, y_min) 
    regr_max = linear_model.LinearRegression()
    regr_max.fit(x, y_max) 
        
    IMGs_fake_1 = np.zeros((nfake*len(AugClass), 1, 300, 300), dtype=np.uint8)
    CellCount_fake_1 = np.zeros((nfake*len(AugClass),), dtype=np.uint8)
    Blur_fake_1 = np.ones((nfake*len(AugClass),), dtype = np.uint8)
    Stain_fake_1 = np.ones((nfake*len(AugClass),), dtype = np.uint8)
    
    nfake_got_total = 0
    for i in np.arange(len(AugClass)):
        formula_tmp = formula_fake_list[i] #formula for current class
        num_formula = formula_tmp.shape[0] #number of formulae for current class

        current_cell_count = AugClass[i] #current new class
        
        if do_filter:
            intensity_min = (regr_min.predict(np.array([current_cell_count]).reshape(-1,1)))[0][0]*multiple_left_bound
            intensity_max = (regr_max.predict(np.array([current_cell_count]).reshape(-1,1)))[0][0]*multiple_right_bound
        else:
            intensity_min = -100.0
            intensity_max = 99999
        
        j=0; nfake_got_local = 0
        while nfake_got_local < nfake:
            #randomly choose one formula to generate current image
            idx_formula = np.arange(num_formula)
            np.random.shuffle(idx_formula)
            idx_formula = idx_formula[0]
            
            #column idx of required existing classes for generating current image in existing 24 classes (col index in formula table)
            idx_required_class = np.where(formula_tmp[idx_formula]!=0)[0]
            #number of images for each required class
            num_required_imgs = formula_tmp[idx_formula,idx_required_class]
            #to store the indicies of require images in IMGs_real
            idx_required_imgs = np.array([], dtype=np.uint8)
            for tmp in range(len(idx_required_class)):
                idx_required_imgs = np.concatenate((idx_required_imgs, np.random.choice(idx_per_class_1[idx_required_class[tmp]], num_required_imgs[tmp], replace=False)))
            required_imgs = IMGs_real[idx_required_imgs,0,:,:]

            if flipping:
                for tmp in range(len(idx_required_imgs)):
                    PIL_im = Image.fromarray(np.uint8(required_imgs[tmp]), mode = 'L')
                    PIL_im = TransHFlip(PIL_im)
                    PIL_im = TransVFlip(PIL_im)
                    required_imgs[tmp] = np.array(PIL_im)

            current_fake_img  = np.amax(required_imgs, axis=0)
            current_intensity = np.mean(current_fake_img)
            
            if ((current_intensity>=intensity_min) and (current_intensity<=intensity_max)) or (current_cell_count<=threshold_for_skipping_filtering):
                IMGs_fake_1[nfake_got_total:(nfake_got_total+1),0,:,:] = current_fake_img
                CellCount_fake_1[nfake_got_total:(nfake_got_total+1)] = current_cell_count
                nfake_got_local += 1
                nfake_got_total += 1
                j=0
            elif j>3000:
                j=0
                if current_intensity<intensity_min:
                    intensity_min *= 0.975
                if current_intensity>intensity_max:
                    intensity_max *= 1.025
            
            j+=1
            
            
            if verbose:
                print("Comb:%d; Class:%d; Steps:%d; Nfake:%d; min:%.4f; max:%.4f; current:%.4f"%(1,AugClass[i],j,nfake_got_local,intensity_min,intensity_max,current_intensity))
    

    if show_sample_img:
        idx_tmp = np.arange(len(IMGs_fake_1))
        np.random.shuffle(idx_tmp)
        idx_tmp=idx_tmp[0]
        plt.imshow(IMGs_fake_1[idx_tmp][0], cmap='gray', vmin=0, vmax=255)
        plt.show()


    #--------------------------------------------------------------------------
    ### 2. Blur 1 and Stain 2
    idx_per_class_2 = []
    for i in range(len(unique_cell_count)):
        idx_per_class_2.append(np.where((Blur_real==1) * (Stain_real==2) * (CellCount_real==unique_cell_count[i]) == True)[0])

    # filtering by a fitted linear regression;
    # rank intensities for each cell count by square error, choose first nfake images
    intensity_real_mins = []
    intensity_real_maxs = []
    for i in range(len(unique_cell_count)):
        current_cellcount = unique_cell_count[i]
        indx_stain_blur_count = np.where(((Blur_real==1) * (Stain_real==2) * (CellCount_real==current_cellcount))==True)[0]
        intensity_real_mins.append(np.min(Intensity_real[indx_stain_blur_count]))
        intensity_real_maxs.append(np.max(Intensity_real[indx_stain_blur_count]))
    x_min = np.array(intensity_real_mins).reshape(-1,1)
    x_max = np.array(intensity_real_maxs).reshape(-1,1)
    y = np.array(list(unique_cell_count)).reshape(-1,1)
    X_min = np.concatenate((x_min,x_min**2),axis=1)
    X_max = np.concatenate((x_max,x_max**2),axis=1)
    regr_min = linear_model.LinearRegression()
    regr_min.fit(X_min, y) 
    c_min=regr_min.intercept_[0]; b_min=regr_min.coef_[0][0]; a_min=regr_min.coef_[0][1]
    regr_max = linear_model.LinearRegression()
    regr_max.fit(X_max, y) 
    c_max=regr_max.intercept_[0]; b_max=regr_max.coef_[0][0]; a_max=regr_max.coef_[0][1]
        
    IMGs_fake_2 = np.zeros((nfake*len(AugClass), 1, 300, 300), dtype=np.uint8)
    CellCount_fake_2 = np.zeros((nfake*len(AugClass),), dtype=np.uint8)
    Blur_fake_2 = np.ones((nfake*len(AugClass),), dtype = np.uint8)
    Stain_fake_2 = 2*np.ones((nfake*len(AugClass),), dtype = np.uint8)
    
    nfake_got_total = 0
    for i in np.arange(len(AugClass)):
        formula_tmp = formula_fake_list[i] #formula for current class
        num_formula = formula_tmp.shape[0] #number of formulae for current class

        current_cell_count = AugClass[i] #current new class
        
        if do_filter:
            intensity_min = (-b_min+np.sqrt(b_min**2-4*a_min*(c_min-current_cell_count)))/(2*a_min)
            intensity_max = (-b_max+np.sqrt(b_max**2-4*a_max*(c_max-current_cell_count)))/(2*a_max)
        else:
            intensity_min = -100.0
            intensity_max = 99999

        j=0; nfake_got_local = 0
        while nfake_got_local < nfake:
            #randomly choose one formula to generate current image
            idx_formula = np.arange(num_formula)
            np.random.shuffle(idx_formula)
            idx_formula = idx_formula[0]
            
            #column idx of required existing classes for generating current image in existing 24 classes (col index in formula table)
            idx_required_class = np.where(formula_tmp[idx_formula]!=0)[0]
            #number of images for each required class
            num_required_imgs = formula_tmp[idx_formula,idx_required_class]
            #to store the indicies of require images in IMGs_real
            idx_required_imgs = np.array([], dtype=np.uint8)
            for tmp in range(len(idx_required_class)):
                idx_required_imgs = np.concatenate((idx_required_imgs, np.random.choice(idx_per_class_2[idx_required_class[tmp]], num_required_imgs[tmp], replace=False)))
            required_imgs = IMGs_real[idx_required_imgs,0,:,:]

            if flipping:
                for tmp in range(len(idx_required_imgs)):
                    PIL_im = Image.fromarray(np.uint8(required_imgs[tmp]), mode = 'L')
                    PIL_im = TransHFlip(PIL_im)
                    PIL_im = TransVFlip(PIL_im)
                    required_imgs[tmp] = np.array(PIL_im)

            current_fake_img  = np.amax(required_imgs, axis=0)
            current_intensity = np.mean(current_fake_img)
            
            if ((current_intensity>=intensity_min) and (current_intensity<=intensity_max)) or (current_cell_count<=threshold_for_skipping_filtering):
                IMGs_fake_2[nfake_got_total:(nfake_got_total+1),0,:,:] = current_fake_img
                CellCount_fake_2[nfake_got_total:(nfake_got_total+1)] = current_cell_count
                nfake_got_local += 1
                nfake_got_total += 1
                j=0
            elif j>3000:
                j=0
                if current_intensity<intensity_min:
                    intensity_min *= 0.975
                if current_intensity>intensity_max:
                    intensity_max *= 1.025
            j+=1
            
            if verbose:
                print("Comb:%d; Class:%d; Steps:%d; Nfake:%d; min:%.4f; max:%.4f; current:%.4f"%(2,AugClass[i],j,nfake_got_local,intensity_min,intensity_max,current_intensity))
    
    if show_sample_img:
        idx_tmp = np.arange(len(IMGs_fake_2))
        np.random.shuffle(idx_tmp)
        idx_tmp=idx_tmp[0]
        plt.imshow(IMGs_fake_2[idx_tmp][0], cmap='gray', vmin=0, vmax=255)
        plt.show()


    #--------------------------------------------------------------------------
    ### 3. Blur 23 and Stain 1
    idx_per_class_3 = []
    for i in range(len(unique_cell_count)):
        idx_per_class_3.append(np.where((Blur_real==23) * (Stain_real==1) * (CellCount_real==unique_cell_count[i]) == True)[0])

    # filtering by a fitted linear regression;
    # rank intensities for each cell count by square error, choose first nfake images
    intensity_real_mins = []
    intensity_real_maxs = []
    for i in range(len(unique_cell_count)):
        current_cellcount = unique_cell_count[i]
        indx_stain_blur_count = np.where(((Blur_real==23) * (Stain_real==1) * (CellCount_real==current_cellcount))==True)[0]
        intensity_real_mins.append(np.min(Intensity_real[indx_stain_blur_count]))
        intensity_real_maxs.append(np.max(Intensity_real[indx_stain_blur_count]))
    y_min = np.array(intensity_real_mins).reshape(-1,1)
    y_max = np.array(intensity_real_maxs).reshape(-1,1)
    x = np.array(list(unique_cell_count)).reshape(-1,1)
    regr_min = linear_model.LinearRegression()
    regr_min.fit(x, y_min) 
    regr_max = linear_model.LinearRegression()
    regr_max.fit(x, y_max) 
        
    IMGs_fake_3 = np.zeros((nfake*len(AugClass), 1, 300, 300), dtype=np.uint8)
    CellCount_fake_3 = np.zeros((nfake*len(AugClass),), dtype=np.uint8)
    Blur_fake_3 = 23*np.ones((nfake*len(AugClass),), dtype = np.uint8)
    Stain_fake_3 = np.ones((nfake*len(AugClass),), dtype = np.uint8)
    
    nfake_got_total = 0
    for i in np.arange(len(AugClass)):
        formula_tmp = formula_fake_list[i] #formula for current class
        num_formula = formula_tmp.shape[0] #number of formulae for current class

        current_cell_count = AugClass[i] #current new class
        
        if do_filter:
            intensity_min = (regr_min.predict(np.array([current_cell_count]).reshape(-1,1)))[0][0]*multiple_left_bound
            intensity_max = (regr_max.predict(np.array([current_cell_count]).reshape(-1,1)))[0][0]*multiple_right_bound
        else:
            intensity_min = -100.0
            intensity_max = 99999
        
        j=0; nfake_got_local = 0
        while nfake_got_local < nfake:
            #randomly choose one formula to generate current image
            idx_formula = np.arange(num_formula)
            np.random.shuffle(idx_formula)
            idx_formula = idx_formula[0]
            
            #column idx of required existing classes for generating current image in existing 24 classes (col index in formula table)
            idx_required_class = np.where(formula_tmp[idx_formula]!=0)[0]
            #number of images for each required class
            num_required_imgs = formula_tmp[idx_formula,idx_required_class]
            #to store the indicies of require images in IMGs_real
            idx_required_imgs = np.array([], dtype=np.uint8)
            for tmp in range(len(idx_required_class)):
                idx_required_imgs = np.concatenate((idx_required_imgs, np.random.choice(idx_per_class_3[idx_required_class[tmp]], num_required_imgs[tmp], replace=False)))
            required_imgs = IMGs_real[idx_required_imgs,0,:,:]

            if flipping:
                for tmp in range(len(idx_required_imgs)):
                    PIL_im = Image.fromarray(np.uint8(required_imgs[tmp]), mode = 'L')
                    PIL_im = TransHFlip(PIL_im)
                    PIL_im = TransVFlip(PIL_im)
                    required_imgs[tmp] = np.array(PIL_im)

            current_fake_img  = np.amax(required_imgs, axis=0)
            current_intensity = np.mean(current_fake_img)
            
            if ((current_intensity>=intensity_min) and (current_intensity<=intensity_max)) or (current_cell_count<=threshold_for_skipping_filtering):
                IMGs_fake_3[nfake_got_total:(nfake_got_total+1),0,:,:] = current_fake_img
                CellCount_fake_3[nfake_got_total:(nfake_got_total+1)] = current_cell_count
                nfake_got_local += 1
                nfake_got_total += 1
                j=0
            elif j>3000:
                j=0
                if current_intensity<intensity_min:
                    intensity_min *= 0.975
                if current_intensity>intensity_max:
                    intensity_max *= 1.025
            j+=1
            
            if verbose:
                print("Comb:%d; Class:%d; Steps:%d; Nfake:%d; min:%.4f; max:%.4f; current:%.4f"%(3,AugClass[i],j,nfake_got_local,intensity_min,intensity_max,current_intensity))
    
    if show_sample_img:
        idx_tmp = np.arange(len(IMGs_fake_3))
        np.random.shuffle(idx_tmp)
        idx_tmp=idx_tmp[0]
        plt.imshow(IMGs_fake_3[idx_tmp][0], cmap='gray', vmin=0, vmax=255)
        plt.show()


    #--------------------------------------------------------------------------
    ### 4. Blur 23 and Stain 2
    idx_per_class_4 = []
    for i in range(len(unique_cell_count)):
        idx_per_class_4.append(np.where((Blur_real==23) * (Stain_real==2) * (CellCount_real==unique_cell_count[i]) == True)[0])

    # filtering by a fitted linear regression;
    # rank intensities for each cell count by square error, choose first nfake images
    intensity_real_mins = []
    intensity_real_maxs = []
    for i in range(len(unique_cell_count)):
        current_cellcount = unique_cell_count[i]
        indx_stain_blur_count = np.where(((Blur_real==23) * (Stain_real==2) * (CellCount_real==current_cellcount))==True)[0]
        intensity_real_mins.append(np.min(Intensity_real[indx_stain_blur_count]))
        intensity_real_maxs.append(np.max(Intensity_real[indx_stain_blur_count]))
    x_min = np.array(intensity_real_mins).reshape(-1,1)
    x_max = np.array(intensity_real_maxs).reshape(-1,1)
    y = np.array(list(unique_cell_count)).reshape(-1,1)
    X_min = np.concatenate((x_min,x_min**2),axis=1)
    X_max = np.concatenate((x_max,x_max**2),axis=1)
    regr_min = linear_model.LinearRegression()
    regr_min.fit(X_min, y) 
    c_min=regr_min.intercept_[0]; b_min=regr_min.coef_[0][0]; a_min=regr_min.coef_[0][1]
    regr_max = linear_model.LinearRegression()
    regr_max.fit(X_max, y) 
    c_max=regr_max.intercept_[0]; b_max=regr_max.coef_[0][0]; a_max=regr_max.coef_[0][1]
        
    IMGs_fake_4 = np.zeros((nfake*len(AugClass), 1, 300, 300), dtype=np.uint8)
    CellCount_fake_4 = np.zeros((nfake*len(AugClass),), dtype=np.uint8)
    Blur_fake_4 = 23*np.ones((nfake*len(AugClass),), dtype = np.uint8)
    Stain_fake_4 = 2*np.ones((nfake*len(AugClass),), dtype = np.uint8)
    
    nfake_got_total = 0
    for i in np.arange(len(AugClass)):
        formula_tmp = formula_fake_list[i] #formula for current class
        num_formula = formula_tmp.shape[0] #number of formulae for current class

        current_cell_count = AugClass[i] #current new class
        
        if do_filter:
            intensity_min = (-b_min+np.sqrt(b_min**2-4*a_min*(c_min-current_cell_count)))/(2*a_min)
            intensity_max = (-b_max+np.sqrt(b_max**2-4*a_max*(c_max-current_cell_count)))/(2*a_max)
        else:
            intensity_min = -100.0
            intensity_max = 99999
        
        j=0; nfake_got_local = 0
        while nfake_got_local < nfake:
            #randomly choose one formula to generate current image
            idx_formula = np.arange(num_formula)
            np.random.shuffle(idx_formula)
            idx_formula = idx_formula[0]
            
            #column idx of required existing classes for generating current image in existing 24 classes (col index in formula table)
            idx_required_class = np.where(formula_tmp[idx_formula]!=0)[0]
            #number of images for each required class
            num_required_imgs = formula_tmp[idx_formula,idx_required_class]
            #to store the indicies of require images in IMGs_real
            idx_required_imgs = np.array([], dtype=np.uint8)
            for tmp in range(len(idx_required_class)):
                idx_required_imgs = np.concatenate((idx_required_imgs, np.random.choice(idx_per_class_4[idx_required_class[tmp]], num_required_imgs[tmp], replace=False)))
            required_imgs = IMGs_real[idx_required_imgs,0,:,:]

            if flipping:
                for tmp in range(len(idx_required_imgs)):
                    PIL_im = Image.fromarray(np.uint8(required_imgs[tmp]), mode = 'L')
                    PIL_im = TransHFlip(PIL_im)
                    PIL_im = TransVFlip(PIL_im)
                    required_imgs[tmp] = np.array(PIL_im)

            current_fake_img  = np.amax(required_imgs, axis=0)
            current_intensity = np.mean(current_fake_img)
            
            if ((current_intensity>=intensity_min) and (current_intensity<=intensity_max)) or (current_cell_count<=threshold_for_skipping_filtering):
                IMGs_fake_4[nfake_got_total:(nfake_got_total+1),0,:,:] = current_fake_img
                CellCount_fake_4[nfake_got_total:(nfake_got_total+1)] = current_cell_count
                nfake_got_local += 1
                nfake_got_total += 1
                j=0
            elif j>3000:
                j=0
                if current_intensity<intensity_min:
                    intensity_min *= 0.975
                if current_intensity>intensity_max:
                    intensity_max *= 1.025
            j+=1
            
            if verbose:
                print("Comb:%d; Class:%d; Steps:%d; Nfake:%d; min:%.4f; max:%.4f; current:%.4f"%(4,AugClass[i],j,nfake_got_local,intensity_min,intensity_max,current_intensity))

    if show_sample_img:
        idx_tmp = np.arange(len(IMGs_fake_4))
        np.random.shuffle(idx_tmp)
        idx_tmp=idx_tmp[0]
        plt.imshow(IMGs_fake_4[idx_tmp][0], cmap='gray', vmin=0, vmax=255)
        plt.show()


    #--------------------------------------------------------------------------
    ### 5. Blur 48 and Stain 1
    idx_per_class_5 = []
    for i in range(len(unique_cell_count)):
        idx_per_class_5.append(np.where((Blur_real==48) * (Stain_real==1) * (CellCount_real==unique_cell_count[i]) == True)[0])

    # filtering by a fitted linear regression;
    # rank intensities for each cell count by square error, choose first nfake images
    intensity_real_mins = []
    intensity_real_maxs = []
    for i in range(len(unique_cell_count)):
        current_cellcount = unique_cell_count[i]
        indx_stain_blur_count = np.where(((Blur_real==48) * (Stain_real==1) * (CellCount_real==current_cellcount))==True)[0]
        intensity_real_mins.append(np.min(Intensity_real[indx_stain_blur_count]))
        intensity_real_maxs.append(np.max(Intensity_real[indx_stain_blur_count]))
    y_min = np.array(intensity_real_mins).reshape(-1,1)
    y_max = np.array(intensity_real_maxs).reshape(-1,1)
    x = np.array(list(unique_cell_count)).reshape(-1,1)
    regr_min = linear_model.LinearRegression()
    regr_min.fit(x, y_min) 
    regr_max = linear_model.LinearRegression()
    regr_max.fit(x, y_max) 
        
    IMGs_fake_5 = np.zeros((nfake*len(AugClass), 1, 300, 300), dtype=np.uint8)
    CellCount_fake_5 = np.zeros((nfake*len(AugClass),), dtype=np.uint8)
    Blur_fake_5 = 48*np.ones((nfake*len(AugClass),), dtype = np.uint8)
    Stain_fake_5 = np.ones((nfake*len(AugClass),), dtype = np.uint8)
    
    nfake_got_total = 0
    for i in np.arange(len(AugClass)):
        formula_tmp = formula_fake_list[i] #formula for current class
        num_formula = formula_tmp.shape[0] #number of formulae for current class

        current_cell_count = AugClass[i] #current new class
        
        if do_filter:
            intensity_min = (regr_min.predict(np.array([current_cell_count]).reshape(-1,1)))[0][0]*multiple_left_bound
            intensity_max = (regr_max.predict(np.array([current_cell_count]).reshape(-1,1)))[0][0]*multiple_right_bound
        else:
            intensity_min = -100.0
            intensity_max = 99999
        
        j=0; nfake_got_local = 0
        while nfake_got_local < nfake:
            #randomly choose one formula to generate current image
            idx_formula = np.arange(num_formula)
            np.random.shuffle(idx_formula)
            idx_formula = idx_formula[0]
            
            #column idx of required existing classes for generating current image in existing 24 classes (col index in formula table)
            idx_required_class = np.where(formula_tmp[idx_formula]!=0)[0]
            #number of images for each required class
            num_required_imgs = formula_tmp[idx_formula,idx_required_class]
            #to store the indicies of require images in IMGs_real
            idx_required_imgs = np.array([], dtype=np.uint8)
            for tmp in range(len(idx_required_class)):
                idx_required_imgs = np.concatenate((idx_required_imgs, np.random.choice(idx_per_class_5[idx_required_class[tmp]], num_required_imgs[tmp], replace=False)))
            required_imgs = IMGs_real[idx_required_imgs,0,:,:]

            if flipping:
                for tmp in range(len(idx_required_imgs)):
                    PIL_im = Image.fromarray(np.uint8(required_imgs[tmp]), mode = 'L')
                    PIL_im = TransHFlip(PIL_im)
                    PIL_im = TransVFlip(PIL_im)
                    required_imgs[tmp] = np.array(PIL_im)

            current_fake_img  = np.amax(required_imgs, axis=0)
            current_intensity = np.mean(current_fake_img)
            
            if ((current_intensity>=intensity_min) and (current_intensity<=intensity_max)) or (current_cell_count<=threshold_for_skipping_filtering):
                IMGs_fake_5[nfake_got_total:(nfake_got_total+1),0,:,:] = current_fake_img
                CellCount_fake_5[nfake_got_total:(nfake_got_total+1)] = current_cell_count
                nfake_got_local += 1
                nfake_got_total += 1
                j=0
            elif j>3000:
                j=0
                if current_intensity<intensity_min:
                    intensity_min *= 0.975
                if current_intensity>intensity_max:
                    intensity_max *= 1.025
            j+=1
            
            if verbose:
                print("Comb:%d; Class:%d; Steps:%d; Nfake:%d; min:%.4f; max:%.4f; current:%.4f"%(5,AugClass[i],j,nfake_got_local,intensity_min,intensity_max,current_intensity))

    if show_sample_img:
        idx_tmp = np.arange(len(IMGs_fake_5))
        np.random.shuffle(idx_tmp)
        idx_tmp=idx_tmp[0]
        plt.imshow(IMGs_fake_5[idx_tmp][0], cmap='gray', vmin=0, vmax=255)
        plt.show()


    #--------------------------------------------------------------------------
    ### 6. Blur 48 and Stain 2
    idx_per_class_6 = []
    for i in range(len(unique_cell_count)):
        idx_per_class_6.append(np.where((Blur_real==48) * (Stain_real==2) * (CellCount_real==unique_cell_count[i]) == True)[0])

    # filtering by a fitted linear regression;
    # rank intensities for each cell count by square error, choose first nfake images
    intensity_real_mins = []
    intensity_real_maxs = []
    for i in range(len(unique_cell_count)):
        current_cellcount = unique_cell_count[i]
        indx_stain_blur_count = np.where(((Blur_real==48) * (Stain_real==2) * (CellCount_real==current_cellcount))==True)[0]
        intensity_real_mins.append(np.min(Intensity_real[indx_stain_blur_count]))
        intensity_real_maxs.append(np.max(Intensity_real[indx_stain_blur_count]))
    x_min = np.array(intensity_real_mins).reshape(-1,1)
    x_max = np.array(intensity_real_maxs).reshape(-1,1)
    y = np.array(list(unique_cell_count)).reshape(-1,1)
    X_min = np.concatenate((x_min,x_min**2),axis=1)
    X_max = np.concatenate((x_max,x_max**2),axis=1)
    regr_min = linear_model.LinearRegression()
    regr_min.fit(X_min, y) 
    c_min=regr_min.intercept_[0]; b_min=regr_min.coef_[0][0]; a_min=regr_min.coef_[0][1]
    regr_max = linear_model.LinearRegression()
    regr_max.fit(X_max, y) 
    c_max=regr_max.intercept_[0]; b_max=regr_max.coef_[0][0]; a_max=regr_max.coef_[0][1]
        
    IMGs_fake_6 = np.zeros((nfake*len(AugClass), 1, 300, 300), dtype=np.uint8)
    CellCount_fake_6 = np.zeros((nfake*len(AugClass),), dtype=np.uint8)
    Blur_fake_6 = 48*np.ones((nfake*len(AugClass),), dtype = np.uint8)
    Stain_fake_6 = 2*np.ones((nfake*len(AugClass),), dtype = np.uint8)
    
    nfake_got_total = 0
    for i in np.arange(len(AugClass)):
        formula_tmp = formula_fake_list[i] #formula for current class
        num_formula = formula_tmp.shape[0] #number of formulae for current class

        current_cell_count = AugClass[i] #current new class
        
        if do_filter:
            intensity_min = (-b_min+np.sqrt(b_min**2-4*a_min*(c_min-current_cell_count)))/(2*a_min)
            intensity_max = (-b_max+np.sqrt(b_max**2-4*a_max*(c_max-current_cell_count)))/(2*a_max)
        else:
            intensity_min = -100.0
            intensity_max = 99999
        
        j=0; nfake_got_local = 0
        while nfake_got_local < nfake:
            #randomly choose one formula to generate current image
            idx_formula = np.arange(num_formula)
            np.random.shuffle(idx_formula)
            idx_formula = idx_formula[0]
            
            #column idx of required existing classes for generating current image in existing 24 classes (col index in formula table)
            idx_required_class = np.where(formula_tmp[idx_formula]!=0)[0]
            #number of images for each required class
            num_required_imgs = formula_tmp[idx_formula,idx_required_class]
            #to store the indicies of require images in IMGs_real
            idx_required_imgs = np.array([], dtype=np.uint8)
            for tmp in range(len(idx_required_class)):
                idx_required_imgs = np.concatenate((idx_required_imgs, np.random.choice(idx_per_class_6[idx_required_class[tmp]], num_required_imgs[tmp], replace=False)))
            required_imgs = IMGs_real[idx_required_imgs,0,:,:]

            if flipping:
                for tmp in range(len(idx_required_imgs)):
                    PIL_im = Image.fromarray(np.uint8(required_imgs[tmp]), mode = 'L')
                    PIL_im = TransHFlip(PIL_im)
                    PIL_im = TransVFlip(PIL_im)
                    required_imgs[tmp] = np.array(PIL_im)

            current_fake_img  = np.amax(required_imgs, axis=0)
            current_intensity = np.mean(current_fake_img)
            
            if ((current_intensity>=intensity_min) and (current_intensity<=intensity_max)) or (current_cell_count<=threshold_for_skipping_filtering):
                IMGs_fake_6[nfake_got_total:(nfake_got_total+1),0,:,:] = current_fake_img
                CellCount_fake_6[nfake_got_total:(nfake_got_total+1)] = current_cell_count
                nfake_got_local += 1
                nfake_got_total += 1
                j=0
            elif j>3000:
                j=0
                if current_intensity<intensity_min:
                    intensity_min *= 0.95
                if current_intensity>intensity_max:
                    intensity_max *= 1.05
            j+=1
            
            if verbose:
                print("Comb:%d; Class:%d; Steps:%d; Nfake:%d; min:%.4f; max:%.4f; current:%.4f"%(6,AugClass[i],j,nfake_got_local,intensity_min,intensity_max,current_intensity))

    if show_sample_img:
        idx_tmp = np.arange(len(IMGs_fake_6))
        np.random.shuffle(idx_tmp)
        idx_tmp=idx_tmp[0]
        plt.imshow(IMGs_fake_6[idx_tmp][0], cmap='gray', vmin=0, vmax=255)
        plt.show()

    #############################################################
    # attach fake images to the end of training images
    IMGs_fake = np.concatenate((IMGs_fake_1, IMGs_fake_2, IMGs_fake_3, IMGs_fake_4, IMGs_fake_5, IMGs_fake_6), axis=0)
    CellCount_fake = np.concatenate((CellCount_fake_1, CellCount_fake_2, CellCount_fake_3, CellCount_fake_4, CellCount_fake_5, CellCount_fake_6))
    Blur_fake = np.concatenate((Blur_fake_1, Blur_fake_2, Blur_fake_3, Blur_fake_4, Blur_fake_5, Blur_fake_6))
    Stain_fake = np.concatenate((Stain_fake_1, Stain_fake_2, Stain_fake_3, Stain_fake_4, Stain_fake_5, Stain_fake_6))

    # dump fake images
    if dump_fake:
        for i in tqdm(range(len(IMGs_fake))):
            image_array=IMGs_fake[i][0]
            filename = fakeImg_dir + 'FakeImg_indx'+str(i)+'_CellCount'+str(CellCount_fake[i])+'_blur'+str(Blur_fake[i])+'_stain'+str(Stain_fake[i]) + '.png'
            im = PIL.Image.fromarray(image_array)
            im.save(filename)

    del IMGs_fake_1, IMGs_fake_2, IMGs_fake_3, IMGs_fake_4, IMGs_fake_5, IMGs_fake_6; gc.collect()

    IMGs_all = np.concatenate((IMGs_real, IMGs_fake), axis=0)
    CellCount_all = np.concatenate((CellCount_real, CellCount_fake))
    Blur_all = np.concatenate((Blur_real, Blur_fake))
    Stain_all = np.concatenate((Stain_real, Stain_fake))

    return IMGs_all, CellCount_all, Blur_all, Stain_all













#    # filtering by a fitted linear regression;
#    # rank intensities for each cell count by square error, choose first nfake images
#    indx_stain_blur = np.where(((Blur_real==1) * (Stain_real==2))==True)[0]
#    y = CellCount_real[indx_stain_blur].reshape(-1,1)
#    if filter_method == "LReg":
#        X = Intensity_real[indx_stain_blur]
#        regr = linear_model.LinearRegression()
#        regr.fit(X, y) #regress cell count on intensity not intensity on cell count
#
#        a0=regr.intercept_[0]; a1=regr.coef_[0][0];
#        x_test = (np.mean(IMGs_fake_2, axis=(1,2,3))).reshape(-1,1)
#        y_test = CellCount_fake_2.reshape(-1,1)
#        x_pred = np.zeros(x_test.shape)
#        for i in range(len(x_pred)):
#            x_pred[i] = (y_test[i]-a0)/a1
#    else:
#        X = np.concatenate((Intensity_real[indx_stain_blur], Intensity_real[indx_stain_blur]**2), axis=1)
#        regr = linear_model.LinearRegression()
#        regr.fit(X, y) #regress cell count on intensity not intensity on cell count
#
#        a0=regr.intercept_[0]; a1=regr.coef_[0][0]; a2=regr.coef_[0][1]
#        x_test = (np.mean(IMGs_fake_2, axis=(1,2,3))).reshape(-1,1)
#        y_test = CellCount_fake_2.reshape(-1,1)
#        x_pred = np.zeros(x_test.shape)
#        for i in range(len(x_pred)):
#            a0_tmp=a0-y_test[i];
#            x_pred[i] = (-a1+np.sqrt(a1**2-4*a0_tmp*a2))/(2*a2)
#    intensity_err = (x_pred-x_test)**2
#
#    IMGs_fake_tmp = np.zeros((1,1,300,300), dtype=np.uint8)
#    CellCount_fake_tmp  = np.zeros((1,), dtype=np.uint8)
#    unique_y_test = list(set(y_test.reshape(-1)))
#    for i in range(len(unique_y_test)):
#        indx_tmp = np.where(y_test==unique_y_test[i])[0]
#        intensity_err_tmp = intensity_err[indx_tmp] #intensities for current cell count
#        selected_indx = np.argsort(intensity_err_tmp) #rank in ascending order; selected_indx is the index of indx_tmp
#        IMGs_fake_tmp = np.concatenate((IMGs_fake_tmp, IMGs_fake_2[indx_tmp[selected_indx[0:nfake]].reshape(-1)]))
#        CellCount_fake_tmp = np.concatenate((CellCount_fake_tmp, CellCount_fake_2[indx_tmp[selected_indx[0:nfake]].reshape(-1)]))
#
#    IMGs_fake_2 = IMGs_fake_tmp[1:] #remove the first image
#    CellCount_fake_2 = CellCount_fake_tmp[1:]
#    del IMGs_fake_tmp, CellCount_fake_tmp; gc.collect()
