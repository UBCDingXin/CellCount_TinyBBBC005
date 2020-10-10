import copy
import os
import h5py
import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
import numpy as np
import random
import timeit
from sklearn import linear_model

### import my stuffs ###
from opts import prepare_options
from utils import IMGs_dataset

#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
args = prepare_options()
print(args)

#-------------------------------
# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)

#-------------------------------
# output folders
save_models_folder = args.root + '/output/saved_models'
os.makedirs(save_models_folder, exist_ok=True)

save_images_folder = args.root + '/output/saved_images'
os.makedirs(save_images_folder, exist_ok=True)

#--------------------------------
# deleted classes for different experiments at different rounds
if (args.experiment_name).split('_')[0] == 'exp2':
    candidate_deleted_classes = [[14, 35, 57, 66, 83],
                                 [10, 31, 70, 83, 91],
                                 [18, 27, 44, 53, 91]]
elif (args.experiment_name).split('_')[0] == 'exp3':
    candidate_deleted_classes = [[61, 66, 70, 74, 78],
                                 [70, 74, 78, 83, 87],
                                 [83, 87, 91, 96, 100]]


#######################################################################################
'''                                  Load Data                                      '''
#######################################################################################
### Tiny-BBBC005 dataset ###
hf = h5py.File(args.path_tinybbbc005, 'r')
train_images = hf['IMGs_train'][:]
train_counts = hf['CellCount_train'][:]
train_blurs = hf['Blur_train'][:]
train_stains = hf['Stain_train'][:]
test_images = hf['IMGs_test'][:]
test_counts = hf['CellCount_test'][:]
test_blurs = hf['Blur_test'][:]
test_stains = hf['Stain_test'][:]
max_count = np.max(train_counts)
hf.close()


### Delete some cell counts in the training set ###
if (args.experiment_name).split('_')[0] in ['exp2', 'exp3']:
    round = int((args.experiment_name).split('_')[1].split('rd')[-1]) # round 1, 2, or 3
    deleted_counts = candidate_deleted_classes[round-1]
    indx_all = set(list(np.arange(len(train_images))))
    indx_deleted = []
    for i in range(len(deleted_counts)):
        indx_i = np.where(train_counts==deleted_counts[i])[0]
        indx_deleted.extend(list(indx_i))
    print("\n Delete {} training samples for counts: ".format(len(indx_deleted)), deleted_counts)
    indx_deleted = set(indx_deleted)
    indx_left = indx_all.difference(indx_deleted)
    indx_left = np.array(list(indx_left))
    print("\n {} training samples are left.".format(len(indx_left)))
    train_images = train_images[indx_left]
    train_counts = train_counts[indx_left]
    train_blurs = train_blurs[indx_left]
    train_stains = train_stains[indx_left]
##end if



#######################################################################################
'''                            Fit and test LQReg                                   '''
#######################################################################################
train_intensity = (np.mean(train_images, axis=(1,2,3))).reshape(-1,1)
test_intensity = (np.mean(test_images, axis=(1,2,3))).reshape(-1,1)

''' Note that upper bound is based on intensity_min; lower bound is based on intensity_max '''
qreg_models = []; lreg_models = []

test_counts_pred = []
test_counts_gt = [] #the order will be different from test_counts

for tmp_stain in [1,2]:
    for tmp_blur in [1,23,48]:
        train_indx_stain_blur = np.where(((train_blurs==tmp_blur) * (train_stains==tmp_stain)) == True)[0]
        test_indx_stain_blur = np.where(((test_blurs==tmp_blur) * (test_stains==tmp_stain)) == True)[0]

        train_intensity_stain_blur = train_intensity[train_indx_stain_blur]
        test_intensity_stain_blur = test_intensity[test_indx_stain_blur]
        if tmp_stain == 1:
            regr = linear_model.LinearRegression()
            regr.fit(train_intensity_stain_blur, train_counts[train_indx_stain_blur])
            test_pred_stain_blur = (regr.predict(test_intensity_stain_blur)).reshape(-1) #prediction from Linear/Quadratic regression
        else:
            regr = linear_model.LinearRegression()
            regr.fit(np.concatenate((train_intensity_stain_blur, train_intensity_stain_blur**2),axis=1), train_counts[train_indx_stain_blur])
            test_pred_stain_blur = (regr.predict(np.concatenate((test_intensity_stain_blur, test_intensity_stain_blur**2),axis=1))).reshape(-1)
        #end if
        test_counts_pred.extend(test_pred_stain_blur.reshape(-1).tolist())
        test_counts_gt.extend(test_counts[test_indx_stain_blur].reshape(-1).tolist())
    #end for tmp_blur
#end for tmp_stain
test_counts_pred = np.array(test_counts_pred)
test_counts_gt = np.array(test_counts_gt)

''' Compute the final prediction errors '''
test_rmse = np.sqrt(np.mean((test_counts_pred.reshape(-1).astype(np.float) - test_counts_gt.reshape(-1).astype(np.float)) ** 2))
test_mae = np.mean(np.absolute(test_counts_pred.reshape(-1).astype(np.float) - test_counts_gt.reshape(-1).astype(np.float)))

print("\n====> LQReg test results: test rmse %.4f, test mae %.4f" % (test_rmse, test_mae))
