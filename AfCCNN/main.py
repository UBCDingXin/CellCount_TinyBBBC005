'''

Main file to train our proposed method

'''

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
from train_class_cnn import train_class_cnn, test_class_cnn
from train_regre_cnn import train_regre_cnn, test_regre_cnn
from AugmentData import AugmentData
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.vgg import VGG


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
# max_count = np.max(train_counts)
max_count = 1.0 #no normalization
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


### Delete extra training samples if needed
if args.num_train<len(train_images):
    indx_all = np.arange(len(train_images))
    indx_left = np.random.choice(indx_all, size=args.num_train, replace=False)
    train_images = train_images[indx_left]
    train_counts = train_counts[indx_left]
    train_blurs = train_blurs[indx_left]
    train_stains = train_stains[indx_left]



#######################################################################################
'''                           Train LQReg for Ensemble                              '''
#######################################################################################
''' If ensemble class CNN and LQReg, train THREE (lower&upper bounds and all) LReg/QReg for each of the SIX combinations '''
''' Do this before data augmentation !!! '''
if args.ensemble and args.predtype == 'class':
    train_intensity_beforeDA = (np.mean(train_images, axis=(1,2,3))).reshape(-1,1) ##train intensity before DA
    train_counts_beforeDA = copy.deepcopy(train_counts)
    train_unique_counts_beforeDA = np.sort(np.array(list(set(train_counts))))

    ''' Note that upper bound is based on intensity_min; lower bound is based on intensity_max '''
    qreg_models = []; lreg_models = []
    qreg_ub_models = []; lreg_ub_models = []
    qreg_lb_models = []; lreg_lb_models = []
    for tmp_stain in [1,2]:
        for tmp_blur in [1,23,48]:
            intensity_mins = []
            intensity_maxs = []
            for i in range(len(train_unique_counts_beforeDA)):
                current_cellcount = train_unique_counts_beforeDA[i]
                indx_stain_blur_count = np.where(((train_blurs==tmp_blur) * (train_stains==tmp_stain) * (train_counts==current_cellcount))==True)[0]
                intensity_mins.append(np.min(train_intensity_beforeDA[indx_stain_blur_count]))
                intensity_maxs.append(np.max(train_intensity_beforeDA[indx_stain_blur_count]))
            y = train_unique_counts_beforeDA.reshape(-1,1)
            x_min = np.array(intensity_mins).reshape(-1,1)
            x_max = np.array(intensity_maxs).reshape(-1,1)
            x = train_intensity_beforeDA[(train_blurs==tmp_blur) * (train_stains==tmp_stain)]
            if tmp_stain == 1:
                regr_min = linear_model.LinearRegression()
                regr_min.fit(x_min, y)
                regr_max = linear_model.LinearRegression()
                regr_max.fit(x_max, y)
                regr = linear_model.LinearRegression()
                regr.fit(x, train_counts[(train_blurs==tmp_blur) * (train_stains==tmp_stain)])
                lreg_ub_models.append(regr_min)
                lreg_lb_models.append(regr_max)
                lreg_models.append(regr)
            else:
                regr_min = linear_model.LinearRegression()
                regr_min.fit(np.concatenate((x_min,x_min**2),axis=1), y)
                regr_max = linear_model.LinearRegression()
                regr_max.fit(np.concatenate((x_max,x_max**2),axis=1), y)
                regr = linear_model.LinearRegression()
                regr.fit(np.concatenate((x,x**2),axis=1),train_counts[(train_blurs==tmp_blur) * (train_stains==tmp_stain)])
                qreg_ub_models.append(regr_min)
                qreg_lb_models.append(regr_max)
                qreg_models.append(regr)
            #end if
        #end for tmp_blur
    #end for tmp_stain
    print("\n Now we have {} LReg, {} LReg_lb, and {} LReg_ub models to adjust the class CNN predictions.".format(len(lreg_models), len(lreg_lb_models), len(lreg_ub_models)))
    print("\n Now we have {} QReg, {} QReg_lb, and {} QReg_ub models to adjust the class CNN predictions.".format(len(qreg_models), len(qreg_lb_models), len(qreg_ub_models)))
##end if args.ensemble



#######################################################################################
'''                              Data Augmentation                                  '''
#######################################################################################
if (args.experiment_name).split('_')[0] in ['exp2', 'exp3'] and args.dataAugment and args.predtype == 'class':
    print('\n Start data augmentation...')
    formulae_dir = args.root + '/formulae/{}/{}/'.format((args.experiment_name).split('_')[0], (args.experiment_name).split('_')[-1])
    train_images, train_counts, train_blurs, train_stains = AugmentData(train_images, train_counts, train_blurs, train_stains, deleted_counts, formulae_dir, nfake=args.nfake, flipping = args.da_flip, one_formula_per_class=args.one_formula_per_class, show_sample_img=False, dump_fake=False, fakeImg_dir=None, do_filter = args.da_filter, verbose=False)
    print('\n After DA, there are {} samples in the training set.'.format(len(train_images)))
#end if
#training and test intensity
train_intensity = (np.mean(train_images, axis=(1,2,3))).reshape(-1,1)
test_intensity = (np.mean(test_images, axis=(1,2,3))).reshape(-1,1)



#######################################################################################
'''                              Create Dataloader                                  '''
#######################################################################################

if args.predtype == 'class': ## classification CNN
    ## convert cell counts to class labels
    train_unique_counts = np.sort(np.array(list(set(train_counts))))
    num_classes = len(train_unique_counts)
    count_to_classlabel = {} #count to class label
    classlabel_to_count = {} #class label to count
    train_classlabels = copy.deepcopy(train_counts)
    for i in range(num_classes):
        count_to_classlabel[train_unique_counts[i]] = i
        classlabel_to_count[i] = train_unique_counts[i]
        train_classlabels[train_classlabels==train_unique_counts[i]] = i #convert count to class label
    #end for i

    ## dataset and dataloader
    train_dataset_class = IMGs_dataset(images=train_images, dot_annots=None, masks=None, counts=train_classlabels, normalize=True, transform=args.transform)
    train_dataloader_class = torch.utils.data.DataLoader(train_dataset_class, batch_size = args.batch_size_train, shuffle=True, num_workers=args.num_workers)

else: ## regression CNN
    num_classes = 1
    ### normalize cell counts ###
    train_counts_normalized = train_counts/max_count
    ## dataset and dataloader
    train_dataset_regre = IMGs_dataset(images=train_images, dot_annots=None, masks=None, counts=train_counts_normalized, normalize=True, transform=args.transform)
    train_dataloader_regre = torch.utils.data.DataLoader(train_dataset_regre, batch_size = args.batch_size_train, shuffle=True, num_workers=args.num_workers)
## end if

## test dataset and dataloader
test_dataset = IMGs_dataset(images=test_images, dot_annots=None, masks=None, counts=test_counts, normalize=True, transform=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size_test, shuffle=False, num_workers=args.num_workers)



#######################################################################################
'''                                  CNN Training                                   '''
#######################################################################################
print("\n -----------------------------------------------------------------------------------------")
print("\n Start training {}({}) in {}  >>>".format(args.cnn_name, args.predtype, args.experiment_name))

def cnn_initialization(cnn_name, num_classes):
    if cnn_name == "ResNet18":
        net = ResNet18(num_classes=num_classes)
    elif cnn_name == "ResNet34":
        net = ResNet34(num_classes=num_classes)
    elif cnn_name == "ResNet50":
        net = ResNet50(num_classes=num_classes)
    elif cnn_name == "ResNet101":
        net = ResNet101(num_classes=num_classes)
    elif cnn_name[0:3] == "VGG":
        net = VGG(cnn_name, num_classes=num_classes)
    net = nn.DataParallel(net)
    net = net.cuda()
    return net

cnn_ckpt_fullpath = save_models_folder + '/ckpt_{}_{}_{}_epochs_{}_transform_{}_dataAugment_{}_seed_{}.pth'.format(args.experiment_name, args.predtype, args.cnn_name, args.epochs, args.transform, args.dataAugment, args.seed)
print('\n' + cnn_ckpt_fullpath)

path_to_ckpt_in_train = save_models_folder + '/ckpts_in_train_{}_{}_{}_transform_{}_seed_{}'.format(args.experiment_name, args.predtype, args.cnn_name, args.transform, args.seed)
os.makedirs(path_to_ckpt_in_train, exist_ok=True)

### train the cnn ###
if not os.path.isfile(cnn_ckpt_fullpath):
    start = timeit.default_timer()
    print("\n Begin Training {}:".format(args.cnn_name))

    cnn_net = cnn_initialization(args.cnn_name, num_classes)

    if args.predtype == 'class':
        cnn_net = train_class_cnn(train_dataloader_class, test_dataloader, classlabel_to_count, cnn_net, path_to_ckpt=path_to_ckpt_in_train)
    else:
        cnn_net = train_regre_cnn(train_dataloader_regre, test_dataloader, max_count, cnn_net, path_to_ckpt=path_to_ckpt_in_train)

    # store model
    torch.save({
        'net_state_dict': cnn_net.state_dict(),
    }, cnn_ckpt_fullpath)

    stop = timeit.default_timer()
    print("{} training finished! Time elapses: {}s".format(args.cnn_name, stop - start))
else:
    print("\n Load pre-trained {}:".format(args.cnn_name))
    checkpoint = torch.load(cnn_ckpt_fullpath)
    cnn_net = cnn_initialization(args.cnn_name, num_classes)
    cnn_net.load_state_dict(checkpoint['net_state_dict'])
# end if

print('\n Test error without ensembling...')
### test the cnn ###
if args.predtype == 'class':
    cnn_output = test_class_cnn(testloader = test_dataloader, classlabel_to_count=classlabel_to_count, net=cnn_net, verbose=True)
    test_counts_pred_main_cnn = cnn_output['Pred'] #predictions on the test set; for ensemble later
else:
    cnn_output = test_regre_cnn(testloader = test_dataloader, max_count=max_count, net=cnn_net, verbose=True)



#######################################################################################
'''                           Ensemble Prediction                                   '''
#######################################################################################
if args.predtype == 'class' and args.ensemble:
    print("\n -----------------------------------------------------------------------------------------")
    print("\n Ensemble {}(class) and {}(regre) in {}  >>>".format(args.cnn_name, args.ensemble_regre_model, args.experiment_name))

    ''' First, construct lower/upper bounds for test predictions; and also let LQReg to make predictions if needed '''
    index_test_reorder = [] #index of test samples in a shuffled order; will use it shuffle test samples and cnn predicitons later on

    test_counts_pred_ensemble_regre_model = []
    test_counts_pred_upper_bounds = []
    test_counts_pred_lower_bounds = []

    for tmp_stain in [1,2]:
        tmp_flag = 0
        for tmp_blur in [1,23,48]:
            # training
            stain_blur_indx = np.where(np.logical_and(test_stains==tmp_stain, test_blurs==tmp_blur)==True)[0]
            test_intensity_stain_blur = test_intensity[stain_blur_indx]
            test_counts_stain_blur = test_counts[stain_blur_indx]

            if tmp_stain == 1:
                regr = lreg_models[tmp_flag]
                if args.ensemble_regre_model == 'LQReg':
                    test_pred_stain_blur = (regr.predict(test_intensity_stain_blur)).reshape(-1) #prediction from Linear/Quadratic regression
                regr = lreg_ub_models[tmp_flag]
                test_counts_ub_stain_blur = (regr.predict(test_intensity_stain_blur)).reshape(-1) #upper bound
                regr = lreg_lb_models[tmp_flag]
                test_counts_lb_stain_blur = (regr.predict(test_intensity_stain_blur)).reshape(-1) #lower bound
            else:
                regr = qreg_models[tmp_flag]
                if args.ensemble_regre_model == 'LQReg':
                    test_pred_stain_blur = (regr.predict(np.concatenate((test_intensity_stain_blur, test_intensity_stain_blur**2),axis=1))).reshape(-1)
                regr = qreg_ub_models[tmp_flag]
                test_counts_ub_stain_blur = (regr.predict(np.concatenate((test_intensity_stain_blur, test_intensity_stain_blur**2),axis=1))).reshape(-1)
                regr = qreg_lb_models[tmp_flag]
                test_counts_lb_stain_blur = (regr.predict(np.concatenate((test_intensity_stain_blur, test_intensity_stain_blur**2),axis=1))).reshape(-1)
            #end if tmp_stain

            if args.ensemble_regre_model == 'LQReg':
                test_counts_pred_ensemble_regre_model.extend(test_pred_stain_blur.tolist())
            test_counts_pred_upper_bounds.extend(test_counts_ub_stain_blur.tolist())
            test_counts_pred_lower_bounds.extend(test_counts_lb_stain_blur.tolist())
            index_test_reorder.extend(stain_blur_indx.tolist())

            tmp_flag+=1
        #end for tmp_blur
    #end for tmp_stain
    if args.ensemble_regre_model == 'LQReg':
        test_counts_pred_ensemble_regre_model = np.array(test_counts_pred_ensemble_regre_model)
    test_counts_pred_upper_bounds = np.array(test_counts_pred_upper_bounds)
    test_counts_pred_lower_bounds = np.array(test_counts_pred_lower_bounds)
    index_test_reorder = np.array(index_test_reorder)

    ''' Second, make predictions by regression CNN if needed  '''
    ''' assume the regression cnn for ensembling is already trained '''
    if args.ensemble_regre_model != 'LQReg':
        ### load pre-trained regression cnn
        ensemble_regre_cnn_path = os.path.join(save_models_folder, args.ensemble_regre_cnn_path)
        assert os.path.exists(ensemble_regre_cnn_path)
        checkpoint = torch.load(ensemble_regre_cnn_path)
        ensemble_cnn_net = cnn_initialization(args.ensemble_regre_model, 1)
        ensemble_cnn_net.load_state_dict(checkpoint['net_state_dict'])
        ### make predictions
        regre_cnn_output = test_regre_cnn(testloader = test_dataloader, max_count=max_count, net=ensemble_cnn_net, verbose=True)
        test_counts_pred_ensemble_regre_model = regre_cnn_output['Pred'].reshape(-1) #predictions on the test set
        test_counts_pred_ensemble_regre_model = test_counts_pred_ensemble_regre_model[index_test_reorder] #re-order

    # test_counts_pred_ensemble_regre_model[test_counts_pred_ensemble_regre_model<1] = 1
    # test_counts_pred_ensemble_regre_model[test_counts_pred_ensemble_regre_model>max_count] = max_count


    ''' Finally, adjust classification CNN predictions by the regression model '''
    ## re-order test predictions
    test_counts_pred_main_cnn = test_counts_pred_main_cnn[index_test_reorder]

    ## adjust
    indx_out_of_bounds = np.logical_or(test_counts_pred_main_cnn<test_counts_pred_lower_bounds, test_counts_pred_main_cnn>test_counts_pred_upper_bounds)
    test_counts_pred_main_cnn[indx_out_of_bounds] = test_counts_pred_ensemble_regre_model[indx_out_of_bounds]

    ''' Compute the final prediction errors '''
    ## re-order test samples
    test_counts_gt = test_counts[index_test_reorder]

    test_acc = np.mean(test_counts_pred_main_cnn.reshape(-1) == test_counts_gt.reshape(-1))
    test_rmse = np.sqrt(np.mean((test_counts_pred_main_cnn.reshape(-1).astype(np.float) - test_counts_gt.reshape(-1).astype(np.float)) ** 2))
    test_mae = np.mean(np.absolute(test_counts_pred_main_cnn.reshape(-1).astype(np.float) - test_counts_gt.reshape(-1).astype(np.float)))

    print("\n====> Ensemble test results: Test RMSE %.4f, Test MAE %.4f, Test Acc %.4f" % (test_rmse, test_mae, test_acc))

''' END '''
