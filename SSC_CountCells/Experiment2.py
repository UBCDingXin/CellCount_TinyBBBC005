#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Augment all unseen classes and train a CNN with cross-entropy loss and mse loss (or with a subsequent classifier or regressor);

Conduct 3-fold CV for 3 times.

No test set. The whole augmented dataset are used for CV.

"""

import os
# wd = 'your path/SSC_CountCells'
wd = '/home/xin/Working directory/Counting_Cells/SSC_CountCells'
# wd = '/media/qiong/icecream/SSC_CountCells'
os.chdir(wd)
import argparse
import h5py
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
import math
import random
from utils import *
from models import *
import gc
import timeit
import scipy.misc
import PIL
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb  # conda install py-xgboost
import multiprocessing
from DataAugmentation import *

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd

#############################
# Settings
#############################
parser = argparse.ArgumentParser(description='Counting cell')
parser.add_argument('--model1', type=str, default='ResNet34',
                    help='First model (default: "ResNet34"); Candidates: LReg, QReg, PReg, VGG11, VGG13, VGG16, ResNet18, ResNet34')
parser.add_argument('--model2', type=str, default = 'CNN',
                    help='Second model (default: CNN); Candidates: None, CNN, RF, GBT')
parser.add_argument('--PredType1', type=str, default='Cl',
                    help='Prediction type for the first model (default: "Cl"); Candidates: Cl and Reg')
parser.add_argument('--PredType2', type=str, default='Cl',
                    help='Prediction type for the second model (default: "Cl"); Candidates: Cl and Reg')
parser.add_argument('--RF_NTree', type=int, default=100, metavar='N',
                    help='Number of trees in RF')
parser.add_argument('--RF_MaxDepth', type=int, default=20, metavar='N',
                    help='Max depth of a single tree in RF') #0 means fully grown
parser.add_argument('--RF_MaxFeatures', type=str, default='sqrt',
                    help='Max features for RF (default: "sqrt"); Candidates: None and sqrt' )
parser.add_argument('--GBT_loss', type=str, default='MSE',
                    help='Loss function for GBT (default:MSE); Candidates: MSE and Poisson')
parser.add_argument('--GBT_MaxDepth', type=int, default=20,
                    help='Maximum depth of a single tree')
parser.add_argument('--GBT_eta', type=float, default=0.1,
                    help='Step size shrinkage')
parser.add_argument('--GBT_Round', type=int, default=100,
                    help='Rounds of boosting')

parser.add_argument('--K', type=int, default=3, metavar='N',
                    help='K-fold CV (default: 3)')
parser.add_argument('--NROUND', type=int, default=3, metavar='N',
                    help='Rounds of CVs (default: 3)')
parser.add_argument('--useNumericalFeatures', action='store_true', default=False,
                    help='Whether use numerical features (i.e., blur level and stain type)?')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train model1 (default: 20)')
parser.add_argument('--batch_size_train', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--batch_size_test', type=int, default=50, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--batch_size_valid', type=int, default=50, metavar='N',
                    help='input batch size for validation (default: 100)')
parser.add_argument('--batch_size_extract', type=int, default=50, metavar='N',
                    help='input batch size for extracting features (default: 8)')
parser.add_argument('--base_lr', type=float, default=1e-3,
                    help='learning rate, default=1e-3')
parser.add_argument('--weight_dacay', type=float, default=1e-4,
                    help='Weigth decay, default=1e-4')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=2019, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model1_flip', action='store_true', default=False,
                    help='Vertically or horizatally flip images for CNN training')
parser.add_argument('--model2_flip', action='store_true', default=False,
                    help='Vertically or horizatally flip images for RF/XGBoost training')

parser.add_argument('--dataAugment', action='store_true', default=False,
                    help='Do data augmentation?')
parser.add_argument('--nfake', type=int, default=25)
parser.add_argument('--DA_flip', action='store_true', default=False,
                    help='Do flipping in data augmentation?')
parser.add_argument('--one_formula_per_class', action='store_true', default=False,
                    help='One formula for each cell count?')

args = parser.parse_args()
# cuda
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
ngpu = torch.cuda.device_count()  # number of gpus
args.base_lr = args.base_lr * ngpu

# data augmentation
nfake = args.nfake
formula_dir = wd + '/data/Exp2_Formulae/'
AugClass=[ 2,  3,  4,  6,  7,  8,  9, 11, 12, 13, 15, 16, 17, 19, 20, 21, 22,
          24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 39, 41, 42, 43, 45,
          46, 47, 49, 50, 51, 52, 54, 55, 56, 58, 59, 60, 62, 63, 64, 65, 67,
          68, 69, 71, 72, 73, 75, 76, 77, 79, 80, 81, 82, 84, 85, 86, 88, 89,
          90, 92, 93, 94, 95, 97, 98, 99]


# random seed
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True


# directories for checkpoint, images and log files
save_models_folder = wd + '/Output/saved_models/'
if not os.path.exists(save_models_folder):
    os.makedirs(save_models_folder)

save_images_folder = wd + '/Output/saved_images/'
if not os.path.exists(save_images_folder):
    os.makedirs(save_images_folder)

ValidCurves_filename = save_images_folder + 'AvgValidLossOverEpochs_Exp2_CV_' + args.model1 + '(' + args.PredType1 + ')_' + args.model2 + '(' + args.PredType2  + ')_epoch' + str(args.epochs) + '_seed' + str(args.seed) + '_Nround' + str(args.NROUND) + '_K' + str(args.K) + '.pdf'

save_logs_folder = wd + '/Output/saved_logs/'
if not os.path.exists(save_logs_folder):
    os.makedirs(save_logs_folder)

#fakeImg_dir = save_images_folder + 'Exp2_fake_images/'
#if not os.path.exists(fakeImg_dir):
#    os.makedirs(fakeImg_dir)

#############################
# Load data from h5 file
#############################
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

#############################
# Functions for CNN training
#############################
def save_model(epoch, save_filename, keep_optimizer=True):
    # Save checkpoint.
    print('Saving..')
    if keep_optimizer:
        state = {
            'net_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'rng_state': torch.get_rng_state()
        }
    else:
        state = {
            'net_state_dict': model.state_dict(),
            'epoch': epoch,
            'rng_state': torch.get_rng_state()
        }
    torch.save(state, save_filename)

def train(epoch, train_loader, verbose=True):
    model.train()
    train_loss = 0
    adjust_learning_rate(optimizer, epoch, args.base_lr)
    for batch_idx, (images, labels, _, _) in enumerate(train_loader):
        images = images.type(torch.float).to(device)
        if args.PredType1 == 'Cl':
            labels = labels.type(torch.long).to(device)
        else:
            labels = labels.view(-1,1)
            labels = labels.type(torch.float).to(device)
        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0 and verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch + 1, batch_idx * len(images), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(images)))
    # end for batch_idx
    train_loss = train_loss / len(train_loader.dataset)
    if verbose:
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch + 1, train_loss))
    del images, labels, outputs;
    gc.collect()
    torch.cuda.empty_cache()
    return train_loss


def test(dataloader, mode='Test', OutputPred=False, verbose=True):
    # mode: 'Test' or 'Valid'
    # OutputPred: output indecices and predicted cell count of misclassified samples, predicted cell count for all test images and predicted probabilties
    model.eval()
    N_samp = dataloader.dataset.n_images
    batch_size = dataloader.batch_size
    CalcProb = nn.Softmax(dim=1)  # softmax function which is used for  computing predicted probabilties
    test_loss = 0
    test_acc = 0  # test accuracy
    test_rmse = 0
    indx_misclassified = []
    CellCount_test_predicted = []
    CellCount_test_truth = []
    Prob_test_predicted = np.zeros((N_samp+batch_size, num_classes))
    with torch.no_grad():
        i=0; tmp=0
        data_iter = iter(dataloader)
        while i < len(data_iter):
            (images, labels, _, _) = data_iter.next()
            images = images.type(torch.float).to(device)
            if args.PredType1 == 'Cl':
                labels = labels.type(torch.long).to(device)
            else:
                labels = labels.view(-1,1)
                labels = labels.type(torch.float).to(device)
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            if args.PredType1 == 'Cl': #classification
                # predictions
                _, predicted = torch.max(outputs.data, 1)
                ## compute test accuracy
                test_acc += (predicted == labels).type(torch.float).sum().item()
                ## compute predicted cell count
                cellcount_predicted = torch.from_numpy(unique_cell_count[predicted.cpu().numpy()]).type(torch.float)
                ## compute test rmse
                cellcount_truth = torch.from_numpy(unique_cell_count[labels.cpu().numpy()]).type(torch.float)
                test_rmse += MSE_loss(cellcount_predicted, cellcount_truth).item() * len(images)
            else: #regression
                ## compute predicted cell count
                ### round to closet integer and ensure prediciton in [1,100]
                cellcount_predicted = torch.round(outputs.type(torch.float))
                indx1 = (cellcount_predicted<1).nonzero()
                cellcount_predicted[indx1]=1.0
                indx100 = (cellcount_predicted>100).nonzero()
                cellcount_predicted[indx100]=100.0
                cellcount_truth = labels.type(torch.float)
                test_rmse += MSE_loss(cellcount_predicted, cellcount_truth).item() * len(images)

            ## output predictions?
            if OutputPred:  # output indecices and predicted cell count of misclassified samples, predicted cell count for all test images and predicted probabilties
                CellCount_test_predicted.extend(cellcount_predicted.cpu().numpy().tolist())
                CellCount_test_truth.extend(cellcount_truth.cpu().numpy().tolist())
                if args.PredType1 == 'Cl':
                    indx_misclassified.extend(i*len(images) + np.where(predicted.cpu().numpy() != labels.cpu().numpy())[0])
                    Prob_test_predicted[tmp:(tmp+images.shape[0])] = (CalcProb(outputs)).cpu().numpy()
            #end if
            tmp+=images.shape[0]
            i+=1
        # end for i
    Prob_test_predicted = Prob_test_predicted[0:N_samp]
    test_loss /= len(dataloader.dataset)
    test_acc /= len(dataloader.dataset)
    test_rmse /= len(dataloader.dataset);
    test_rmse = math.sqrt(test_rmse)
    if verbose:
        print('====> %s set loss: %.4f; %s set accuracy: %.4f; %s RMSE %.4f' % (
        mode, test_loss, mode, test_acc, mode, test_rmse))

    del images, labels, outputs;
    gc.collect()
    torch.cuda.empty_cache()

    if OutputPred:  # output indecices and predicted cell count of misclassified samples, predicted cell count for all test images and predicted probabilties
        CellCount_test_predicted = (np.array(CellCount_test_predicted)).astype(np.int)
        CellCount_test_truth = (np.array(CellCount_test_truth)).astype(np.int)
        return (test_loss, test_acc, test_rmse, np.array(indx_misclassified), CellCount_test_predicted[indx_misclassified], CellCount_test_truth[indx_misclassified], CellCount_test_predicted, Prob_test_predicted)
    else:
        return (test_loss, test_acc, test_rmse)


def ExtractFeatures(dataloader):
    model.eval()
    N_samp = dataloader.dataset.n_images
    batch_size = dataloader.batch_size
    # compute the length of extracted features
    img_tmp = torch.from_numpy(IMGs_train[0:1]).type(torch.float).cuda()
    _, features_tmp = model(img_tmp)
    dim_features = features_tmp.shape[1]
    # extract features
    ImgFeatures = np.zeros((N_samp+batch_size, dim_features), dtype=np.float32)
    with torch.no_grad():
        tmp=0; i=0
        data_iter = iter(dataloader)
        while i < len(data_iter):
            (images, labels, _, _) = data_iter.next()
            images = images.type(torch.float).to(device)
            _, features = model(images)
            ImgFeatures[tmp:(tmp+images.shape[0])]=features.cpu().numpy()
            tmp+=images.shape[0]
            i+=1
    return ImgFeatures[0:N_samp]


def EmbeddingFeatures(input, embedding_dim=1):
    if args.model2 == "GBT":
        embedding_dim=100
    # input: blur level or stain type
    input_unique = list(set(input))
    input_unique.sort()
    num_levels = len(input_unique)
    input_rev = np.zeros(input.shape).astype(np.int)
    # convert levels to 0,1,2,...
    for i in range(num_levels):
        indx = np.where(input == input_unique[i])[0]
        input_rev[indx] = i
    if embedding_dim > 1:
        embedding = nn.Embedding(num_levels, embedding_dim)
        input_rev = torch.from_numpy(input_rev).type(torch.long)
        output = embedding(input_rev)
        output = output.detach().numpy()
    else:  # do not do embedding
        output = input_rev.reshape(input.shape[0], 1)
    return output


#############################
# NROUND K-fold Cross Validation
#############################
MSE_loss = nn.MSELoss()
CE_loss = nn.CrossEntropyLoss()
CV_DataSplit_indices_all = []
valid_rmse_all = np.zeros((args.NROUND, args.K))  # valid RMSE
valid_acc_all = np.zeros((args.NROUND, args.K))  # valid accuracy
valid_rmse_duringTrain = np.zeros((args.NROUND, args.K, args.epochs))
valid_acc_duringTrain = np.zeros((args.NROUND, args.K, args.epochs))

misclassified_cellcount_predicted_valid = []
misclassified_cellcount_true_valid = []
misclassified_blur_valid = []
misclassified_stain_valid = []

misclassified_cellcount_predicted_test = []
misclassified_cellcount_true_test = []

start = timeit.default_timer()
for nround in range(args.NROUND):
    np.random.seed(args.seed + nround)  # specify seed

    # -------------------------------------------------------------------------
    # data augmentation
    if args.dataAugment:
        IMGs_train, CellCount_train, Blur_train, Stain_train = AugmentData(IMGs_train_raw, CellCount_train_raw, Blur_train_raw, Stain_train_raw, AugClass, formula_dir, nfake=nfake, flipping = args.DA_flip, one_formula_per_class=args.one_formula_per_class, show_sample_img=False, dump_fake=False, fakeImg_dir=None)
    else:
        IMGs_train = IMGs_train_raw
        CellCount_train = CellCount_train_raw
        Blur_train = Blur_train_raw
        Stain_train = Stain_train_raw

    Intensity_train = np.mean(IMGs_train, axis=(1,2,3))
    Intensity_train = Intensity_train.reshape(-1,1)

    #determine the loss function of CNNs; if num_classes>1, cross-entropy; else MSE
    if args.PredType1 == 'Cl':
        num_classes = len(np.array(list(set(CellCount_train))))
        criterion = CE_loss
    else:
        num_classes = 1
        criterion = MSE_loss

    N_train = len(IMGs_train)
    N_per_fold = int(N_train / args.K)
    assert N_train % args.K == 0
    (_, _, img_length, img_width) = IMGs_train.shape

    # -------------------------------------------------------------------------
    # Split data into K folds
    ## specify indices for samples in each fold
    indx_all = np.arange(N_train)
    np.random.shuffle(indx_all)
    indx_tmp = 0;
    i = 0
    CV_DataSplit_indices = np.zeros((args.K, N_per_fold),dtype=np.int)  # store data split in each cross-validation
    while i < args.K:
        CV_DataSplit_indices[i] = indx_all[indx_tmp:(indx_tmp + N_per_fold)]
        i += 1;
        indx_tmp += N_per_fold
    CV_DataSplit_indices_all.append(CV_DataSplit_indices)
    # end while i

    # -------------------------------------------------------------------------
    # start CV
    for k in range(args.K):
        # Specify indices of training and valiation sets in nround-th CV
        valid_indices = CV_DataSplit_indices[k]
        train_indices = [x for x in indx_all if x not in valid_indices]
        # Specify training and validation samples
        train_images = IMGs_train[train_indices]
        train_cellcount = CellCount_train[train_indices]
        train_blur = Blur_train[train_indices]
        train_stain = Stain_train[train_indices]
        train_intensity = Intensity_train[train_indices]

        valid_images = IMGs_train[valid_indices]
        valid_cellcount = CellCount_train[valid_indices]
        valid_stain = Stain_train[valid_indices]
        valid_blur = Blur_train[valid_indices]
        valid_intensity = Intensity_train[valid_indices]

        ## if Classification, then convert cell count into a categorical variable with M levels from 0 to M-1
        unique_cell_count = np.sort(np.array(list(set(train_cellcount))))
        ### the prediction type of model 1
        if args.PredType1 == 'Cl': # model 1 is a classification method
            train_labels_m1 = np.zeros(train_cellcount.shape).astype(np.int)
            valid_labels_m1 = np.zeros(valid_cellcount.shape).astype(np.int)
            for i in range(len(unique_cell_count)):
                indx_train = np.where(train_cellcount == unique_cell_count[i])[0]
                train_labels_m1[indx_train] = i
                indx_valid = np.where(valid_cellcount == unique_cell_count[i])[0]
                valid_labels_m1[indx_valid] = i
        else:
            train_labels_m1 = train_cellcount
            valid_labels_m1 = valid_cellcount
        ### the prediction type of model 2
        if args.PredType2 == 'Cl':
            assert args.PredType1 == 'Cl'
            train_labels_m2 = np.zeros(train_cellcount.shape).astype(np.int)
            valid_labels_m2 = np.zeros(valid_cellcount.shape).astype(np.int)
            for i in range(len(unique_cell_count)):
                indx_train = np.where(train_cellcount == unique_cell_count[i])[0]
                train_labels_m2[indx_train] = i
                indx_valid = np.where(valid_cellcount == unique_cell_count[i])[0]
                valid_labels_m2[indx_valid] = i
        else:
            train_labels_m2 = train_cellcount
            valid_labels_m2 = valid_cellcount

        # Data loader for training set, validation set and testing set
        train_dataset = BBBCDataset([train_images, train_labels_m1], transform=(0.5, 0.5),rotation = False,flipping=args.model1_flip)
        if args.model2_flip:
            train_images_flip, _, _, _ = Dataset_flip(train_images, train_labels_m2, train_blur, train_stain)
            train_labels_m2 = np.tile(train_labels_m2, 4)
            train_dataset_featureExtraction = BBBCDataset([train_images_flip, train_labels_m2], transform=(0.5, 0.5))
            del train_images_flip; gc.collect()
        else:
            train_dataset_featureExtraction = BBBCDataset([train_images, train_labels_m2], transform=(0.5, 0.5))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=8)
        train_loader_featureExtraction = DataLoader(train_dataset_featureExtraction, batch_size=args.batch_size_extract, shuffle=False, num_workers=8)

        valid_dataset = BBBCDataset([valid_images, valid_labels_m1], transform=(0.5, 0.5))
        valid_dataset_featureExtraction = BBBCDataset([valid_images, valid_labels_m2], transform=(0.5, 0.5))
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size_valid, shuffle=False, num_workers=8)
        valid_loader_featureExtraction = DataLoader(valid_dataset_featureExtraction, batch_size=args.batch_size_extract, shuffle=False, num_workers=8)


        # initialize model and optimizer
        # Train model
        # first model
        if args.model1 not in ['LReg', 'PReg', 'QReg']:
            save_filename = save_models_folder + 'ckpt_Exp2_CV_' + args.model1 + '(' + args.PredType1 + ')_epoch' + str(args.epochs) + '_seed' + str(args.seed) + '_Round' + str(nround) + '_k' + str(k) + '_CNNFlip_' + str(args.model1_flip) + '_DA_' + str(args.dataAugment) + '_OneForOne_' + str(args.one_formula_per_class) + '.pth'
            if args.model1[0:3] == "VGG":
                model = VGG_resized(args.model1, ngpu, num_classes).to(device)
            elif args.model1 == "ResNet18":
                model = ResNet18_resized(ngpu, num_classes).to(device)
            elif args.model1 == "ResNet34":
                model = ResNet34_resized(ngpu, num_classes).to(device)
            else:
                raise Exception("Model {} unknown.".format(args.model1))

            optimizer = optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_dacay)

            if not os.path.isfile(save_filename):
                for epoch in range(args.epochs):
                    train_loss = train(epoch, train_loader, verbose=False)
                    (_, valid_acc, valid_rmse) = test(valid_loader, mode='Valid', verbose=False)
                    valid_rmse_duringTrain[nround, k, epoch] = valid_rmse
                    valid_acc_duringTrain[nround, k, epoch] = valid_acc
                    print("Round [%d/%d], Fold [%d/%d], Epoch [%d/%d]: train loss %.4f, valid acc %.4f, valid rmse %.4f" % (nround + 1, args.NROUND, k + 1, args.K, epoch + 1, args.epochs, train_loss, valid_acc, valid_rmse))
                    # save model
                    if epoch + 1 == args.epochs:
                        # save trained model
                        save_filename = save_models_folder + 'ckpt_Exp2_CV_' + args.model1 + '(' + args.PredType1 + ')_epoch' + str(args.epochs) + '_seed' + str(args.seed) + '_Round' + str(nround) + '_k' + str(k) + '_CNNFlip_' + str(args.model1_flip) + '_DA_' + str(args.dataAugment) + '_OneForOne_' + str(args.one_formula_per_class) + '.pth'
                        save_model(epoch, save_filename, keep_optimizer=False)
                end = timeit.default_timer()
                print("Time elapsed: %f" % (end - start))
                torch.cuda.empty_cache()
            else:
                # load pre-trained cnn
                checkpoint = torch.load(save_filename)
                model.load_state_dict(checkpoint['net_state_dict'])

        elif args.model1 == "LReg": #linear regression
            assert args.PredType1 == 'Reg'
            CellCount_valid_true = []
            CellCount_valid_predicted = []
            # fit 6 linear regressions and validate/test it
            for tmp_stain in [1,2]:
                for tmp_blur in [1,23,48]:
                    # training
                    stain_blur_indx = np.logical_and(train_stain==tmp_stain, train_blur==tmp_blur)
                    Intensity_train_stain_blur = train_intensity[stain_blur_indx]
                    CellCount_train_stain_blur = (train_cellcount[stain_blur_indx]).reshape(-1,1)
                    regr = linear_model.LinearRegression()
                    regr.fit(Intensity_train_stain_blur, CellCount_train_stain_blur)
                    # validation
                    stain_blur_indx = np.logical_and(valid_stain==tmp_stain, valid_blur==tmp_blur)
                    Intensity_valid_stain_blur = valid_intensity[stain_blur_indx]
                    CellCount_valid_stain_blur = valid_cellcount[stain_blur_indx]
                    CellCount_valid_Pred = (regr.predict(Intensity_valid_stain_blur)).reshape(-1)
                    CellCount_valid_true.extend(list(CellCount_valid_stain_blur))
                    CellCount_valid_predicted.extend(list(CellCount_valid_Pred))
            #end for
            CellCount_valid_true = np.array(CellCount_valid_true)
            CellCount_valid_predicted = np.array(CellCount_valid_predicted)
            ## rounding
            CellCount_valid_predicted = (np.around(CellCount_valid_predicted)).astype(np.int)
            indx1 = np.where(CellCount_valid_predicted<1)[0]
            CellCount_valid_predicted[indx1]=1
            indx100 = np.where(CellCount_valid_predicted>100)[0]
            CellCount_valid_predicted[indx100]=100
            #end linear regression training

        elif args.model1 == "QReg": #linear regression
            assert args.PredType1 == 'Reg'
            CellCount_valid_true = []
            CellCount_valid_predicted = []
            # fit 6 linear regressions and validate/test it
            for tmp_stain in [1,2]:
                for tmp_blur in [1,23,48]:
                    # training
                    stain_blur_indx = np.logical_and(train_stain==tmp_stain, train_blur==tmp_blur)
                    if tmp_stain == 1:
                        Intensity_train_stain_blur = train_intensity[stain_blur_indx]
                    else:
                        Intensity_train_stain_blur = np.concatenate((train_intensity[stain_blur_indx], train_intensity[stain_blur_indx]**2), axis=1)
                    CellCount_train_stain_blur = (train_cellcount[stain_blur_indx]).reshape(-1,1)
                    regr = linear_model.LinearRegression()
                    regr.fit(Intensity_train_stain_blur, CellCount_train_stain_blur)
                    # validation
                    stain_blur_indx = np.logical_and(valid_stain==tmp_stain, valid_blur==tmp_blur)
                    if tmp_stain == 1:
                        Intensity_valid_stain_blur = valid_intensity[stain_blur_indx]
                    else:
                        Intensity_valid_stain_blur = np.concatenate((valid_intensity[stain_blur_indx],valid_intensity[stain_blur_indx]**2), axis=1)
                    CellCount_valid_stain_blur = valid_cellcount[stain_blur_indx]
                    CellCount_valid_Pred = (regr.predict(Intensity_valid_stain_blur)).reshape(-1)
                    CellCount_valid_true.extend(list(CellCount_valid_stain_blur))
                    CellCount_valid_predicted.extend(list(CellCount_valid_Pred))
            #end for
            CellCount_valid_true = np.array(CellCount_valid_true)
            CellCount_valid_predicted = np.array(CellCount_valid_predicted)
            ## rounding
            CellCount_valid_predicted = (np.around(CellCount_valid_predicted)).astype(np.int)
            indx1 = np.where(CellCount_valid_predicted<1)[0]
            CellCount_valid_predicted[indx1]=1
            indx100 = np.where(CellCount_valid_predicted>100)[0]
            CellCount_valid_predicted[indx100]=100
            #end linear regression training

        elif args.model1 == "PReg": #Poisson regression
            assert args.PredType1 == 'Reg'
            CellCount_valid_true = []
            CellCount_valid_predicted = []
            # fit 6 Poisson regressions and validate/test it
            for tmp_stain in [1,2]:
                for tmp_blur in [1,23,48]:
                    # training
                    stain_blur_indx = np.logical_and(train_stain==tmp_stain, train_blur==tmp_blur)
                    Intensity_train_stain_blur = train_intensity[stain_blur_indx]
                    CellCount_train_stain_blur = train_cellcount[stain_blur_indx].reshape(-1,1)
                    intensity_count = pd.DataFrame(np.concatenate((CellCount_train_stain_blur,Intensity_train_stain_blur),1),columns=["count","intensity"])
                    regr = smf.glm('count ~ intensity', data=intensity_count, family=sm.families.Poisson()).fit()
                    # regr = (sm.GLM(CellCount_train_stain_blur, Intensity_train_stain_blur, family = sm.families.Poisson())).fit()
                    # validation
                    stain_blur_indx = np.logical_and(valid_stain==tmp_stain, valid_blur==tmp_blur)
                    Intensity_valid_stain_blur = valid_intensity[stain_blur_indx]
                    CellCount_valid_stain_blur = valid_cellcount[stain_blur_indx]
                    Intensity_valid_stain_blur = pd.DataFrame(Intensity_valid_stain_blur,columns=["intensity"])
                    CellCount_valid_Pred = (np.array(regr.predict(Intensity_valid_stain_blur))).reshape(-1)
                    CellCount_valid_true.extend(list(CellCount_valid_stain_blur))
                    CellCount_valid_predicted.extend(list(CellCount_valid_Pred))
            #end fitting
            CellCount_valid_true = np.array(CellCount_valid_true)
            CellCount_valid_predicted = np.array(CellCount_valid_predicted)
            ## rounding
            CellCount_valid_predicted = (np.around(CellCount_valid_predicted)).astype(np.int)
            indx1 = np.where(CellCount_valid_predicted<1)[0]
            CellCount_valid_predicted[indx1]=1
            indx100 = np.where(CellCount_valid_predicted>100)[0]
            CellCount_valid_predicted[indx100]=100
            #end Poisson regression training
        #end if args.model1

        # Prediciton or train a subsequent model for prediction
        if args.model2 == "CNN":
            # Validate model
            (_, valid_acc_all[nround, k], valid_rmse_all[nround, k], indx_misclassified, misclassified_cellcount_predicted, misclassified_cellcount_truth, CellCount_valid_predicted, _) = test(valid_loader, mode='Valid', OutputPred=True, verbose=False)
            torch.cuda.empty_cache()
        elif args.model2 in ['RF', 'GBT']:
            # extract features
            TrainFeatures = ExtractFeatures(train_loader_featureExtraction)
            ValidFeatures = ExtractFeatures(valid_loader_featureExtraction)
            # use numerical features?
            if args.useNumericalFeatures:
                TrainNumFeatures = np.concatenate((EmbeddingFeatures(train_blur), EmbeddingFeatures(train_stain)),axis=1)
                TrainFeatures = np.concatenate((TrainFeatures, TrainNumFeatures), axis=1)
                ValidNumFeatures = np.concatenate((EmbeddingFeatures(Blur_train[valid_indices]), EmbeddingFeatures(Stain_train[valid_indices])),axis=1)
                ValidFeatures = np.concatenate((ValidFeatures, ValidNumFeatures), axis=1)
            # train RF and GBT for prediction
            if args.model2 == "RF":  # random forest
                n_estimator = args.RF_NTree
                if args.RF_MaxDepth>=1:
                    max_depth = args.RF_MaxDepth
                else:
                    max_depth = None #fully grown
                if args.RF_MaxFeatures == "None":
                    args.RF_MaxFeatures = None
                max_features = args.RF_MaxFeatures  # default: None; "sqrt"
                if args.PredType2 == "Cl":
                    # train random forest
                    clf = RandomForestClassifier(n_estimators=n_estimator, criterion='gini', max_depth=max_depth, max_features=max_features, n_jobs=multiprocessing.cpu_count())
                    clf.fit(TrainFeatures, train_labels_m2)
                    # prediction
                    ## validation
                    Labels_valid_predicted = clf.predict(ValidFeatures)
                    CellCount_valid_predicted = unique_cell_count[Labels_valid_predicted.astype(np.int)]
                else:
                    # train random forest
                    regr = RandomForestRegressor(n_estimators=n_estimator, criterion='mse', max_depth=max_depth, max_features=max_features, n_jobs=multiprocessing.cpu_count())
                    regr.fit(TrainFeatures, train_labels_m2)
                    # prediction
                    ## validation
                    CellCount_valid_predicted = regr.predict(ValidFeatures)
                    CellCount_valid_predicted = (np.around(CellCount_valid_predicted)).astype(np.int)
                    indx1 = np.where(CellCount_valid_predicted < 1)[0]
                    CellCount_valid_predicted[indx1] = 1
                    indx100 = np.where(CellCount_valid_predicted > 100)[0]
                    CellCount_valid_predicted[indx100] = 100
            elif args.model2 == "GBT":  # GBT
                booster = 'gbtree'
                max_depth = args.GBT_MaxDepth
                eta = args.GBT_eta
                subsample = 1  # Subsample ratio of the training instances. prevent overfitting
                nthread = multiprocessing.cpu_count()
                num_round = args.GBT_Round
                if args.PredType2 == "Cl":
                    # train GBT
                    param = {'booster': booster, 'max_depth': max_depth, 'eta': eta, 'objective': 'multi:softmax', 'subsample': subsample, 'nthread': nthread, 'num_class': num_classes}
                    dtrain = xgb.DMatrix(TrainFeatures, label=train_labels_m2)
                    bst = xgb.train(param, dtrain, num_round)
                    # prediction
                    ## validation
                    dvalid = xgb.DMatrix(ValidFeatures)
                    Labels_valid_predicted = bst.predict(dvalid)
                    CellCount_valid_predicted = unique_cell_count[Labels_valid_predicted.astype(np.int)]
                else:
                    # train GBT
                    # reg:linear or count:poisson
                    if args.GBT_loss == "MSE":
                        gbt_loss = "reg:linear"
                    else:
                        gbt_loss = "count:poisson"
                    param = {'booster': booster, 'max_depth': max_depth, 'eta': eta, 'objective': gbt_loss,
                             'subsample': subsample, 'nthread': nthread}
                    dtrain = xgb.DMatrix(TrainFeatures, label=train_labels_m2)
                    bst = xgb.train(param, dtrain, num_round)
                    # prediction
                    ## validation
                    dvalid = xgb.DMatrix(ValidFeatures)
                    CellCount_valid_predicted = bst.predict(dvalid)
                    CellCount_valid_predicted = (np.around(CellCount_valid_predicted)).astype(np.int)
                    indx1 = np.where(CellCount_valid_predicted < 1)[0]
                    CellCount_valid_predicted[indx1] = 1
                    indx100 = np.where(CellCount_valid_predicted > 100)[0]
                    CellCount_valid_predicted[indx100] = 100


        # misclassified validation images and rmse
        if args.model2 in ['RF', 'GBT']:
            indx_tmp = np.where(CellCount_valid_predicted != valid_cellcount)[0]
            misclassified_cellcount_predicted_valid.extend((CellCount_valid_predicted[indx_tmp]).tolist())
            misclassified_cellcount_true_valid.extend((valid_cellcount[indx_tmp]).tolist())
            misclassified_blur_valid.extend(valid_blur[indx_tmp].tolist())
            misclassified_stain_valid.extend(valid_stain[indx_tmp].tolist())

            # compute valid/test accuracy, valid/test rmse for the subsequent model
            valid_acc_all[nround, k] = np.mean(CellCount_valid_predicted == CellCount_train[valid_indices])
            valid_rmse_all[nround, k] = np.sqrt(np.mean((CellCount_valid_predicted.astype(np.float) - CellCount_train[valid_indices].astype(np.float)) ** 2))
        elif args.model2 == 'None' and args.model1 in ['LReg', 'PReg', 'QReg']: #for linear regression and Poisson regression
            # compute valid/test accuracy, valid/test rmse for the subsequent model
            valid_acc_all[nround, k] = np.mean(CellCount_valid_predicted == CellCount_valid_true)
            valid_rmse_all[nround, k] = np.sqrt(np.mean((CellCount_valid_predicted.astype(np.float) - CellCount_valid_true.astype(np.float)) ** 2))

        print("Round [%d/%d], Fold [%d/%d]: valid acc %.4f, valid rmse %.4f" % (nround + 1, args.NROUND, k + 1, args.K, valid_acc_all[nround, k], valid_rmse_all[nround, k]))
    # end for k
# end for nround
end = timeit.default_timer()
print("Total time elapsed: %f" % (end - start))

# print averaged validation accuracy and rmse
print("Avg. valid acc %.4f (%.4f), Avg. valid rmse %.4f (%.4f)" % (
valid_acc_all.mean(), valid_acc_all.std(), valid_rmse_all.mean(), valid_rmse_all.std()))
