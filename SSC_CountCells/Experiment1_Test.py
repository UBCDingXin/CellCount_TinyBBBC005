"""

Test for Experiment 1

"""

import os
#wd = 'your path/SSC_CountCells'
wd = '/home/xin/OneDrive/Working_directory/Annotation-free_Cell_Counting/SSC_CountCells'

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
import gc
import timeit
import scipy.misc
import PIL
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb # install py-xgboost
import multiprocessing
from utils import *
from models import *
#from DataAugmentation import *

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
parser.add_argument('--model1', type=str, default='ResNet34',help='First model (default: "ResNet34"); Candidates: LReg, QReg, PReg, VGG11, VGG13, VGG16, ResNet18, ResNet34')
parser.add_argument('--model2', type=str, default = 'CNN',help='Second model (default: None); Candidates: None, CNN, RF, GBT')
parser.add_argument('--PredType1', type=str, default='Cl',help='Prediction type for the first model (default: "Cl"); Candidates: Cl and Reg')
parser.add_argument('--PredType2', type=str, default='Cl',help='Prediction type for the second model (default: "Cl"); Candidates: Cl and Reg')
parser.add_argument('--EnsemblePred', action='store_true', default=False,
                    help='Ensemble ResNet34 and QReg?')

parser.add_argument('--RF_NTree', type=int, default=500, metavar='N',
                    help='Number of trees in RF')
parser.add_argument('--RF_MaxDepth', type=int, default=20, metavar='N',
                    help='Max depth of a single tree in RF') #0 means fully grown
parser.add_argument('--RF_MaxFeatures', type=str, default='sqrt',
                    help='Max features for RF (default: "sqrt"); Candidates: None and sqrt' ) #None means use all features
parser.add_argument('--GBT_loss', type=str, default='MSE',
                    help='Loss function for GBT (default:MSE); Candidates: MSE and Poisson')
parser.add_argument('--GBT_MaxDepth', type=int, default=20,
                    help='Maximum depth of a single tree')
parser.add_argument('--GBT_eta', type=float, default=0.1,
                    help='Step size shrinkage')
parser.add_argument('--GBT_Round', type=int, default=100,
                    help='Rounds of boosting')

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
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model1_flip', action='store_true', default=False,
                    help='Vertically or horizatally flip images for CNN training')
parser.add_argument('--model2_flip', action='store_true', default=False,
                    help='Vertically or horizatally flip images for RF/XGBoost training')


args = parser.parse_args()
# cuda
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
ngpu = torch.cuda.device_count()  # number of gpus
args.base_lr = args.base_lr * ngpu

# random seed
random.seed(args.seed)
torch.manual_seed(args.seed) #control CNN
torch.backends.cudnn.deterministic = True
np.random.seed(args.seed) #control random forest

# directories for checkpoint and images
save_models_folder = wd + '/Output/saved_models/'
if not os.path.exists(save_models_folder):
    os.makedirs(save_models_folder)
PreTrainedModel_filename = save_models_folder + 'ckpt_Exp1_' + args.model1 + '(' + args.PredType1 + ')_epoch' + str(args.epochs) + '_seed' + str(args.seed) + '_CNNFlip_' + str(args.model1_flip) + '.pth'

save_images_folder = wd + '/Output/saved_images/'
if not os.path.exists(save_images_folder):
    os.makedirs(save_images_folder)
TrainCurves_filename = save_images_folder + 'TrainCurves_Exp1_' + args.model1 + '(' + args.PredType1 + ')_epoch' + str(args.epochs) + '_seed' + str(args.seed) + '_CNNFlip_' + str(args.model1_flip) + '.pdf'
TestCurves_filename = save_images_folder + 'TestCurves_Exp1_' + args.model1 + '(' + args.PredType1 + ')_epoch' + str(args.epochs) + '_seed' + str(args.seed) + '_CNNFlip_' + str(args.model1_flip) + '.pdf'


#############################
# Load data from h5 file
#############################
hf = h5py.File('./data/CellCount_resized_dataset.h5', 'r')
IMGs_train = hf['IMGs_resized_train'].value
Blur_train = hf['Blur_train'].value
Stain_train = hf['Stain_train'].value
CellCount_train = hf['CellCount_train'].value
IMGs_test = hf['IMGs_resized_test'].value
Blur_test = hf['Blur_test'].value
Stain_test = hf['Stain_test'].value
CellCount_test = hf['CellCount_test'].value
hf.close()

Intensity_train = np.sum(IMGs_train, axis=(1,2,3))
Intensity_test = np.sum(IMGs_test, axis=(1,2,3))
Intensity_train = Intensity_train.reshape(-1,1)
Intensity_test = Intensity_test.reshape(-1,1)

# if Classification, then convert cell count into a categorical variable with 24 levels from 0 to 23
unique_cell_count = np.array(list(set(CellCount_train)))
### the prediction type of model 1
if args.PredType1 == 'Cl':
    Labels_train_m1 = np.zeros(CellCount_train.shape).astype(np.int)
    Labels_test_m1 = np.zeros(CellCount_test.shape).astype(np.int)
    for i in range(len(unique_cell_count)):
        indx_train = np.where(CellCount_train==unique_cell_count[i])[0]
        Labels_train_m1[indx_train] = i
        indx_test = np.where(CellCount_test==unique_cell_count[i])[0]
        Labels_test_m1[indx_test] = i
else:
    Labels_train_m1 = CellCount_train
    Labels_test_m1 = CellCount_test
### the prediction type of model 2
if args.PredType2 == 'Cl':
    Labels_train_m2 = np.zeros(CellCount_train.shape).astype(np.int)
    Labels_test_m2 = np.zeros(CellCount_test.shape).astype(np.int)
    for i in range(len(unique_cell_count)):
        indx_train = np.where(CellCount_train==unique_cell_count[i])[0]
        Labels_train_m2[indx_train] = i
        indx_test = np.where(CellCount_test==unique_cell_count[i])[0]
        Labels_test_m2[indx_test] = i
else:
    Labels_train_m2 = CellCount_train
    Labels_test_m2 = CellCount_test


train_dataset = BBBCDataset([IMGs_train, Labels_train_m1, Blur_train, Stain_train], transform = (0.5, 0.5), rotation = False, flipping = args.model1_flip)
if args.model2_flip:
    IMGs_train_flip, _, _, _ = Dataset_flip(IMGs_train, Labels_train_m2, Blur_train, Stain_train)
    Labels_train_m2 = np.tile(Labels_train_m2, 4)
    train_dataset_featureExtraction = BBBCDataset([IMGs_train_flip, Labels_train_m2], transform=(0.5, 0.5))
else:
    train_dataset_featureExtraction = BBBCDataset([IMGs_train, Labels_train_m2, Blur_train, Stain_train], transform=(0.5, 0.5))
test_dataset = BBBCDataset([IMGs_test, Labels_test_m1, Blur_test, Stain_test], transform = (0.5, 0.5))
test_dataset_featureExtraction = BBBCDataset([IMGs_test, Labels_test_m2, Blur_test, Stain_test], transform = (0.5, 0.5))

train_loader = DataLoader(train_dataset, batch_size = args.batch_size_train, shuffle=True, num_workers=8)
train_loader_featureExtraction = DataLoader(train_dataset_featureExtraction, batch_size = args.batch_size_extract, shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size = args.batch_size_test, shuffle=False, num_workers=8)
test_loader_featureExtraction = DataLoader(test_dataset_featureExtraction, batch_size = args.batch_size_extract, shuffle=False, num_workers=8)




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

def train(epoch):
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
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch+1, batch_idx * len(images), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(images)))
    #end for batch_idx
    train_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.6f}'.format(
          epoch+1, train_loss))
    del images, labels, outputs; gc.collect()
    torch.cuda.empty_cache()
    return train_loss

def test(OutputPred=False, verbose=True):
    model.eval()
    CalcProb = nn.Softmax(dim=1) #softmax function which is used for  computing predicted probabilties
    N_test = IMGs_test.shape[0]
    assert N_test % args.batch_size_test == 0
    test_loss = 0
    test_acc = 0 #test accuracy
    test_rmse = 0
    indx_misclassified = []
    CellCount_test_predicted = []
    Prob_test_predicted = np.zeros((N_test, num_classes))
    with torch.no_grad():
        for i, (images, labels, _, _) in enumerate(test_loader):
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
                cellcount_predicted = torch.round(outputs.type(torch.float))
                indx1 = (cellcount_predicted<1).nonzero()
                cellcount_predicted[indx1]=1
                indx100 = (cellcount_predicted>100).nonzero()
                cellcount_predicted[indx100]=100
                cellcount_truth = labels.type(torch.float)
                test_rmse += MSE_loss(cellcount_predicted, cellcount_truth).item() * len(images)

            ## output predictions?
            if OutputPred:  # output indecices and predicted cell count of misclassified samples, predicted cell count for all test images and predicted probabilties
                CellCount_test_predicted.extend(cellcount_predicted.cpu().numpy().tolist())
                if args.PredType1 == 'Cl':
                    indx_misclassified.extend(i*len(images)+np.where(predicted.cpu().numpy()!=labels.cpu().numpy())[0])
                    Prob_test_predicted[int(i*args.batch_size_test):int((i+1)*args.batch_size_test)] = (CalcProb(outputs)).cpu().numpy()
            #end if
        #end for i
    test_loss /= len(test_loader.dataset)
    test_acc /= len(test_loader.dataset)
    test_rmse /= len(test_loader.dataset); test_rmse = math.sqrt(test_rmse)
    if verbose:
        print('====> Test set loss: %.4f; Test set accuracy: %.4f; Test RMSE %.4f' % (test_loss, test_acc, test_rmse))

    del images, labels, outputs; gc.collect()
    torch.cuda.empty_cache()

    if OutputPred: #output indecices and predicted cell count of misclassified samples, predicted cellcounts of all test images, and predicted probabilties
        CellCount_test_predicted = (np.array(CellCount_test_predicted)).astype(np.int)
        return (test_loss, test_acc, test_rmse, np.array(indx_misclassified), CellCount_test_predicted[indx_misclassified], CellCount_test_predicted, Prob_test_predicted)
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

###############################
# If ensemble ResNet34 and QReg, train three QReg for each combination at here!
if args.EnsemblePred:
    # !!!!!!!!!!!!!!!!!!!!!!!!!
    # Note that upper bound is based on intensity_min; lower bound is based on intensity_max
    qreg_models = []; lreg_models = []
    qreg_ub_models = []; lreg_ub_models = []
    qreg_lb_models = []; lreg_lb_models = []
    for tmp_stain in [1,2]:
        for tmp_blur in [1,23,48]:
            intensity_mins = []
            intensity_maxs = []
            for i in range(len(unique_cell_count)):
                current_cellcount = unique_cell_count[i]
                indx_stain_blur_count = np.where(((Blur_train==tmp_blur) * (Stain_train==tmp_stain) * (CellCount_train==current_cellcount))==True)[0]
                intensity_mins.append(np.min(Intensity_train[indx_stain_blur_count]))
                intensity_maxs.append(np.max(Intensity_train[indx_stain_blur_count]))
            y = unique_cell_count.reshape(-1,1)
            x_min = np.array(intensity_mins).reshape(-1,1)
            x_max = np.array(intensity_maxs).reshape(-1,1)
            x = Intensity_train[(Blur_train==tmp_blur) * (Stain_train==tmp_stain)]
            if tmp_stain == 1:
                regr_min = linear_model.LinearRegression()
                regr_min.fit(x_min, y)
                regr_max = linear_model.LinearRegression()
                regr_max.fit(x_max, y)
                regr = linear_model.LinearRegression()
                regr.fit(x, CellCount_train[(Blur_train==tmp_blur) * (Stain_train==tmp_stain)])
                lreg_ub_models.append(regr_min)
                lreg_lb_models.append(regr_max)
                lreg_models.append(regr)
            else:
                regr_min = linear_model.LinearRegression()
                regr_min.fit(np.concatenate((x_min,x_min**2),axis=1), y)
                regr_max = linear_model.LinearRegression()
                regr_max.fit(np.concatenate((x_max,x_max**2),axis=1), y)
                regr = linear_model.LinearRegression()
                regr.fit(np.concatenate((x,x**2),axis=1),CellCount_train[(Blur_train==tmp_blur) * (Stain_train==tmp_stain)])
                qreg_ub_models.append(regr_min)
                qreg_lb_models.append(regr_max)
                qreg_models.append(regr)
#end if args.EnsemblePred:


###############################
# Train First model
###############################
if args.model1 not in ['LReg', 'QReg', 'PReg']:
    MSE_loss = nn.MSELoss()
    CE_loss = nn.CrossEntropyLoss()
    if args.PredType1 == 'Cl':
        num_classes = len(list(set(CellCount_train)))
        criterion = CE_loss
    else:
        num_classes = 1
        criterion = MSE_loss

    if args.model1[0:3]=="VGG":
        model = VGG_resized(args.model1, ngpu, num_classes).to(device)
    elif args.model1 == "ResNet18":
        model = ResNet18_resized(ngpu, num_classes).to(device)
    elif args.model1 == "ResNet34":
        model = ResNet34_resized(ngpu, num_classes).to(device)
    else:
        raise Exception("Model {} unknown.".format(args.cnn))

    optimizer = optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_dacay)

    if not os.path.isfile(PreTrainedModel_filename):
        train_loss_all = np.zeros(args.epochs)
        test_loss_all = np.zeros(args.epochs)
        test_acc_all = np.zeros(args.epochs)
        test_rmse_all = np.zeros(args.epochs)
        start = timeit.default_timer()
        for epoch in range(args.epochs):
            train_loss_all[epoch] = train(epoch)
            torch.cuda.empty_cache()
            (test_loss_all[epoch], test_acc_all[epoch], test_rmse_all[epoch]) = test()
            # save model
            if epoch+1 == args.epochs:
                # save trained model
                save_filename = save_models_folder + 'ckpt_Exp1_' + args.model1 + '(' + args.PredType1 + ')_epoch' + str(epoch+1) + '_seed' + str(args.seed) + '_CNNFlip_' + str(args.model1_flip) + '.pth'
                save_model(epoch, save_filename, keep_optimizer=True)
        stop = timeit.default_timer()
        print("Time elapsed %.3f" % (stop-start))
        # plot training curves
        losses = [train_loss_all, test_loss_all]
        loss_names = ['Train loss', 'Test loss']
        PlotTrainCurves(losses, loss_names, TrainCurves_filename)
        losses = [test_acc_all, test_rmse_all]
        loss_names = ['Test accuracy', 'Test RMSE']
        PlotTrainCurves(losses, loss_names, TestCurves_filename)
        torch.cuda.empty_cache()
    else:
        #-------------------------------------------
        # load pre-trained cnn
        checkpoint = torch.load(PreTrainedModel_filename)
        model.load_state_dict(checkpoint['net_state_dict'])

elif args.model1 == "LReg": #linear regression
    assert args.PredType1 == 'Reg'
    CellCount_test_true = []
    CellCount_test_predicted = []
    # fit 6 linear regressions and validate/test it
    for tmp_stain in [1,2]:
        for tmp_blur in [1,23,48]:
            # training
            stain_blur_indx = np.logical_and(Stain_train==tmp_stain, Blur_train==tmp_blur)
            Intensity_train_stain_blur = Intensity_train[stain_blur_indx]
            CellCount_train_stain_blur = (CellCount_train[stain_blur_indx]).reshape(-1,1)
            regr = linear_model.LinearRegression()
            regr.fit(Intensity_train_stain_blur, CellCount_train_stain_blur)
            # test
            stain_blur_indx = np.logical_and(Stain_test==tmp_stain, Blur_test==tmp_blur)
            Intensity_test_stain_blur = Intensity_test[stain_blur_indx]
            CellCount_test_stain_blur = CellCount_test[stain_blur_indx]
            CellCount_test_Pred = (regr.predict(Intensity_test_stain_blur)).reshape(-1)
            CellCount_test_true.extend(list(CellCount_test_stain_blur))
            CellCount_test_predicted.extend(list(CellCount_test_Pred))
    #end for
    CellCount_test_true = np.array(CellCount_test_true)
    CellCount_test_predicted = np.array(CellCount_test_predicted)
    ## rounding
    CellCount_test_predicted = (np.around(CellCount_test_predicted)).astype(np.int)
    indx1 = np.where(CellCount_test_predicted<1)[0]
    CellCount_test_predicted[indx1]=1
    indx100 = np.where(CellCount_test_predicted>100)[0]
    CellCount_test_predicted[indx100]=100
    #end linear regression training

elif args.model1 == "QReg": #linear regression
    assert args.PredType1 == 'Reg'
    CellCount_test_true = []
    CellCount_test_predicted = []
    # fit 6 linear regressions and validate/test it
    for tmp_stain in [1,2]:
        for tmp_blur in [1,23,48]:
            # training
            stain_blur_indx = np.logical_and(Stain_train==tmp_stain, Blur_train==tmp_blur)
            if tmp_stain == 1:
                Intensity_train_stain_blur = Intensity_train[stain_blur_indx]
            else:
                Intensity_train_stain_blur = np.concatenate((Intensity_train[stain_blur_indx], Intensity_train[stain_blur_indx]**2), axis=1)
            CellCount_train_stain_blur = (CellCount_train[stain_blur_indx]).reshape(-1,1)
            regr = linear_model.LinearRegression()
            regr.fit(Intensity_train_stain_blur, CellCount_train_stain_blur)
            # test
            stain_blur_indx = np.logical_and(Stain_test==tmp_stain, Blur_test==tmp_blur)
            if tmp_stain == 1:
                Intensity_test_stain_blur = Intensity_test[stain_blur_indx]
            else:
                Intensity_test_stain_blur = np.concatenate((Intensity_test[stain_blur_indx], Intensity_test[stain_blur_indx]**2), axis=1)
            CellCount_test_stain_blur = CellCount_test[stain_blur_indx]
            CellCount_test_Pred = (regr.predict(Intensity_test_stain_blur)).reshape(-1)
            CellCount_test_true.extend(list(CellCount_test_stain_blur))
            CellCount_test_predicted.extend(list(CellCount_test_Pred))
    #end for
    CellCount_test_true = np.array(CellCount_test_true)
    CellCount_test_predicted = np.array(CellCount_test_predicted)
    ## rounding
    CellCount_test_predicted = (np.around(CellCount_test_predicted)).astype(np.int)
    indx1 = np.where(CellCount_test_predicted<1)[0]
    CellCount_test_predicted[indx1]=1
    indx100 = np.where(CellCount_test_predicted>100)[0]
    CellCount_test_predicted[indx100]=100
    #end linear regression training

elif args.model1 == "PReg": #Poisson regression
    assert args.PredType1 == 'Reg'
    CellCount_test_true = []
    CellCount_test_predicted = []
    # fit 6 Poisson regressions and validate/test it
    for tmp_stain in [1,2]:
        for tmp_blur in [1,23,48]:
            # training
            stain_blur_indx = np.logical_and(Stain_train==tmp_stain, Blur_train==tmp_blur)
            Intensity_train_stain_blur = Intensity_train[stain_blur_indx]
            CellCount_train_stain_blur = CellCount_train[stain_blur_indx].reshape(-1,1)
            if tmp_stain == 1:
              intensity_count = pd.DataFrame(np.concatenate((CellCount_train_stain_blur,Intensity_train_stain_blur),1),columns=["count","intensity"])
              regr = smf.glm('count ~ intensity', data=intensity_count, family=sm.families.Poisson()).fit()
            else:
              intensity_count = pd.DataFrame(np.concatenate((CellCount_train_stain_blur,Intensity_train_stain_blur,Intensity_train_stain_blur**2),1),columns=["count","intensity","intensity_sqr"])
              regr = smf.glm('count ~ intensity + intensity_sqr', data=intensity_count, family=sm.families.Poisson()).fit()
            # test
            stain_blur_indx = np.logical_and(Stain_test==tmp_stain, Blur_test==tmp_blur)
            Intensity_test_stain_blur = Intensity_test[stain_blur_indx]
            CellCount_test_stain_blur = CellCount_test[stain_blur_indx]
            if tmp_stain == 1:
                Intensity_test_stain_blur = pd.DataFrame(Intensity_test_stain_blur,columns=["intensity"])
            else:
                Intensity_test_stain_blur = pd.DataFrame(np.concatenate((Intensity_test_stain_blur,Intensity_test_stain_blur**2),1),columns=["intensity","intensity_sqr"])
            CellCount_test_Pred = (np.array(regr.predict(Intensity_test_stain_blur))).reshape(-1)
            CellCount_test_true.extend(list(CellCount_test_stain_blur))
            CellCount_test_predicted.extend(list(CellCount_test_Pred))
    #end fitting
    CellCount_test_true = np.array(CellCount_test_true)
    CellCount_test_predicted = np.array(CellCount_test_predicted)
    ## rounding
    CellCount_test_predicted = (np.around(CellCount_test_predicted)).astype(np.int)
    indx1 = np.where(CellCount_test_predicted<1)[0]
    CellCount_test_predicted[indx1]=1
    indx100 = np.where(CellCount_test_predicted>100)[0]
    CellCount_test_predicted[indx100]=100
    #end Poisson regression training
#end if args.model1


###############################
# Train Second model
###############################
if args.model2 == "CNN":
    # Test model
    (test_loss, test_acc, test_rmse, indx_misclassified, misclassified_cellcount_predicted, CellCount_test_predicted, Prob_test_predicted) = test(OutputPred=True)
    torch.cuda.empty_cache()

elif args.model2 in ['RF', 'GBT']:
    # extract features
    TrainFeatures = ExtractFeatures(train_loader_featureExtraction)
    TestFeatures = ExtractFeatures(test_loader_featureExtraction)
    # use numerical features?
    if args.useNumericalFeatures:
        TrainNumFeatures = np.concatenate((EmbeddingFeatures(Blur_train), EmbeddingFeatures(Stain_train)), axis=1)
        TrainFeatures = np.concatenate((TrainFeatures, TrainNumFeatures), axis=1)
        TestNumFeatures = np.concatenate((EmbeddingFeatures(Blur_test), EmbeddingFeatures(Stain_test)), axis=1)
        TestFeatures = np.concatenate((TestFeatures, TestNumFeatures), axis=1)
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
            clf = RandomForestClassifier(n_estimators=n_estimator, criterion='gini', max_depth=max_depth, max_features = max_features, n_jobs=multiprocessing.cpu_count())
            clf.fit(TrainFeatures, Labels_train_m2)
            # prediction
            Labels_test_predicted = clf.predict(TestFeatures)
            CellCount_test_predicted = unique_cell_count[Labels_test_predicted.astype(np.int)]
        else:
            # train random forest
            regr = RandomForestRegressor(n_estimators=n_estimator, criterion='mse', max_depth=max_depth, max_features = max_features, n_jobs=multiprocessing.cpu_count())
            regr.fit(TrainFeatures, Labels_train_m2)
            # prediction
            CellCount_test_predicted = regr.predict(TestFeatures)
            CellCount_test_predicted = (np.around(CellCount_test_predicted)).astype(np.int)
            indx1 = np.where(CellCount_test_predicted<1)[0]
            CellCount_test_predicted[indx1]=1
            indx100 = np.where(CellCount_test_predicted>100)[0]
            CellCount_test_predicted[indx100]=100
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
            dtrain = xgb.DMatrix(TrainFeatures, label=Labels_train_m2)
            bst = xgb.train(param, dtrain, num_round)
            # prediction
            ## testing
            dtest = xgb.DMatrix(TestFeatures)
            Labels_test_predicted = bst.predict(dtest)
            CellCount_test_predicted = unique_cell_count[Labels_test_predicted.astype(np.int)]
        else:
            # train GBT
            # reg:linear or count:poisson
            if args.GBT_loss == "MSE":
                gbt_loss = "reg:linear"
            else:
                gbt_loss = "count:poisson"
            param = {'booster': booster, 'max_depth': max_depth, 'eta': eta, 'objective': gbt_loss,
                     'subsample': subsample, 'nthread': nthread}
            dtrain = xgb.DMatrix(TrainFeatures, label=Labels_train_m2)
            bst = xgb.train(param, dtrain, num_round)
            # prediction
            ## testing
            dtest = xgb.DMatrix(TestFeatures)
            CellCount_test_predicted = bst.predict(dtest)
            CellCount_test_predicted = (np.around(CellCount_test_predicted)).astype(np.int)
            indx1 = np.where(CellCount_test_predicted < 1)[0]
            CellCount_test_predicted[indx1] = 1
            indx100 = np.where(CellCount_test_predicted > 100)[0]
            CellCount_test_predicted[indx100] = 100

#---------------------------------------------------------------------------------------
# ensemble prediction!!
if args.EnsemblePred:
    assert args.model1 not in ['LReg', 'PReg', 'QReg'];
    CellCount_test_true_ensemble = []
    CellCount_test_predicted_ensemble = []
    for tmp_stain in [1,2]:
        tmp_flag = 0
        for tmp_blur in [1,23,48]:
            test_stain_blur_indx = np.logical_and(Stain_test==tmp_stain, Blur_test==tmp_blur)
            Intensity_test_stain_blur = Intensity_test[test_stain_blur_indx]
            CellCount_test_stain_blur = CellCount_test[test_stain_blur_indx]
            CellCount_test_Pred_CNN = CellCount_test_predicted[test_stain_blur_indx] #prediction from CNN (or RF)
            if tmp_stain == 1:
                regr = lreg_models[tmp_flag]
                CellCount_test_Pred_LQReg = (regr.predict(Intensity_test_stain_blur)).reshape(-1) #prediction from Linear/Quadratic regression
                regr = lreg_ub_models[tmp_flag]
                CellCount_test_UpperBound = (regr.predict(Intensity_test_stain_blur)).reshape(-1) #upper bound
                regr = lreg_lb_models[tmp_flag]
                CellCount_test_LowerBound = (regr.predict(Intensity_test_stain_blur)).reshape(-1) #lower bound
            else:
                regr = qreg_models[tmp_flag]
                CellCount_test_Pred_LQReg = (regr.predict(np.concatenate((Intensity_test_stain_blur, Intensity_test_stain_blur**2),axis=1))).reshape(-1)
                regr = qreg_ub_models[tmp_flag]
                CellCount_test_UpperBound = (regr.predict(np.concatenate((Intensity_test_stain_blur, Intensity_test_stain_blur**2),axis=1))).reshape(-1)
                regr = qreg_lb_models[tmp_flag]
                CellCount_test_LowerBound = (regr.predict(np.concatenate((Intensity_test_stain_blur, Intensity_test_stain_blur**2),axis=1))).reshape(-1)
            indx1 = np.where(CellCount_test_Pred_LQReg < 1)[0]
            CellCount_test_Pred_LQReg[indx1] = 1
            indx100 = np.where(CellCount_test_Pred_LQReg > 100)[0]
            CellCount_test_Pred_LQReg[indx100] = 100
            tmp_flag+=1

            #if prediction from CNN is in the inverval, then keep it; otherwise use prediction from LQReg
            indx_OutInterval = np.logical_or(CellCount_test_Pred_CNN<CellCount_test_LowerBound, CellCount_test_Pred_CNN>CellCount_test_UpperBound)
            CellCount_test_Pred_CNN[indx_OutInterval] = CellCount_test_Pred_LQReg[indx_OutInterval]

            CellCount_test_true_ensemble.extend(list(CellCount_test_stain_blur))
            CellCount_test_predicted_ensemble.extend(list(CellCount_test_Pred_CNN))
    #end for tmp_stain and tmp_blur
    CellCount_test_true = np.array(CellCount_test_true_ensemble)
    CellCount_test_predicted = np.array(CellCount_test_predicted_ensemble)


################################################################
# misclassified validation images
if args.model2 in ['RF', 'GBT'] and not args.EnsemblePred:
    indx_misclassified = np.where(CellCount_test_predicted!=CellCount_test)[0]
    misclassified_cellcount_predicted = CellCount_test_predicted[indx_misclassified]
    CellCount_test_true = CellCount_test
elif args.model2 in ['CNN'] and not args.EnsemblePred:
    CellCount_test_true = CellCount_test

test_acc = (CellCount_test_predicted.reshape(-1) == CellCount_test_true.reshape(-1)).mean()
test_rmse = np.sqrt(((CellCount_test_predicted.reshape(-1).astype(np.float) - CellCount_test_true.reshape(-1).astype(np.float)) ** 2).mean())
test_mae = np.mean(np.absolute(CellCount_test_predicted.reshape(-1).astype(np.float)-CellCount_test_true.reshape(-1).astype(np.float)))

print("Test acc: %.4f; Test RMSE: %.4f; Test MAE: %.4f" % (test_acc, test_rmse, test_mae))


#for bar plot
test_err = CellCount_test_predicted.reshape(-1).astype(float)-CellCount_test_true.reshape(-1).astype(float)
if args.EnsemblePred:
    filename_fig = "Exp1_Test_hist_" + args.model1 + "(" + args.PredType1 + ")+" + args.model2 + "(" + args.PredType2 + ")_EnsemblePred"
else:
    filename_fig = "Exp1_Test_hist_" + args.model1 + "(" + args.PredType1 + ")+" + args.model2 + "(" + args.PredType2 + ")"
np.save(filename_fig, test_err)


# # histogram of absolute valid errors
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.pyplot import figure, show
# test_err = CellCount_test_predicted.astype(float)-CellCount_test_true.astype(float)
# fig = figure(figsize=(6,6))
# ax = sns.countplot(x=test_err, color="#1f77b4")
# def change_width(ax, new_value) :
#     for patch in ax.patches :
#         current_width = patch.get_width()
#         diff = current_width - new_value
#         # we change the bar width
#         patch.set_width(new_value)
#         # we recenter the bar
#         patch.set_x(patch.get_x() + diff * .5)
# # If the bar width is too wide, uncomment the next line
# # change_width(ax, .2)
# plt.xlabel('Prediction error')
# fig = ax.get_figure()
# if args.EnsemblePred:
#     filename_fig = "Exp1_Test_hist_" + args.model1 + "(" + args.PredType1 + ")+" + args.model2 + "(" + args.PredType2 + ")_EnsemblePred"
# else:
#     filename_fig = "Exp1_Test_hist_" + args.model1 + "(" + args.PredType1 + ")+" + args.model2 + "(" + args.PredType2 + ")"
# fig.savefig(filename_fig+".png", dpi=300)
# np.save(filename_fig, test_err)
