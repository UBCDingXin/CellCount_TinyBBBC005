"""
Randomly delete 5 classes from the training set
5 classes are determined before running this script

"""


import os
# wd = 'your path/SSC_CountCells'
wd = '/home/xin/Working directory/Counting_Cells/SSC_CountCells'
# wd = '/media/qiong/icecream/SSC_CountCells'
#wd = 'C:/Users/xin/Desktop/Github/SSC_CountCells'
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
                    help='Prediction type for the first model (default: "Cl"); Candidates: Cl, Reg')
parser.add_argument('--PredType2', type=str, default='Cl',
                    help='Prediction type for the second model (default: "Cl"); Candidates: Cl, Reg')
parser.add_argument('--EnsemblePred', action='store_true', default=False,
                    help='Ensemble ResNet34 and QReg?')

parser.add_argument('--RF_NTree', type=int, default=500, metavar='N',
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

parser.add_argument('--NROUND', type=int, default=3)
parser.add_argument('--dataAugment', action='store_true', default=False,
                    help='Do data augmentation?')
parser.add_argument('--nfake', type=int, default=20) #25*5/6~= 20
parser.add_argument('--DA_flip', action='store_true', default=False,
                    help='Do flipping in data augmentation?')
parser.add_argument('--one_formula_per_class', action='store_true', default=False,
                    help='One formula for each cell count?')
parser.add_argument('--knowMC', action='store_true', default=False,
                    help='know which are classes are missing? Otherwise, know a range of missing classes')
parser.add_argument('--DA_not_filter', action='store_false', default=True,
                    help='Do filtering in DA? default is True')

args = parser.parse_args()
# cuda
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
ngpu = torch.cuda.device_count() #number of gpus
args.base_lr = args.base_lr*ngpu


# data augmentation
nfake = args.nfake
formula_dir = wd + '/data/Exp3_Formulae/'

# random seed
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

# directories for checkpoint and images
save_models_folder = wd + '/Output/saved_models/'
if not os.path.exists(save_models_folder):
    os.makedirs(save_models_folder)

save_images_folder = wd + '/Output/saved_images/'
if not os.path.exists(save_images_folder):
    os.makedirs(save_images_folder)

#############################
# Data Loader
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

IMGs_train = np.concatenate((IMGs_train, IMGs_test), axis=0)
Blur_train = np.concatenate((Blur_train, Blur_test))
Stain_train = np.concatenate((Stain_train, Stain_test))
CellCount_train = np.concatenate((CellCount_train, CellCount_test))

del IMGs_test; gc.collect()

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
    # adjust_learning_rate_v2(optimizer, epoch, args.base_lr)
    for batch_idx, (images, labels, _, _) in enumerate(train_loader):
        images = images.type(torch.float).to(device)
        if args.PredType1 in ['Cl', 'Mixed']:
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


def test(dataloader, mode='Valid', OutputPred=False, verbose=True):
    # mode: 'Test' or 'Valid'
    # OutputPred: output indecices and predicted cell count of misclassified samples, predicted cell count for all test images and predicted probabilties
    model.eval()
    N_samp = dataloader.dataset.n_images
    batch_size = dataloader.batch_size
    CalcProb = nn.Softmax(dim=1)  # softmax function which is used for  computing predicted probabilties
    # test_loss = 0
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
            if args.PredType1 in ['Cl', 'Mixed']:
                labels = labels.type(torch.long).to(device)
            else:
                labels = labels.view(-1,1)
                labels = labels.type(torch.float).to(device)
            outputs, _ = model(images)
            # loss = criterion(outputs, labels)
            # test_loss += loss.item()

            if args.PredType1 in ['Cl', 'Mixed']: #classification
                # predictions
                _, predicted = torch.max(outputs.data, 1)
                ## compute predicted cell count
                cellcount_predicted = torch.from_numpy(unique_cell_count_train[predicted.cpu().numpy()]).type(torch.float)
                ## compute test rmse
                cellcount_truth = torch.from_numpy(unique_cell_count_valid[labels.cpu().numpy()]).type(torch.float)
                test_rmse += MSE_loss(cellcount_predicted, cellcount_truth).item() * len(images)
                ## compute test accuracy
                # test_acc += (predicted == labels).type(torch.float).sum().item()
                test_acc += (cellcount_truth == cellcount_predicted).type(torch.float).sum().item()
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
    # test_loss /= len(dataloader.dataset)
    test_acc /= len(dataloader.dataset)
    test_rmse /= len(dataloader.dataset);
    test_rmse = math.sqrt(test_rmse)
    if verbose:
        # print('====> %s set loss: %.4f; %s set accuracy: %.4f; %s RMSE %.4f' % (
        # mode, test_loss, mode, test_acc, mode, test_rmse))
        print('====> %s set accuracy: %.4f; %s RMSE %.4f' % (mode, test_acc, mode, test_rmse))

    del images, labels, outputs;
    gc.collect()
    torch.cuda.empty_cache()

    if OutputPred:  # output indecices and predicted cell count of misclassified samples, predicted cell count for all test images and predicted probabilties
        CellCount_test_predicted = (np.array(CellCount_test_predicted)).astype(np.int)
        CellCount_test_truth = (np.array(CellCount_test_truth)).astype(np.int)
        # return (test_loss, test_acc, test_rmse, np.array(indx_misclassified), CellCount_test_predicted[indx_misclassified], CellCount_test_truth[indx_misclassified], CellCount_test_predicted, Prob_test_predicted)
        return (test_acc, test_rmse, np.array(indx_misclassified),
                CellCount_test_predicted[indx_misclassified], CellCount_test_truth[indx_misclassified],
                CellCount_test_predicted, Prob_test_predicted)
    else:
        # return (test_loss, test_acc, test_rmse)
        return (test_acc, test_rmse)


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

valid_rmse_all = np.zeros((args.NROUND,1))  # valid RMSE
valid_acc_all = np.zeros((args.NROUND,1))  # valid accuracy
valid_mae_all = np.zeros((args.NROUND,1))  # valid MAE
valid_rmse_duringTrain = np.zeros((args.NROUND, args.epochs))
valid_acc_duringTrain = np.zeros((args.NROUND, args.epochs))

deleted_classes = np.array([[14, 35, 57, 66, 83],
                            [10, 31, 70, 83, 91],
                            [18, 27, 44, 53, 91]])

CellCount_valid_true_all = []
CellCount_valid_predicted_all = []

start = timeit.default_timer()
for nround in range(args.NROUND):
    np.random.seed(nround + args.seed)

    #-----------------------------------------------------------------------------
    # divide training data into 2 parts: train (3000) and valid (600); ratio 5:1
    # split within each combination
    train_indices = []
    valid_indices = []
    for stain_tmp in [1,2]:
        for blur_tmp in [1,23,48]:
            indx_all_comb = np.where((Stain_train==stain_tmp)*(Blur_train==blur_tmp)==True)[0]
            np.random.shuffle(indx_all_comb)
            train_indices_comb = indx_all_comb[0:500]
            valid_indices_comb = np.array([x for x in indx_all_comb if x not in train_indices_comb])
            train_indices.extend(list(train_indices_comb))
            valid_indices.extend(list(valid_indices_comb))
    train_indices = np.array(train_indices)
    valid_indices = np.array(valid_indices)

    # indx_all = np.arange(len(IMGs_train))
    # np.random.shuffle(indx_all)
    # train_indices = indx_all[0:3000]
    # valid_indices = np.array([x for x in indx_all if x not in train_indices])

    # determine training images
    train_images = IMGs_train[train_indices]
    train_blur = Blur_train[train_indices]
    train_stain = Stain_train[train_indices]
    train_cellcount = CellCount_train[train_indices]

    # remove some classes from the training set
    # decide which classes to remove from the training set
    unique_cell_count = np.array(list(set(train_cellcount)))
    removed_cell_count = deleted_classes[nround]
    indx_train_removed = np.array([], dtype=np.int)
    for i in range(len(removed_cell_count)):
        indx_train_removed = np.concatenate((indx_train_removed, np.where(train_cellcount==removed_cell_count[i])[0]))

    indx_train_all = np.arange(len(train_images))
    indx_train_left = np.array(list(set(indx_train_all).difference(set(indx_train_removed))))
    train_images = train_images[indx_train_left]
    train_blur = train_blur[indx_train_left]
    train_stain = train_stain[indx_train_left]
    train_cellcount = train_cellcount[indx_train_left]

    # filename1 = wd+"/Exp3_round"+str(nround)+"_train_intensity_vs_cellcount_beforeAugment.png"
    # DA_intensity_analysis(train_images, train_cellcount, train_stain, train_blur, filename1)

    # -------------------------------------------------------------------------
    # If ensemble ResNet34 and QReg, train three QReg for each combination at here!
    if args.EnsemblePred:
        unique_cell_count_beforeDA = np.array(list(set(train_cellcount)))
        train_intensity_beforeDA = (np.mean(train_images, axis=(1,2,3))).reshape(-1,1)

        # !!!!!!!!!!!!!!!!!!!!!!!!!
        # Note that upper bound is based on intensity_min; lower bound is based on intensity_max
        qreg_models = []; lreg_models = []
        qreg_ub_models = []; lreg_ub_models = []
        qreg_lb_models = []; lreg_lb_models = []
        for tmp_stain in [1,2]:
            for tmp_blur in [1,23,48]:
                intensity_mins = []
                intensity_maxs = []
                for i in range(len(unique_cell_count_beforeDA)):
                    current_cellcount = unique_cell_count_beforeDA[i]
                    indx_stain_blur_count = np.where(((train_blur==tmp_blur) * (train_stain==tmp_stain) * (train_cellcount==current_cellcount))==True)[0]
                    intensity_mins.append(np.min(train_intensity_beforeDA[indx_stain_blur_count]))
                    intensity_maxs.append(np.max(train_intensity_beforeDA[indx_stain_blur_count]))
                y = unique_cell_count_beforeDA.reshape(-1,1)
                x_min = np.array(intensity_mins).reshape(-1,1)
                x_max = np.array(intensity_maxs).reshape(-1,1)
                x = train_intensity_beforeDA[(train_blur==tmp_blur) * (train_stain==tmp_stain)]
                if tmp_stain == 1:
                    regr_min = linear_model.LinearRegression()
                    regr_min.fit(x_min, y)
                    regr_max = linear_model.LinearRegression()
                    regr_max.fit(x_max, y)
                    regr = linear_model.LinearRegression()
                    regr.fit(x, train_cellcount[(train_blur==tmp_blur) * (train_stain==tmp_stain)])
                    lreg_ub_models.append(regr_min)
                    lreg_lb_models.append(regr_max)
                    lreg_models.append(regr)
                else:
                    regr_min = linear_model.LinearRegression()
                    regr_min.fit(np.concatenate((x_min,x_min**2),axis=1), y)
                    regr_max = linear_model.LinearRegression()
                    regr_max.fit(np.concatenate((x_max,x_max**2),axis=1), y)
                    regr = linear_model.LinearRegression()
                    regr.fit(np.concatenate((x,x**2),axis=1),train_cellcount[(train_blur==tmp_blur) * (train_stain==tmp_stain)])
                    qreg_ub_models.append(regr_min)
                    qreg_lb_models.append(regr_max)
                    qreg_models.append(regr)
    #end if args.EnsemblePred:


    # -------------------------------------------------------------------------
    # data augmentation
    if args.dataAugment:
        formula_dir_current = formula_dir + 'Round' + str(nround+1) + '/'
        if args.knowMC:
            AugClass = list(deleted_classes[nround])
        else:
            AugClass = list(deleted_classes[nround]) #todo
        train_images, train_cellcount, train_blur, train_stain = AugmentData(train_images, train_cellcount, train_blur, train_stain, AugClass, formula_dir_current, nfake=nfake, flipping = args.DA_flip, one_formula_per_class=args.one_formula_per_class, show_sample_img=False, dump_fake=False, fakeImg_dir=None, do_filter = args.DA_not_filter, verbose=False)

    # determine validation images
    valid_images = IMGs_train[valid_indices]
    valid_blur = Blur_train[valid_indices]
    valid_stain = Stain_train[valid_indices]
    valid_cellcount = CellCount_train[valid_indices]

    # filename2 = wd+"/Exp3_round"+str(nround)+"_valid_intensity_vs_cellcount.png"
    # DA_intensity_analysis(valid_images, valid_cellcount, valid_stain, valid_blur, filename2)
    #
    # filename3 = wd+"/Exp3_round"+str(nround)+"_train_intensity_vs_cellcount_afterAugment_knownMC_"+str(args.knowMC)+".png"
    # DA_intensity_analysis(train_images, train_cellcount, train_stain, train_blur, filename3)

    #training and validation intensity
    train_intensity = (np.mean(train_images, axis=(1,2,3))).reshape(-1,1)
    valid_intensity = (np.mean(valid_images, axis=(1,2,3))).reshape(-1,1)

    #determine the loss function of CNNs; if num_classes>1, cross-entropy; else MSE
    if args.PredType1 == 'Cl':
        num_classes = len(list(set(train_cellcount)))
        criterion = CE_loss
    else:
        num_classes = 1
        criterion = MSE_loss

    #-----------------------------------------------------------------------------
    # data loaders
    # if Classification, then convert cell count into a categorical variable with 24 levels from 0 to 23
    train_labels = np.zeros(train_cellcount.shape)
    valid_labels = np.zeros(valid_cellcount.shape)
    unique_cell_count_train = np.sort(np.array(list(set(train_cellcount))))
    unique_cell_count_valid = np.sort(np.array(list(set(valid_cellcount))))

    ### the prediction type of model 1
    if args.PredType1 == 'Cl': # model 1 is a classification method
        train_labels_m1 = np.zeros(train_cellcount.shape).astype(np.int)
        valid_labels_m1 = np.zeros(valid_cellcount.shape).astype(np.int)
        for i in range(len(unique_cell_count_train)):
            indx_train = np.where(train_cellcount==unique_cell_count_train[i])[0]
            train_labels_m1[indx_train] = i
        for i in range(len(unique_cell_count_valid)):
            indx_valid = np.where(valid_cellcount==unique_cell_count_valid[i])[0]
            valid_labels_m1[indx_valid] = i
    else:
        train_labels_m1 = train_cellcount
        valid_labels_m1 = valid_cellcount
    ### the prediction type of model 2
    if args.PredType2 == 'Cl':
        assert args.PredType1 == 'Cl'
        train_labels_m2 = np.zeros(train_cellcount.shape).astype(np.int)
        valid_labels_m2 = np.zeros(valid_cellcount.shape).astype(np.int)
        for i in range(len(unique_cell_count_train)):
            indx_train = np.where(train_cellcount==unique_cell_count_train[i])[0]
            train_labels_m2[indx_train] = i
        for i in range(len(unique_cell_count_valid)):
            indx_valid = np.where(valid_cellcount==unique_cell_count_valid[i])[0]
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

    #---------------------------------------------------------------------------------------
    # Train model
    # first model
    if args.model1 not in ['LReg', 'PReg', 'QReg']:
        save_filename = save_models_folder + 'ckpt_Exp3_CV_' + args.model1 + '(' + args.PredType1 + ')_epoch' + str(args.epochs) + '_seed' + str(args.seed) + '_Round' + str(nround) + '_CNNFlip_' + str(args.model1_flip) + '_DA_' + str(args.dataAugment) + '_OneForOne_' + str(args.one_formula_per_class) + '_knowMC_' + str(args.knowMC) + '.pth'
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
                (valid_acc, valid_rmse) = test(valid_loader, mode='Valid', verbose=False)
                valid_rmse_duringTrain[nround, epoch] = valid_rmse
                valid_acc_duringTrain[nround, epoch] = valid_acc
                print("Round [%d/%d], Epoch [%d/%d]: train loss %.4f, valid acc %.4f, valid rmse %.4f" % (nround + 1, args.NROUND, epoch + 1, args.epochs, train_loss, valid_acc, valid_rmse))
                # save model
                if epoch + 1 == args.epochs:
                    # save trained model
                    save_filename = save_models_folder + 'ckpt_Exp3_CV_' + args.model1 + '(' + args.PredType1 + ')_epoch' + str(args.epochs) + '_seed' + str(args.seed) + '_Round' + str(nround) + '_CNNFlip_' + str(args.model1_flip) + '_DA_' + str(args.dataAugment) + '_OneForOne_' + str(args.one_formula_per_class) + '_knowMC_' + str(args.knowMC) + '.pth'
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
    elif args.model1 == "QReg": # quadratic regression
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
                    Intensity_train_stain_blur = np.concatenate((train_intensity[stain_blur_indx],train_intensity[stain_blur_indx]**2),axis=1)
                CellCount_train_stain_blur = (train_cellcount[stain_blur_indx]).reshape(-1,1)
                regr = linear_model.LinearRegression()
                regr.fit(Intensity_train_stain_blur, CellCount_train_stain_blur)
                # validation
                stain_blur_indx = np.logical_and(valid_stain==tmp_stain, valid_blur==tmp_blur)
                if tmp_stain == 1:
                    Intensity_valid_stain_blur = valid_intensity[stain_blur_indx]
                else:
                    Intensity_valid_stain_blur = np.concatenate((valid_intensity[stain_blur_indx],valid_intensity[stain_blur_indx]**2),axis=1)
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
        (valid_acc_all[nround], valid_rmse_all[nround], indx_misclassified, misclassified_cellcount_predicted, misclassified_cellcount_truth, CellCount_valid_predicted, _) = test(valid_loader, mode='Valid', OutputPred=True, verbose=False)
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
                CellCount_valid_predicted = unique_cell_count_train[Labels_valid_predicted.astype(np.int)]
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
                CellCount_valid_predicted = unique_cell_count_train[Labels_valid_predicted.astype(np.int)]
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
    #end if args.model2


    #---------------------------------------------------------------------------------------
    # ensemble prediction!!
    if args.EnsemblePred:
        assert args.model1 not in ['LReg', 'PReg', 'QReg'];
        CellCount_valid_true_ensemble = []
        CellCount_valid_predicted_ensemble = []
        for tmp_stain in [1,2]:
            tmp_flag = 0
            for tmp_blur in [1,23,48]:
                stain_blur_indx = np.logical_and(valid_stain==tmp_stain, valid_blur==tmp_blur)
                Intensity_valid_stain_blur = valid_intensity[stain_blur_indx]
                CellCount_valid_stain_blur = valid_cellcount[stain_blur_indx]
                CellCount_valid_Pred_CNN = CellCount_valid_predicted[stain_blur_indx] #prediction from CNN (or RF)
                if tmp_stain == 1:
                    regr = lreg_models[tmp_flag]
                    CellCount_valid_Pred_LQReg = (regr.predict(Intensity_valid_stain_blur)).reshape(-1) #prediction from Linear/Quadratic regression
                    regr = lreg_ub_models[tmp_flag]
                    CellCount_UpperBound = (regr.predict(Intensity_valid_stain_blur)).reshape(-1) #upper bound
                    regr = lreg_lb_models[tmp_flag]
                    CellCount_LowerBound = (regr.predict(Intensity_valid_stain_blur)).reshape(-1) #lower bound
                else:
                    regr = qreg_models[tmp_flag]
                    CellCount_valid_Pred_LQReg = (regr.predict(np.concatenate((Intensity_valid_stain_blur, Intensity_valid_stain_blur**2),axis=1))).reshape(-1)
                    regr = qreg_ub_models[tmp_flag]
                    CellCount_UpperBound = (regr.predict(np.concatenate((Intensity_valid_stain_blur, Intensity_valid_stain_blur**2),axis=1))).reshape(-1)
                    regr = qreg_lb_models[tmp_flag]
                    CellCount_LowerBound = (regr.predict(np.concatenate((Intensity_valid_stain_blur, Intensity_valid_stain_blur**2),axis=1))).reshape(-1)
                indx1 = np.where(CellCount_valid_Pred_LQReg < 1)[0]
                CellCount_valid_Pred_LQReg[indx1] = 1
                indx100 = np.where(CellCount_valid_Pred_LQReg > 100)[0]
                CellCount_valid_Pred_LQReg[indx100] = 100
                tmp_flag+=1

                #if prediction from CNN is in the inverval, then keep it; otherwise use prediction from LQReg
                indx_OutInterval = np.logical_or(CellCount_valid_Pred_CNN<CellCount_LowerBound, CellCount_valid_Pred_CNN>CellCount_UpperBound)
                CellCount_valid_Pred_CNN[indx_OutInterval] = CellCount_valid_Pred_LQReg[indx_OutInterval]

                CellCount_valid_true_ensemble.extend(list(CellCount_valid_stain_blur))
                CellCount_valid_predicted_ensemble.extend(list(CellCount_valid_Pred_CNN))
        #end for tmp_stain and tmp_blur
        CellCount_valid_true = np.array(CellCount_valid_true_ensemble)
        CellCount_valid_predicted = np.array(CellCount_valid_predicted_ensemble)

    #---------------------------------------------------------------------------------------
    # rmse
    if args.model2 in ['RF', 'GBT', 'CNN'] and not args.EnsemblePred:
        CellCount_valid_true = valid_cellcount

    # compute valid/test accuracy, valid/test rmse for the subsequent model
    valid_acc_all[nround] = np.mean(CellCount_valid_predicted.reshape(-1) == CellCount_valid_true.reshape(-1))
    valid_rmse_all[nround] = np.sqrt(np.mean((CellCount_valid_predicted.reshape(-1).astype(np.float) - CellCount_valid_true.reshape(-1).astype(np.float)) ** 2))
    valid_mae_all[nround] = np.mean(np.absolute(CellCount_valid_predicted.reshape(-1).astype(np.float) - CellCount_valid_true.reshape(-1).astype(np.float)))

    print("Round [%d/%d]: valid acc %.4f, valid rmse %.4f, valid_mae %.4f" % (nround + 1, args.NROUND, valid_acc_all[nround], valid_rmse_all[nround], valid_mae_all[nround]))

    CellCount_valid_true_all.extend(CellCount_valid_true.tolist())
    CellCount_valid_predicted_all.extend(CellCount_valid_predicted.tolist())
# end for nround
end = timeit.default_timer()
print("Total time elapsed: %f" % (end - start))

print("valid rmse: Round1 %.4f, Round2 %.4f, Round3 %.4f" % (valid_rmse_all[0], valid_rmse_all[1], valid_rmse_all[2]))
print("valid mae: Round1 %.4f, Round2 %.4f, Round3 %.4f" % (valid_mae_all[0], valid_mae_all[1], valid_mae_all[2]))


# print averaged validation accuracy and rmse
print("Avg. valid acc %.4f (%.4f), Avg. valid rmse %.4f (%.4f), Avg. valid mae %.4f (%.4f)" % (valid_acc_all.mean(), valid_acc_all.std(), valid_rmse_all.mean(), valid_rmse_all.std(), valid_mae_all.mean(), valid_mae_all.std()))


#for bar plot
err_valid_all = np.array(CellCount_valid_true_all).reshape(-1).astype(float) - np.array(CellCount_valid_predicted_all).reshape(-1).astype(float)
if args.EnsemblePred:
    filename_fig = "Exp3_hist_" + args.model1 + "(" + args.PredType1 + ")+" + args.model2 + "(" + args.PredType2 + ")_EnsemblePred"
    if args.dataAugment:
        filename_fig = "Exp3_hist_" + args.model1 + "(" + args.PredType1 + ")+" + args.model2 + "(" + args.PredType2 + ")_DA_EnsemblePred"
else:
    filename_fig = "Exp3_hist_" + args.model1 + "(" + args.PredType1 + ")+" + args.model2 + "(" + args.PredType2 + ")"
    if args.dataAugment:
        filename_fig = "Exp3_hist_" + args.model1 + "(" + args.PredType1 + ")+" + args.model2 + "(" + args.PredType2 + ")_DA"
np.save(filename_fig, err_valid_all)

# # histogram of absolute valid errors
# import matplotlib.pyplot as plt
# err_valid_all = np.array(CellCount_valid_true_all).astype(float) - np.array(CellCount_valid_predicted_all).astype(float)
# num_bins = 10
# plt.hist(err_valid_all, bins=num_bins, density=False)
# plt.xlim(-18,18)
# plt.ylim(0,1800)
# plt.title("Histogram of prediction errors")
# plt.xlabel('Prediction error')
# plt.ylabel('Frequency')
# if args.EnsemblePred:
#     filename_fig = "Exp3_hist_" + args.model1 + "(" + args.PredType1 + ")+" + args.model2 + "(" + args.PredType2 + ")_EnsemblePred"
#     if args.dataAugment:
#         filename_fig = "Exp3_hist_" + args.model1 + "(" + args.PredType1 + ")+" + args.model2 + "(" + args.PredType2 + ")_DA_EnsemblePred"
# else:
#     filename_fig = "Exp3_hist_" + args.model1 + "(" + args.PredType1 + ")+" + args.model2 + "(" + args.PredType2 + ")"
#     if args.dataAugment:
#         filename_fig = "Exp3_hist_" + args.model1 + "(" + args.PredType1 + ")+" + args.model2 + "(" + args.PredType2 + ")_DA"
# plt.savefig(filename_fig+'.png', dpi=300)
# np.save(filename_fig, err_valid_all)
