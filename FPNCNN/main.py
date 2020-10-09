import os
import h5py
import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
import numpy as np
import random
import timeit
from collections import OrderedDict


### import my stuffs ###
from opts import prepare_options
from utils import *
from models.fpn_mask import FPN
from models.vgg_count import Counter
from train_fpn_mask import train_fpn_mask
from train_vgg_count import train_vgg, test_vgg

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

save_traincurves_folder = args.root + '/output/training_loss_fig'
os.makedirs(save_traincurves_folder, exist_ok=True)

#######################################################################################
'''                                  Load Data                                      '''
#######################################################################################
### vgg dataset ###
# hf = h5py.File(args.path_vgg_dataset, 'r')
# vgg_images = hf['IMGs_grey'][:]
# vgg_dot_annots = hf['IMGs_dots_grey'][:]
# hf.close()

### Tiny-BBBC005 dataset ###
hf = h5py.File(args.path_tinybbbc005, 'r')
tiny3bc005_train_images = hf['IMGs_train'][:]
tiny3bc005_train_masks = hf['MASKs_train'][:]
tiny3bc005_train_counts = hf["CellCount_train"][:]
tiny3bc005_test_images = hf['IMGs_test'][:]
tiny3bc005_test_masks = hf['MASKs_test'][:]
tiny3bc005_test_counts = hf["CellCount_test"][:]
max_count = np.max(tiny3bc005_train_counts)
max_mask = np.max(tiny3bc005_train_masks)
hf.close()

### Delete some cell counts in the training set ###
if args.deleted_counts != 'None':
    deleted_counts = args.deleted_counts
    deleted_counts = [int(count) for count in deleted_counts.split("_")]
    indx_all = set(list(np.arange(len(tiny3bc005_train_images))))
    indx_deleted = []
    for i in range(len(deleted_counts)):
        indx_i = np.where(tiny3bc005_train_counts == deleted_counts[i])[0]
        indx_deleted.extend(list(indx_i))
    print("\n Delete {} training samples for counts: ".format(len(indx_deleted)), deleted_counts)
    indx_deleted = set(indx_deleted)
    indx_left = indx_all.difference(indx_deleted)
    indx_left = np.array(list(indx_left))
    print("\n {} training samples are left.".format(len(indx_left)))
    tiny3bc005_train_images = tiny3bc005_train_images[indx_left]
    tiny3bc005_train_counts = tiny3bc005_train_counts[indx_left]
    tiny3bc005_train_masks = tiny3bc005_train_masks[indx_left]

### normalize cell counts ###
tiny3bc005_train_counts = tiny3bc005_train_counts / max_count
### normalize masks to [0,1] ###
# tiny3bc005_train_masks[tiny3bc005_train_masks>0] = 1
# tiny3bc005_test_masks[tiny3bc005_test_masks>0] = 1
tiny3bc005_train_masks = tiny3bc005_train_masks / 255.0
tiny3bc005_test_masks = tiny3bc005_test_masks / 255.0

# vgg_dataset_unet1 = IMGs_dataset(images=vgg_images, dot_annots=vgg_dot_annots, masks=None, counts=None, normalize=True, transform=args.unet1_transform)
# vgg_dataloader_unet1 = torch.utils.data.DataLoader(vgg_dataset_unet1, batch_size = args.unet1_batch_size_train, shuffle=True, num_workers=args.num_workers)

tiny3bc005_dataset_mask = IMGs_dataset(images=tiny3bc005_train_images,
                                       dot_annots=None,
                                       masks=tiny3bc005_train_masks,
                                       counts=tiny3bc005_train_counts,
                                       normalize=True,
                                       transform=args.fpn_transform)
tiny3bc005_dataloader_mask = torch.utils.data.DataLoader(tiny3bc005_dataset_mask,
                                                         batch_size=args.fpn_batch_size_train,
                                                         shuffle=True,
                                                         num_workers=args.num_workers)

tiny3bc005_train_dataset = IMGs_dataset(images=tiny3bc005_train_images,
                                        dot_annots=None,
                                        masks=None,
                                        counts=tiny3bc005_train_counts,
                                        normalize=True,
                                        transform=args.cnn_transform)
tiny3bc005_train_dataloader = torch.utils.data.DataLoader(tiny3bc005_train_dataset,
                                                          batch_size=args.cnn_batch_size_train,
                                                          shuffle=True,
                                                          num_workers=args.num_workers)

tiny3bc005_test_dataset = IMGs_dataset(images=tiny3bc005_test_images,
                                       dot_annots=None,
                                       masks=None,
                                       counts=tiny3bc005_test_counts,
                                       normalize=True,
                                       transform=False)
tiny3bc005_test_dataloader = torch.utils.data.DataLoader(tiny3bc005_test_dataset,
                                                         batch_size=args.cnn_batch_size_test,
                                                         shuffle=False,
                                                         num_workers=args.num_workers)

#######################################################################################
'''                       FPN Training                                   '''
#######################################################################################
print("\n -----------------------------------------------------------------------------------------")
print("\n Start training FPN >>>")

fpn_ckpt_fullpath = save_models_folder + '/ckpt_FPN_epochs_{}_transform_{}_seed_{}.pth'.format(
    args.fpn_epochs, args.fpn_transform, args.seed)
print('\n' + fpn_ckpt_fullpath)

path_to_ckpt_in_train = save_models_folder + '/ckpts_in_train_FPN_transform_{}_seed_{}'.format(
    args.fpn_transform, args.seed)
os.makedirs(path_to_ckpt_in_train, exist_ok=True)

path_to_images_in_train = save_images_folder + '/images_in_train_FPN_transform_{}_seed_{}'.format(
    args.fpn_transform, args.seed)
os.makedirs(path_to_images_in_train, exist_ok=True)

### train the FPN ###
# if not os.path.isfile(fpn_ckpt_fullpath):
#     start = timeit.default_timer()
#     print("\n Begin training FPN:")

#     ## randomly choose some tiny-bbbc005 images to test during training
#     indx_test = np.random.choice(np.arange(len(tiny3bc005_test_images)), size=16)
#     test_images = tiny3bc005_test_images[indx_test]
#     test_masks = tiny3bc005_test_masks[indx_test]

#     _, _, h, w = test_images.shape
#     # fpn = FPN(h, w).cuda()
#     fpn = FPN(h, w).cuda()
#     fpn = nn.DataParallel(fpn)
#     fpn = train_fpn_mask(trainloader=tiny3bc005_dataloader_mask,
#                          test_images=test_images,
#                          test_masks=test_masks,
#                          fpn=fpn,
#                          save_images_folder=path_to_images_in_train,
#                          path_to_ckpt=path_to_ckpt_in_train)

#     # store model
#     torch.save({
#         'net_state_dict': fpn.state_dict(),
#     }, fpn_ckpt_fullpath)

#     stop = timeit.default_timer()
#     print("FPN mask training finished! Time elapses: {}s".format(stop - start))
# else:
#     print("\n Load pre-trained FPN mask:")
#     checkpoint = torch.load(fpn_ckpt_fullpath)
#     ## randomly choose some tiny-bbbc005 images to test during training
#     indx_test = np.random.choice(np.arange(len(tiny3bc005_test_images)), size=16)
#     test_images = tiny3bc005_test_images[indx_test]
#     test_masks = tiny3bc005_test_masks[indx_test]

#     _, _, h, w = test_images.shape()
#     fpn = FPN(h, w).cuda()
#     fpn = nn.DataParallel(fpn)
#     fpn.load_state_dict(checkpoint['net_state_dict'])
# end if

#######################################################################################
'''                                  VGG Training                                   '''
#######################################################################################
print("\n -----------------------------------------------------------------------------------------")
print("\n Start training {} in {}  >>>".format(args.cnn_name, args.experiment_name))

cnn_ckpt_fullpath = save_models_folder + '/ckpt_{}_{}_epochs_{}_transform_{}_seed_{}.pth'.format(
    args.experiment_name, args.cnn_name, args.cnn_epochs, args.cnn_transform, args.seed)
print('\n' + cnn_ckpt_fullpath)

path_to_ckpt_in_train = save_models_folder + '/ckpts_in_train_{}_{}_transform_{}_seed_{}'.format(
    args.experiment_name, args.cnn_name, args.cnn_transform, args.seed)
os.makedirs(path_to_ckpt_in_train, exist_ok=True)

### train the vgg counting network ###
if not os.path.isfile(cnn_ckpt_fullpath):
    start = timeit.default_timer()
    print("\n Begin Training {}:".format(args.cnn_name))

    # models
    # load pretrained fpn
    indx_test = np.random.choice(np.arange(len(tiny3bc005_test_images)), size=16)
    test_images = tiny3bc005_test_images[indx_test]
    test_masks = tiny3bc005_test_masks[indx_test]

    _, _, h, w = test_images.shape
    fpn = FPN(h, w).cuda()
    fpn = nn.DataParallel(fpn)
    fpn_ckpt_fullpath = save_models_folder + '/ckpt_FPN_epochs_50_transform_False_seed_2020.pth'
    checkpoint = torch.load(fpn_ckpt_fullpath)
    fpn.load_state_dict(checkpoint['net_state_dict'])

    count = Counter(h // 2, w // 2, model_class=args.cnn_name).cuda()

    count = nn.DataParallel(count)
    cnn_net = train_vgg(trainloader=tiny3bc005_train_dataloader,
                        testloader=tiny3bc005_test_dataloader,
                        max_count=max_count,
                        fpn=fpn,
                        net=count,
                        path_to_ckpt=path_to_ckpt_in_train)

    # store model
    torch.save({
        'net_state_dict': cnn_net.state_dict(),
    }, cnn_ckpt_fullpath)

    stop = timeit.default_timer()
    print("{} training finished! Time elapses: {}s".format(args.cnn_name, stop - start))
else:
    print("\n Load pre-trained {}:".format(args.cnn_name))
    checkpoint = torch.load(cnn_ckpt_fullpath)
    # load pretrained fpn
    indx_test = np.random.choice(np.arange(len(tiny3bc005_test_images)), size=16)
    test_images = tiny3bc005_test_images[indx_test]
    test_masks = tiny3bc005_test_masks[indx_test]

    _, _, h, w = test_images.shape
    fpn = FPN(h, w).cuda()
    fpn = nn.DataParallel(fpn)
    checkpoint = torch.load(fpn_ckpt_fullpath)
    fpn.load_state_dict(checkpoint['net_state_dict'])
    count = Counter(h // 2, w // 2, model_class=arg.cnn_name).cuda()
    count = nn.DataParallel(count)
    model = nn.Sequential(OrderedDict([('fpn', fpn), ('counter', count)]))
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['net_state_dict'])
# end if

### test the vgg counting network ###
_, _ = test_vgg(testloader=tiny3bc005_test_dataloader,
                max_count=max_count,
                model=cnn_net,
                verbose=True)
