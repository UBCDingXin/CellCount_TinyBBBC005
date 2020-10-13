import os
import h5py
import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
import numpy as np
import random
import timeit


### import my stuffs ###
from opts import prepare_options
from utils import *
from models import *
from train_unet import train_unet
from train_cnn import train_cnn, test_cnn


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
hf = h5py.File(args.path_vgg_dataset, 'r')
vgg_images = hf['IMGs_grey'][:]
vgg_dot_annots = hf['IMGs_dots_grey'][:]
hf.close()

### Tiny-BBBC005 dataset ###
hf = h5py.File(args.path_tinybbbc005, 'r')
tiny3bc005_train_images = hf['IMGs_train'][:]
tiny3bc005_train_counts = hf["CellCount_train"][:]
tiny3bc005_test_images = hf['IMGs_test'][:]
tiny3bc005_test_counts = hf["CellCount_test"][:]
# max_count = np.max(tiny3bc005_train_counts)
max_count = 1.0
hf.close()

### Delete some cell counts in the training set ###
if args.deleted_counts != 'None':
    deleted_counts = args.deleted_counts
    deleted_counts = [int(count) for count in deleted_counts.split("_")]
    indx_all = set(list(np.arange(len(tiny3bc005_train_images))))
    indx_deleted = []
    for i in range(len(deleted_counts)):
        indx_i = np.where(tiny3bc005_train_counts==deleted_counts[i])[0]
        indx_deleted.extend(list(indx_i))
    print("\n Delete {} training samples for counts: ".format(len(indx_deleted)), deleted_counts)
    indx_deleted = set(indx_deleted)
    indx_left = indx_all.difference(indx_deleted)
    indx_left = np.array(list(indx_left))
    print("\n {} training samples are left.".format(len(indx_left)))
    tiny3bc005_train_images = tiny3bc005_train_images[indx_left]
    tiny3bc005_train_counts = tiny3bc005_train_counts[indx_left]


### Delete extra training samples if needed
if args.num_train<len(tiny3bc005_train_images):
    indx_all = np.arange(len(tiny3bc005_train_images))
    indx_left = np.random.choice(indx_all, size=args.num_train, replace=False)
    tiny3bc005_train_images = tiny3bc005_train_images[indx_left]
    tiny3bc005_train_counts = tiny3bc005_train_counts[indx_left]


unique_cell_counts = list(set(tiny3bc005_train_counts))
print("\n There are {} unique cell counts in the training set.".format(len(unique_cell_counts)))

### normalize cell counts ###
tiny3bc005_train_counts = tiny3bc005_train_counts/max_count



### create data loader ###
vgg_dataset = IMGs_dataset(images=vgg_images, dot_annots=vgg_dot_annots, masks=None, counts=None, normalize=True, transform=args.unet_transform)
vgg_dataloader = torch.utils.data.DataLoader(vgg_dataset, batch_size = args.unet_batch_size_train, shuffle=True, num_workers=args.num_workers)

tiny3bc005_train_dataset = IMGs_dataset(images=tiny3bc005_train_images, dot_annots=None, masks=None, counts=tiny3bc005_train_counts, normalize=True, transform=args.cnn_transform)
tiny3bc005_train_dataloader = torch.utils.data.DataLoader(tiny3bc005_train_dataset, batch_size = args.cnn_batch_size_train, shuffle=True, num_workers=args.num_workers)

tiny3bc005_test_dataset = IMGs_dataset(images=tiny3bc005_test_images, dot_annots=None, masks=None, counts=tiny3bc005_test_counts, normalize=True, transform=False)
tiny3bc005_test_dataloader = torch.utils.data.DataLoader(tiny3bc005_test_dataset, batch_size = args.cnn_batch_size_test, shuffle=False, num_workers=args.num_workers)



#######################################################################################
'''                                 U-Net Training                                  '''
#######################################################################################
print("\n -----------------------------------------------------------------------------------------")
print("\n Start training U-Net >>>")

unet_ckpt_fullpath = save_models_folder + '/ckpt_U-Net_epochs_{}_transform_{}_seed_{}.pth'.format(args.unet_epochs, args.unet_transform, args.seed)
print('\n' + unet_ckpt_fullpath)

path_to_ckpt_in_train = save_models_folder + '/ckpts_in_train_U-Net_transform_{}_seed_{}'.format(args.unet_transform, args.seed)
os.makedirs(path_to_ckpt_in_train, exist_ok=True)

path_to_images_in_train = save_images_folder + '/images_in_train_U-Net_transform_{}_seed_{}'.format(args.unet_transform, args.seed)
os.makedirs(path_to_images_in_train, exist_ok=True)

### train the unet ###
if not os.path.isfile(unet_ckpt_fullpath):
    start = timeit.default_timer()
    print("\n Begin Training U-Net:")

    ## randomly choose some tiny-bbbc005 images to test U-Net during training
    indx_test = np.random.choice(np.arange(len(tiny3bc005_train_images)), size=16)
    test_images = tiny3bc005_train_images[indx_test]

    unet = UNet().cuda()
    unet = nn.DataParallel(unet)
    unet = train_unet(vgg_dataloader, test_images, unet, save_images_folder=path_to_images_in_train, path_to_ckpt=path_to_ckpt_in_train)

    # store model
    torch.save({
        'net_state_dict': unet.state_dict(),
    }, unet_ckpt_fullpath)

    stop = timeit.default_timer()
    print("U-Net training finished! Time elapses: {}s".format(stop - start))
else:
    print("\n Load pre-trained U-Net:")
    checkpoint = torch.load(unet_ckpt_fullpath)
    unet = UNet().cuda()
    unet = nn.DataParallel(unet)
    unet.load_state_dict(checkpoint['net_state_dict'])
# end if



#######################################################################################
'''                                  VGG Training                                   '''
#######################################################################################
print("\n -----------------------------------------------------------------------------------------")
print("\n Start training {} in {} >>>".format(args.cnn_name, args.experiment_name))

cnn_ckpt_fullpath = save_models_folder + '/ckpt_{}_{}_epochs_{}_transform_{}_seed_{}.pth'.format(args.experiment_name, args.cnn_name, args.cnn_epochs, args.cnn_transform, args.seed)
print('\n' + cnn_ckpt_fullpath)

path_to_ckpt_in_train = save_models_folder + '/ckpts_in_train_{}_{}_transform_{}_seed_{}'.format(args.experiment_name, args.cnn_name, args.cnn_transform, args.seed)
os.makedirs(path_to_ckpt_in_train, exist_ok=True)

### train the cnn ###
if not os.path.isfile(cnn_ckpt_fullpath):
    start = timeit.default_timer()
    print("\n Begin Training {}:".format(args.cnn_name))

    cnn_net = VGG(args.cnn_name).cuda()
    cnn_net = nn.DataParallel(cnn_net)
    cnn_net = train_cnn(trainloader = tiny3bc005_train_dataloader, testloader = tiny3bc005_test_dataloader, max_count=max_count, net=cnn_net, unet=unet, path_to_ckpt=path_to_ckpt_in_train)

    # store model
    torch.save({
        'net_state_dict': cnn_net.state_dict(),
    }, cnn_ckpt_fullpath)

    stop = timeit.default_timer()
    print("{} training finished! Time elapses: {}s".format(args.cnn_name, stop - start))
else:
    print("\n Load pre-trained {}:".format(args.cnn_name))
    checkpoint = torch.load(cnn_ckpt_fullpath)
    cnn_net = VGG(args.cnn_name).cuda()
    cnn_net = nn.DataParallel(cnn_net)
    cnn_net.load_state_dict(checkpoint['net_state_dict'])
# end if

### test the cnn ###
_, _ = test_cnn(testloader = tiny3bc005_test_dataloader, max_count=max_count, net=cnn_net, unet=unet, verbose=True)
