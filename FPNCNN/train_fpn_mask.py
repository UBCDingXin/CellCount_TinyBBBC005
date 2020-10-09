import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
import os
import timeit
from utils import *

### import settings ###
from opts import prepare_options
''' Settings '''
args = prepare_options()

# some parameters in the opts
epochs = args.fpn_epochs
resume_epoch = args.fpn_resume_epoch
lr_base = args.fpn_lr_base
lr_decay_epochs = args.fpn_lr_decay_epochs
lr_decay_epochs = [int(epoch) for epoch in lr_decay_epochs.split("_")]
lr_decay_factor = args.fpn_lr_decay_factor
weight_decay = args.fpn_weight_decay


# decay learning rate every args.dre_lr_decay_epochs epochs
def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate """
    lr = lr_base

    num_decays = len(lr_decay_epochs)
    for decay_i in range(num_decays):
        if epoch >= lr_decay_epochs[decay_i]:
            lr = lr * lr_decay_factor
        #end if epoch
    #end for decay_i
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_fpn_mask(trainloader, test_images, test_masks, fpn, save_images_folder, path_to_ckpt=None):

    # evaluate fpn on test_images (assume unnormalized)
    test_images = test_images / 255.0
    test_images = torch.from_numpy(test_images).type(torch.float).cuda()
    n_row = min(4, int(np.sqrt(test_images.shape[0])))
    save_image(test_images.data,
               save_images_folder + '/test_images.png',
               nrow=n_row,
               normalize=False)
    test_masks = torch.from_numpy(test_masks).type(torch.float).cuda()
    save_image(test_masks.data, save_images_folder + '/test_masks.png', nrow=n_row, normalize=False)

    # nets
    fpn = fpn.cuda()

    # define optimizer
    # optimizer = torch.optim.Adam(fpn.parameters(), lr = lr_base, betas=(0.5, 0.999), weight_decay=weight_decay)
    # optimizer = torch.optim.RMSprop(fpn.parameters(), lr=lr_base, alpha=0.99, eps=1e-08, weight_decay=weight_decay, momentum=0, centered=False)
    optimizer = torch.optim.SGD(fpn.parameters(),
                                lr=lr_base,
                                momentum=0.9,
                                weight_decay=weight_decay)

    if path_to_ckpt is not None and resume_epoch > 0:
        print("Loading ckpt to resume training FPN >>>")
        ckpt_fullpath = path_to_ckpt + "/fpn_checkpoint_epoch_{}.pth".format(resume_epoch)
        checkpoint = torch.load(ckpt_fullpath)
        fpn.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
        steps = checkpoint['steps']
    else:
        steps = 0

    start_time = timeit.default_timer()
    for epoch in range(resume_epoch, epochs):

        adjust_learning_rate(optimizer, epoch)

        train_loss = 0

        for batch_idx, batch_samples in enumerate(trainloader):

            fpn.train()

            batch_images = batch_samples['image']
            batch_true_mask = batch_samples['mask']  #already normalized to [0,1]

            batch_size_curr = batch_images.shape[0]
            assert batch_size_curr == batch_true_mask.shape[0]

            batch_images = batch_images.type(torch.float).cuda()
            batch_true_mask = batch_true_mask.type(torch.float).cuda()
            assert batch_true_mask.max() <= 1 and batch_true_mask.min() >= 0

            #forward pass
            ### TODO: accumulate gradient
            batch_recon_mask_mean_var = fpn(batch_images)
            loss = fpn_loss(batch_recon_mask_mean_var, batch_true_mask)

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()

            steps += 1

            if steps % 100 == 0:
                with torch.no_grad():
                    test_batch_recon_mask_mean, test_batch_recon_mask_logvar = fpn(test_images)

                test_batch_recon_mask_mean = test_batch_recon_mask_mean[-1].cpu()
                test_batch_recon_mask_var = torch.exp(test_batch_recon_mask_logvar[-1]).cpu()
                # test_batch_recon_mask_mean = torch.sigmoid(test_batch_recon_mask_mean[-1]).cpu()
                # test_batch_recon_mask_var = torch.sigmoid(torch.exp(test_batch_recon_mask_logvar[-1])).cpu()
                # test_batch_recon_mask_mean[test_batch_recon_mask_mean >= 0.5] = 1
                # test_batch_recon_mask_mean[test_batch_recon_mask_mean < 0.5] = 0
                # test_batch_recon_mask_var[test_batch_recon_mask_var >= 0.5] = 1
                # test_batch_recon_mask_var[test_batch_recon_mask_var < 0.5] = 0
                save_image(test_batch_recon_mask_mean.data,
                           save_images_folder + '/mean_{}.png'.format(steps),
                           nrow=n_row,
                           normalize=True)
                save_image(test_batch_recon_mask_var.data,
                           save_images_folder + '/var_{}.png'.format(steps),
                           nrow=n_row,
                           normalize=True)

            if steps % 20 == 0:
                print("FPN (mask): [step {}] [epoch {}/{}] [train loss {}] [Time {}]".format(
                    steps, epoch + 1, epochs, train_loss / (batch_idx + 1),
                    timeit.default_timer() - start_time))
        # end for batch_idx

        if path_to_ckpt is not None and (epoch + 1) % 50 == 0:
            save_file = path_to_ckpt + "/fpn_checkpoint_epoch_{}.pth".format(epoch + 1)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save(
                {
                    'steps': steps,
                    'net_state_dict': fpn.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'rng_state': torch.get_rng_state()
                }, save_file)
    #end for epoch

    return fpn
