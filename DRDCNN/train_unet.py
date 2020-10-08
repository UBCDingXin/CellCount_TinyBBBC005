import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
import os
import timeit


### import settings ###
from opts import prepare_options

''' Settings '''
args = prepare_options()


# some parameters in the opts
epochs = args.unet_epochs
resume_epoch = args.unet_resume_epoch
lr_base = args.unet_lr_base
lr_decay_epochs = args.unet_lr_decay_epochs
lr_decay_epochs = [int(epoch) for epoch in lr_decay_epochs.split("_")]
lr_decay_factor = args.unet_lr_decay_factor
weight_decay = args.unet_weight_decay

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


def train_unet(trainloader, test_images, unet, save_images_folder, path_to_ckpt=None):

    # evaluate unet on test_images (assume unnormalized)
    test_images = test_images/255.0
    test_images = (test_images-0.5)/0.5
    test_images = torch.from_numpy(test_images).type(torch.float).cuda()
    n_row=min(4, int(np.sqrt(test_images.shape[0])))
    save_image(test_images.data, save_images_folder + '/test_images.png', nrow=n_row, normalize=True)

    # nets
    unet = unet.cuda()

    # define optimizer
    optimizer = torch.optim.Adam(unet.parameters(), lr = lr_base, betas=(0.0, 0.999), weight_decay=weight_decay)
    # optimizer = torch.optim.RMSprop(unet.parameters(), lr=lr_base, alpha=0.99, eps=1e-08, weight_decay=weight_decay, momentum=0, centered=False)

    # criterion
    criterion = nn.MSELoss()

    if path_to_ckpt is not None and resume_epoch>0:
        print("Loading ckpt to resume training U-net >>>")
        ckpt_fullpath = path_to_ckpt + "/unet_checkpoint_epoch_{}.pth".format(resume_epoch)
        checkpoint = torch.load(ckpt_fullpath)
        unet.load_state_dict(checkpoint['net_state_dict'])
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

            unet.train()

            batch_images = batch_samples['image']
            batch_true_maps = batch_samples['density_map']

            batch_size_curr = batch_images.shape[0]
            assert batch_size_curr == batch_true_maps.shape[0]

            batch_images = batch_images.type(torch.float).cuda()
            batch_true_maps = batch_true_maps.type(torch.float).cuda()

            #forward pass
            batch_recon_maps = unet(batch_images)
            loss = criterion(batch_recon_maps, batch_true_maps)

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()

            steps += 1

            if steps % 100 == 0:
                with torch.no_grad():
                    test_batch_recon_maps = unet(test_images)
                    test_batch_recon_maps = test_batch_recon_maps/255.0
                save_image(test_batch_recon_maps.data, save_images_folder + '/tiny3bc005_{}.png'.format(steps), nrow=n_row, normalize=True)
                save_image(batch_recon_maps[0:n_row**2].data, save_images_folder + '/vgg_{}.png'.format(steps), nrow=n_row, normalize=True)

            if steps % 20 == 0:
                print("U-net: [step {}] [epoch {}/{}] [train loss {}] [Time {}]".format(steps, epoch+1, epochs, train_loss/(batch_idx+1), timeit.default_timer()-start_time) )
        # end for batch_idx

        if path_to_ckpt is not None and (epoch+1) % 50 == 0:
            save_file = path_to_ckpt + "/unet_checkpoint_epoch_{}.pth".format(epoch+1)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                    'steps': steps,
                    'net_state_dict': unet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'rng_state': torch.get_rng_state()
            }, save_file)
    #end for epoch

    return unet
