'''
Function for training a classification CNN.
'''


import torch
import torch.nn as nn
import os
import timeit
import math

### import settings ###
from opts import prepare_options

''' Settings '''
args = prepare_options()


# some parameters in the opts
cnn_name = args.cnn_name
epochs = args.epochs
resume_epoch = args.resume_epoch
lr_base = args.lr_base
lr_decay_epochs = args.lr_decay_epochs
lr_decay_epochs = [int(epoch) for epoch in lr_decay_epochs.split("_")]
lr_decay_factor = args.lr_decay_factor
weight_decay = args.weight_decay


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


def train_class_cnn(trainloader, testloader, classlabel_to_count, net, path_to_ckpt=None):
    '''
    net: cnn net
    classlabel_to_count: a dict stores a mapping from a class label to a cell count
    '''

    # nets
    net = net.cuda()

    # define optimizer
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr_base, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr_base, weight_decay=weight_decay)

    # criterion
    criterion = nn.CrossEntropyLoss()

    if path_to_ckpt is not None and resume_epoch>0:
        print("Loading ckpt to resume training the CNN >>>")
        ckpt_fullpath = path_to_ckpt + "/class_{}_checkpoint_epoch_{}.pth".format(cnn_name, resume_epoch)
        checkpoint = torch.load(ckpt_fullpath)
        net.load_state_dict(checkpoint['net_state_dict'])
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

            net.train()

            batch_images = batch_samples['image']
            batch_labels = batch_samples['count'] #class labels

            batch_size_curr = batch_images.shape[0]
            assert batch_size_curr == batch_labels.shape[0]

            batch_images = batch_images.type(torch.float).cuda()
            batch_labels = batch_labels.type(torch.long).cuda()

            #forward pass
            batch_output = net(batch_images)
            loss = criterion(batch_output, batch_labels)

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()

            steps += 1

            if steps % 20 == 0:
                print("\nclass {}: [step {}] [epoch {}/{}] [train loss {:.3f}] [Time {:.3f}]".format(cnn_name, steps, epoch+1, epochs, train_loss/(batch_idx+1), timeit.default_timer()-start_time) )
        # end for batch_idx

        _ = test_class_cnn(testloader, classlabel_to_count, net, verbose=True)
    #end for epoch

        if path_to_ckpt is not None and (epoch+1) % 50 == 0:
            save_file = path_to_ckpt + "/class_{}_checkpoint_epoch_{}.pth".format(cnn_name, epoch+1)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                    'steps': steps,
                    'net_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'rng_state': torch.get_rng_state()
            }, save_file)
    #end for epoch

    return net


def test_class_cnn(testloader, classlabel_to_count, net, verbose=True):
    net = net.eval()

    # criterion
    square_error = nn.MSELoss(reduction="sum")
    absolute_error = nn.L1Loss(reduction="sum")

    test_rmse = 0
    test_mae = 0
    test_acc = 0 #test accuracy
    predicted_counts = []

    with torch.no_grad():
        for batch_idx, batch_samples in enumerate(testloader):
            batch_images = batch_samples['image']
            batch_counts = batch_samples['count'] #unnormalized; original cell count

            batch_images = batch_images.type(torch.float).cuda()
            batch_counts = batch_counts.type(torch.float)

            #forward pass
            batch_output = net(batch_images)
            _, batch_output = torch.max(batch_output.data, 1) ##convert one-hot vectors to class labels
            ## convert class labels to cell counts
            batch_pred_counts = []
            for j in batch_output:
                batch_pred_counts.append(classlabel_to_count[j.item()])
            #end j
            batch_pred_counts = torch.tensor(batch_pred_counts).type(torch.float)
            predicted_counts.append(batch_pred_counts)

            batch_sum_of_square_error = square_error(batch_pred_counts.view(-1), batch_counts.view(-1))
            batch_sum_of_absolute_error = absolute_error(batch_pred_counts.view(-1), batch_counts.view(-1))
            test_rmse += batch_sum_of_square_error.item()
            test_mae += batch_sum_of_absolute_error.item()
            test_acc += (batch_pred_counts.view(-1) == batch_counts.view(-1)).type(torch.float).sum().item()
        #end for batch_idx
    test_rmse = math.sqrt(test_rmse/len(testloader.dataset))
    test_mae = test_mae/len(testloader.dataset)
    test_acc = test_acc/len(testloader.dataset)
    predicted_counts = torch.cat(predicted_counts, dim=0).numpy()

    if verbose:
        print('\n====> Test set RMSE: %.4f; Test set MAE: %.4f; Test set Acc.: %.4f' % (test_rmse, test_mae, test_acc))

    output = {}
    output['RMSE'] = test_rmse
    output['MAE'] = test_mae
    output['Accuracy'] = test_acc
    output['Pred'] = predicted_counts

    return output
