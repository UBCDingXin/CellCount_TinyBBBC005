"""
Some useful functions

"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torchvision
from PIL import Image
import gc

#------------------------------------------------------------------------------
# Training and testing dataset
TransHFlip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
TransVFlip = torchvision.transforms.RandomVerticalFlip(p=0.5)

class BBBCDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform = (0.5, 0.5), rotation = False, flipping = False, byblur = None, bystain = None):
        super(BBBCDataset, self).__init__()

        # data: data is a list which contians [IMGs, Labels] or [IMGs, Labels, Blur, Stain, IMGs_Names]
        # transform: mean and standard deviation
        # rotation: randomly rotate
        # flipping: randomly flip images
        # byBlur: generate images by blur level; Its value can be 1, 23, 48 or None (means take all data)
        # byStain: generate images by Stain type; Its value can be 1, 2, or None (means take all data)
        self.n_input = len(data)
        self.transform = transform
        self.rotation = rotation
        self.flipping = flipping

        assert bystain in [1, 2, None]
        assert byblur in [1,23,48,None]

        if byblur is None and bystain is None:
            sel_indx = np.arange((data[0]).shape[0])
        elif byblur is not None and bystain is None:
            sel_indx = np.where(data[1]==byblur)[0]
        elif byblur is None and bystain is not None:
            sel_indx = np.where(data[2]==bystain)[0]
        elif byblur is not None and bystain is not None:
            sel_indx = np.where(data[1]==byblur and data[2]==bystain)[0]

        self.IMGs = data[0][sel_indx]
        self.Labels = data[1][sel_indx]
        self.n_images = self.IMGs.shape[0]

        if self.n_input>2:
            self.Blur = data[2][sel_indx]
            self.Stain = data[3][sel_indx]
            #self.IMGs_Names = data[4][sel_indx]

    def __getitem__(self, index):

        image = self.IMGs[index]
        image_label = self.Labels[index]

        if self.rotation:
            PIL_im = Image.fromarray(np.uint8(image[0]), mode = 'L')
            rotation_degree = np.random.choice(np.array([0, 90, 180, 270]))
            PIL_im = torchvision.transforms.functional.rotate(PIL_im, rotation_degree)
            image[0] = np.array(PIL_im)

        if self.flipping:
            PIL_im = Image.fromarray(np.uint8(image[0]), mode = 'L')
            PIL_im = TransHFlip(PIL_im)
            PIL_im = TransVFlip(PIL_im)
            image[0] = np.array(PIL_im)

        if self.transform is not None:
            image = image/255.0
            image = (image-self.transform[0])/self.transform[1]

        if self.n_input>2:
            image_blur = self.Blur[index]
            image_stain = self.Stain[index]
            #image_name = self.IMGs_Names[index]
        else:
            image_blur = 0
            image_stain = 0
        return image, image_label, image_blur, image_stain

    def __len__(self):
        return self.n_images


class BBBCDataset_blurPairs(torch.utils.data.Dataset):
    def __init__(self, data, transform = (0.5, 0.5), flipping = False):
        super(BBBCDataset_blurPairs, self).__init__()

        # data: data is a list which contians [IMGs1, IMGs2]
        # transform: mean and standard deviation
        # flipping: randomly flip images
        self.n_input = len(data)
        self.transform = transform
        self.flipping = flipping

        self.IMGs1 = data[0]
        self.IMGs2 = data[1]
        self.n_images = self.IMGs1.shape[0]

    def __getitem__(self, index):

        image1 = self.IMGs1[index]
        image2 = self.IMGs2[index]

        if self.flipping:
            PIL_im1 = Image.fromarray(np.uint8(image1[0]), mode = 'L')
            PIL_im1 = TransHFlip(PIL_im1)
            PIL_im1 = TransVFlip(PIL_im1)
            image1[0] = np.array(PIL_im1)
            PIL_im2 = Image.fromarray(np.uint8(image2[0]), mode = 'L')
            PIL_im2 = TransHFlip(PIL_im2)
            PIL_im2 = TransVFlip(PIL_im2)
            image2[0] = np.array(PIL_im2)

        if self.transform is not None:
            image1 = image1/255.0
            image1 = (image1-self.transform[0])/self.transform[1]
            image2 = image2/255.0
            image2 = (image2-self.transform[0])/self.transform[1]

        return image1, image2

    def __len__(self):
        return self.n_images

#----------------------------------------------------------------------------
# flip images; for RF and XGBoost
TransHFlip_Det = torchvision.transforms.RandomHorizontalFlip(p=1) #deterministic, not random
TransVFlip_Det = torchvision.transforms.RandomVerticalFlip(p=1)
def Dataset_flip(IMGs, CellCount, Blur, Stain):
    N_raw = IMGs.shape[0]

    #horizontal flipping
    IMGs_tmp = np.zeros(IMGs.shape).astype(np.uint8)
    for i in range(N_raw):
        image=IMGs[i][0]
        PIL_im = Image.fromarray(np.uint8(image), mode = 'L')
        PIL_im = TransHFlip_Det(PIL_im)
        IMGs_tmp[i][0] = np.array(PIL_im)
    IMGs_new = np.concatenate((IMGs, IMGs_tmp), axis=0)
    del IMGs_tmp; gc.collect()

    #vertical flipping
    IMGs_tmp = np.zeros(IMGs.shape).astype(np.uint8)
    for i in range(N_raw):
        image=IMGs[i][0]
        PIL_im = Image.fromarray(np.uint8(image), mode = 'L')
        PIL_im = TransVFlip_Det(PIL_im)
        IMGs_tmp[i][0] = np.array(PIL_im)
    IMGs_new = np.concatenate((IMGs_new, IMGs_tmp), axis=0)
    del IMGs_tmp; gc.collect()

    #horizontal and vertical flipping
    IMGs_tmp = np.zeros(IMGs.shape).astype(np.uint8)
    for i in range(N_raw):
        image=IMGs[i][0]
        PIL_im = Image.fromarray(np.uint8(image), mode = 'L')
        PIL_im = TransHFlip_Det(PIL_im)
        PIL_im = TransVFlip_Det(PIL_im)
        IMGs_tmp[i][0] = np.array(PIL_im)
    IMGs_new = np.concatenate((IMGs_new, IMGs_tmp), axis=0)
    del IMGs_tmp; gc.collect()

    CellCount_new = np.tile(CellCount, 4)
    Blur_new = np.tile(Blur, 4)
    Stain_new = np.tile(Stain, 4)

    return IMGs_new, CellCount_new, Blur_new, Stain_new



#----------------------------------------------------------------------------
# adjust CNN learning rate
def adjust_learning_rate(optimizer, epoch, base_lr):
    """decrease the learning rate over epochs"""
    lr = base_lr
    if epoch >= 10: #base_lr = 10**-3
        lr /= 10
    if epoch >= 30:
        lr /= 10
    if epoch >= 50:
        lr /= 10
    if epoch >= 70:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_v2(optimizer, epoch, base_lr):
    """decrease the learning rate over epochs"""
    lr = base_lr
    if epoch >= 10: #base_lr = 10**-3
        lr /= 10
    if epoch >= 30:
        lr /= 10
    # if epoch >= 50:
    #     lr /= 10
    # if epoch >= 70:
    #     lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


#----------------------------------------------------------------------------
def PlotTrainCurves(losses, loss_names, TrainCurves_filename):
    #losses: a list of losses
    #loss_names: a list of names
    n_loss = len(losses)
    assert n_loss == len(loss_names)

    colors = ['red', 'blue', 'black', 'green']

    x_axis = np.arange(start = 1, stop = len(losses[0])+1)
    plt.switch_backend('agg')
    mpl.style.use('seaborn')
    for i in range(n_loss):
        plt.plot( x_axis, np.array(losses[i]), color=colors[i], linewidth=1, label=loss_names[i])
    plt.legend()
    plt.title("Training Curves")
    plt.savefig(TrainCurves_filename)




#----------------------------------------------------------------------------
def DA_intensity_analysis(plot_imgs, plot_cellcount, plot_stain, plot_blur, filename):
    plot_intensity = (np.mean(plot_imgs, axis=(1,2,3))).reshape(-1,1)

    plt.figure(figsize=(20, 10))
    plt.subplot(231)
    indx_stain_blur = np.where((plot_stain==1)*(plot_blur==1)==True)[0]
    x=plot_intensity[indx_stain_blur]; y = plot_cellcount[indx_stain_blur]
    plt.scatter(x, y)
    plt.xlim(0,100)
    plt.xlabel("intensity")
    plt.ylabel("cell count")
    plt.text(np.max(plot_intensity)+2, np.max(y)/2, 'stain=1', horizontalalignment='center', verticalalignment='center',rotation=-90)
    plt.text(np.max(plot_intensity)/2, np.max(y)+8, 'blur=1', horizontalalignment='center', verticalalignment='center')

    plt.subplot(232)
    indx_stain_blur = np.where((plot_stain==1)*(plot_blur==23)==True)[0]
    x=plot_intensity[indx_stain_blur]; y = plot_cellcount[indx_stain_blur]
    plt.scatter(x, y)
    plt.xlim(0,100)
    plt.xlabel("intensity")
    plt.ylabel("cell count")
    plt.text(np.max(plot_intensity)+2, np.max(y)/2, 'stain=1', horizontalalignment='center', verticalalignment='center',rotation=-90)
    plt.text(np.max(plot_intensity)/2, np.max(y)+8, 'blur=23', horizontalalignment='center', verticalalignment='center')

    plt.subplot(233)
    indx_stain_blur = np.where((plot_stain==1)*(plot_blur==48)==True)[0]
    x=plot_intensity[indx_stain_blur]; y = plot_cellcount[indx_stain_blur]
    plt.scatter(x, y)
    plt.xlim(0,100)
    plt.xlabel("intensity")
    plt.ylabel("cell count")
    plt.text(np.max(plot_intensity)+2, np.max(y)/2, 'stain=1', horizontalalignment='center', verticalalignment='center',rotation=-90)
    plt.text(np.max(plot_intensity)/2, np.max(y)+8, 'blur=48', horizontalalignment='center', verticalalignment='center')

    plt.subplot(234)
    indx_stain_blur = np.where((plot_stain==2)*(plot_blur==1)==True)[0]
    x=plot_intensity[indx_stain_blur]; y = plot_cellcount[indx_stain_blur]
    plt.scatter(x, y)
    plt.xlabel("intensity")
    plt.ylabel("cell count")
    plt.text(np.max(plot_intensity)+8, np.max(y)/2, 'stain=2', horizontalalignment='center', verticalalignment='center',rotation=-90)
    plt.text(np.max(plot_intensity)/2, np.max(y)+8, 'blur=1', horizontalalignment='center', verticalalignment='center')

    plt.subplot(235)
    indx_stain_blur = np.where((plot_stain==2)*(plot_blur==23)==True)[0]
    x=plot_intensity[indx_stain_blur]; y = plot_cellcount[indx_stain_blur]
    plt.scatter(x, y)
    plt.xlabel("intensity")
    plt.ylabel("cell count")
    plt.text(np.max(plot_intensity)+8, np.max(y)/2, 'stain=2', horizontalalignment='center', verticalalignment='center',rotation=-90)
    plt.text(np.max(plot_intensity)/2, np.max(y)+8, 'blur=23', horizontalalignment='center', verticalalignment='center')

    plt.subplot(236)
    indx_stain_blur = np.where((plot_stain==2)*(plot_blur==48)==True)[0]
    x=plot_intensity[indx_stain_blur]; y = plot_cellcount[indx_stain_blur]
    plt.scatter(x, y)
    plt.xlabel("intensity")
    plt.ylabel("cell count")
    plt.text(np.max(plot_intensity)+8, np.max(y)/2, 'stain=2', horizontalalignment='center', verticalalignment='center',rotation=-90)
    plt.text(np.max(plot_intensity)/2, np.max(y)+8, 'blur=48', horizontalalignment='center', verticalalignment='center')

    plt.tight_layout()
    plt.savefig(filename, format="png")
    # plt.show()
