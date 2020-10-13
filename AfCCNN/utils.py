import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib as mpl
import PIL
from PIL import Image
import scipy
import scipy.ndimage

################################################################################
TransHFlip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
TransVFlip = torchvision.transforms.RandomVerticalFlip(p=0.5)
class IMGs_dataset(torch.utils.data.Dataset):
    def __init__(self, images, dot_annots=None, masks=None, counts=None, normalize=True, transform=False):
        '''
        images: numpy array images;
        dot_annots: dot annotations; numpy array images
        masks: segmentation masks
        counts: cell counts
        normalize: normalize images to [-1,1]
        transform: random transform on images; horizontal and vertical flip, random rotation (90,180,270)
        '''

        super(IMGs_dataset, self).__init__()

        self.images = images
        self.image_mean = np.mean(images)
        self.image_std = np.std(images)
        self.n_images = len(self.images)
        self.counts = counts
        self.dot_annots = dot_annots
        self.masks = masks

        if self.counts is not None:
            assert self.n_images == len(self.counts)
        if self.dot_annots is not None:
            assert self.n_images == len(self.dot_annots)
        if self.masks is not None:
            assert self.n_images == len(self.masks)
        self.transform = transform
        self.normalize = normalize

    def __getitem__(self, index):

        ## for grey scale only
        image = self.images[index]
        if self.dot_annots is not None:
            dot_annot = self.dot_annots[index]
        if self.masks is not None:
            mask = self.masks[index]

        if self.transform:
            image_pil = Image.fromarray(np.uint8(image[0]), mode = 'L') #H * W
            # rotation_degree = np.random.choice(np.array([0, 90, 180, 270]))
            # image_pil = torchvision.transforms.functional.rotate(image_pil, rotation_degree)
            image_pil = TransHFlip(image_pil)
            image_pil = TransVFlip(image_pil)
            image[0] = np.array(image_pil)
            if self.dot_annots is not None:
                dot_annot_pil = Image.fromarray(np.uint8(dot_annot[0]), mode = 'L')
                dot_annot_pil = torchvision.transforms.functional.rotate(dot_annot_pil, rotation_degree)
                dot_annot_pil = TransHFlip(dot_annot_pil)
                dot_annot_pil = TransVFlip(dot_annot_pil)
                dot_annot[0] = np.array(dot_annot_pil)
            if self.masks is not None:
                mask_pil = Image.fromarray(np.uint8(mask[0]), mode = 'L')
                mask_pil = torchvision.transforms.functional.rotate(mask_pil, rotation_degree)
                mask_pil = TransHFlip(mask_pil)
                mask_pil = TransVFlip(mask_pil)
                mask[0] = np.array(mask_pil)

        if self.normalize:
            image = image/255.0
            image = (image-0.5)/0.5
            # image = (image-self.image_mean)/self.image_std

        if self.counts is not None:
            count = self.counts[index]

        output = {}
        output['image'] = image
        if self.dot_annots is not None:
            ## Gaussian filter setting is based on line 53-54 of https://github.com/WeidiXie/cell_counting_v2/blob/master/train.py
            density_map = 100.0 * (dot_annot[0] > 0)
            density_map = scipy.ndimage.gaussian_filter(density_map, sigma=(1, 1), order=0)
            density_map = density_map[np.newaxis,:,:]
            output['density_map'] = density_map
        if self.masks is not None:
            output['mask'] = mask
        if self.counts is not None:
            output['count'] = count

        return output

    def __len__(self):
        return self.n_images



################################################################################
def PlotLoss(loss, filename):
    x_axis = np.arange(start = 1, stop = len(loss)+1)
    plt.switch_backend('agg')
    mpl.style.use('seaborn')
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x_axis, np.array(loss))
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),  shadow=True, ncol=3)
    #plt.title('Training Loss')
    plt.savefig(filename)
