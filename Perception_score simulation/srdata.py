from __future__ import print_function, division
import os
import torch
import pandas as pd

import matplotlib
matplotlib.use('Agg')

from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable

import scipy.io as sio
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Reproducible results
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

plt.ion()   # interactive mode

class SRData(Dataset):
    def __init__(self, score_file, root_dir, subdir_edsr, subdir_enhanced, subdir_hr, transform=None, train=True):
        self.root_dir = root_dir
        self.subdir_edsr = subdir_edsr
        self.subdir_enhanced = subdir_enhanced
        self.subdir_hr = subdir_hr

        self.transform = transform
        self.split = 'train' if train else 'test'

        self.score_file = score_file

    def __len__(self):
        # return len(self.images_hr)
        return 2401

    def __getitem__(self, idx):
        # img_name = os.path.join(self.root_dir, str(idx)+ '.png')  # get the img file path + name
        score_name = os.path.join(self.root_dir, self.score_file)
        scores_content = sio.loadmat(score_name)
        scores_arr = scores_content['scores_tot']

        if (idx == 0):
            img_name = os.path.join(self.root_dir, self.subdir_edsr, '0.png')
            image = io.imread(img_name)
            score = scores_arr[0][0]
        elif (idx <= 800): 
            img_name = os.path.join(self.root_dir, self.subdir_edsr, str(idx)+ '.png')
            image = io.imread(img_name)
            score = scores_arr[idx-1][0]
        elif (idx <= 1600):
            img_name = os.path.join(self.root_dir, self.subdir_hr, str(idx-800)+ '.png')
            image = io.imread(img_name)
            score = scores_arr[idx-801][1]
        else: 
            img_name = os.path.join(self.root_dir, self.subdir_enhanced, str(idx-1600)+ '.png')
            image = io.imread(img_name)
            score = scores_arr[idx-1601][2]

        sample = {'image': image, 'score': score}
        
        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]     # (h, w, color)
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h/w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w/h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'score': sample['score']}

class ToTensor(object):
    """ convert ndarrays in sample to Tensors """
    def __call__(self, sample):
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'score': sample['score']}

transformed_dataset = SRData(score_file = 'F_per_arr.mat',
                                      root_dir = '/usr/project/xtmp/superresoluter/approximater_training_set/',
                                      subdir_edsr = 'EDSR/DIV2K_train/',
                                      subdir_enhanced = 'EhanceNet/DIV2K_train/',
                                      subdir_hr = 'HR/DIV2K_train/',
                                      transform=transforms.Compose([
                                          Rescale((1020, 2040)),
                                          ToTensor()
                                      ]))

#sample = transformed_dataset[0]
#print('idx = 0 img: ', sample['image'].size())

"""
for i in range(700, len(transformed_dataset)):
    sample = transformed_dataset[i+1]
    print('Rescale image: ', i+1, sample['image'].size(), 'score = ' , sample['score'])
"""


## Data loader
def get_train_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(transformed_dataset, batch_size = batch_size,
                                               shuffle=True, num_workers=4 )# sampler=train_sampler, num_workers=2)
    return(train_loader)

batch_size = 100
train_loader = get_train_loader(batch_size)
n_batches = len(train_loader)

for i, data in enumerate(train_loader, 1):
    inputs = data['image']
    scores = data['score'].float()

    inputs = Variable(inputs)
    scores = Variable(scores)

    print(i, ' inputs: ', inputs.size(), 'scores: ', scores.size())
    
