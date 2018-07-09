from __future__ import print_function, division
import os
import io
import torch
import pandas as pd

import matplotlib
matplotlib.use('Agg')

from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Import other function files
from load_enhanced import EnhancedDataset
from load_enhanced import CNN
from load_enhanced import Rescale
from load_enhanced import ToTensor
from load_enhanced import get_train_loader

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Data loader
def get_train_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(transformed_dataset, batch_size = batch_size,
                                               shuffle=True, num_workers=2 )# sampler=train_sampler, num_workers=2)
    
    return(train_loader)

import time
                                               
## Getting the training data
transformed_dataset = EnhancedDataset(root_dir = '/home/home2/leichen/SuperResolutor/Dataset/Enhanced_train/',
                                      transform=transforms.Compose([
                                          Rescale((1020, 2040)),
                                          ToTensor()
                                      ]))

print(transformed_dataset)
"""
for i in range(100):
    sample = transformed_dataset[i+1]
    print('Rescale image: ', i+1, sample['image'].size())
"""

#sample = transformed_dataset[0]
#print('check whether image 0 exist', sample['image'].size())

sample = transformed_dataset[600]
print('image 600: ', 600, sample['image'].size())

sample = transformed_dataset[700]
print('image 700: ', 700, sample['image'].size())

## Test the data loader
batch_size = 40
train_loader = get_train_loader(batch_size)
n_batches = len(train_loader)

print(train_loader)
# print(train_loader.size())

for i_batch, sample_batched in enumerate(train_loader):
    print(i_batch, sample_batched['image'].size())
              
    

## Getting the training score
