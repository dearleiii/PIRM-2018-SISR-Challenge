from __future__ import print_function, division
import os
import torch
import pandas as pd

import matplotlib
matplotlib.use('Agg')

from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

# no landmarks required
# landmarks_frame = pd.read_csv('faces/face_landmarks.csv')

class DIVvalidDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        # return len(self.landmarks_frame)
        return 100
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, '0'+str(idx)+ '.png')  # get the img file path + name
        image = io.imread(img_name)
        # landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample

# face_dataset = FaceLandmarksDataset(csv_file='faces/face_landmarks.csv',root_dir='faces/')
div_valid_dataset = DIVvalidDataset(root_dir = 'div2k/')

fig = plt.figure()

for i in range(len(div_valid_dataset)):
    sample = div_valid_dataset[i+801]

    print(i, sample['image'].shape)

    
