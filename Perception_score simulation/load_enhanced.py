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

# Reproducible results
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

plt.ion()   # interactive mode

class EnhancedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return 800
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(idx)+ '.png')  # get the img file path + name
        image = io.imread(img_name)
        sample = {'image': image}
        print(idx, sample['image'].shape)
        
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

        print('original sample: ', image.shape)
        # print('Rescale sample: ', img.shape)
        # print('Original image pixels: ', image)
        # print('resize img pixels: ', img)
        return {'image': img}

class ToTensor(object):
    """ convert ndarrays in sample to Tensors """
    def __call__(self, sample):
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image)}

# enhanced_dataset = EnhancedDataset(root_dir = 'enhanced_train/')

fig = plt.figure()

transformed_dataset = EnhancedDataset(root_dir = 'enhanced_train/',
                                      transform=transforms.Compose([
                                          Rescale((1020, 2040)),
                                          ToTensor()
                                      ]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i+1]
    print('Rescale image: ', i+1, sample['image'].size())
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()        # can be super(CNN), or super(Net), difference?
        self.conv1 = nn.Sequential (
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride = 1,
                padding = 2,               # for same width and length for img after conv2d
                # input: (1, 28, 28) -> output(16, 28, 28)
            ),
c            nn.ReLU(),                     # activation function
            nn.MaxPool2d(kernel_size = 2), # choose max value in 2*2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),    # (16, 14, 14) -> (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),               # (32, 7, 7)
        )

        self.regressor = nn.Sequential(
            # what's nn.Dropout used for?
            nn.Linear(32 * 7 * 7, 10),  # fully connected layer: output 10 classes
            # what's nn.BatchNormld used for?
            # what's ReLu used for here?
            nn.Linear(10, 10),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.regressor(x)
        return output, x
    
cnn = CNN()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr = LR)
loss_func = nn.MSEloss()


# Train the model
