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
        print('resize img pixels: ', img[:,:4,:4])
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

transformed_dataset = EnhancedDataset(root_dir = '/home/home2/leichen/SuperResolutor/Dataset/Enhanced_train/',
                                      transform=transforms.Compose([
                                          Rescale((1020, 2040)),
                                          ToTensor()
                                      ]))


for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i+1]
    print('Rescale image: ', i+1, sample['image'].size())


class CNN(nn.Module):

    # out batch shape for input x is 3* 1020 * 2040
    
    def __init__(self):
        super(CNN, self).__init__()        # can be super(CNN), or super(Net), difference?
        self.conv1 = nn.Sequential (
            # Computes the activation of the first convolution
            # Size changes from (3, 1020, 2040) to (18, 510, 2010)
            nn.Conv2d(
                in_channels=3,             # Input channels = 3            
                out_channels=18,           # Output channels = 18 
                kernel_size=5,
                stride = 1,
                padding = 2,               # for same width and length for img after conv2d
                                           # input: (3, 1020, 2040) -> output(18, 1020, 2040)
            ),
            nn.ReLU(),                     # activation function
            nn.MaxPool2d(kernel_size = 2), # choose max value in 2*2 area, output shape (18, 510, 1020)
                                           # Can set up maxPool2d differently, ex. MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.conv2 = nn.Sequential(
            # Conpute the activation of the second convolution
            # Size changes from (18, 510, 2010) to (36, 255, 510)
            nn.Conv2d(18, 36, 5, 1, 2),        # (18, 510, 1020) -> (36, 510, 1020)
            nn.ReLU(),
            nn.MaxPool2d(2),                   # (36, 255, 510)
        )
        
        self.regressor = nn.Sequential(
            # what's nn.Dropout used for?
            nn.Linear(36 * 255 * 510, 64),  
            # what's nn.BatchNormld used for?
            # what's ReLu used for here?
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Linear(64, 10),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        #Reshape data to input to the input layer of the neural net
        #Size changes from (36, 255, 510) to (1, flatten)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(x.size(0), -1)

        #Computes the activation of the first fully connected layer
        #Size changes from (1, flatten) to (1, 64)
        output = self.regressor(x)
        return output, x
        

cnn = CNN()
print(cnn)

# Wrap the loss & optimization functions to find CNN the right weights 
optimizer = torch.optim.Adam(cnn.parameters(), lr = 0.001)
loss_func = nn.MSELoss()


# Train the model
# Each time we pass through the loop (called and “epoch”), we compute a forward pass on the network and implement backpropagation to adjust the weights
# We’ll also record some other measurements like loss and time passed, so that we can analyze them as the net trains itself.

# Data loader
def get_train_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(transformed_dataset, batch_size = batch_size,
                                               shuffle=True, num_workers=2 )# sampler=train_sampler, num_workers=2)
    return(train_loader)

import scipy.io as sio
mat_content = sio.loadmat('F_per_Enhancednet-DIV2K80.mat')
scores_arr = mat_content['F_per_array']
scores_torch = torch.from_numpy(scores_arr)

print(scores_arr.shape)


import time

def trainNet(net, batch_size, n_epochs, learning_rate):
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    # Getting the training data
    train_loader = get_train_loader(batch_size)
    n_batches = len(train_loader)

    # Time for printing
    training_start_time = time.time()

    # Loop fr n_epochs
    for epoch in range(n_epochs):
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0

        for i, data in enumerate(train_loader, 0):
            # Convert numpy arrays to torch tensors
            # inputs = torch.from_numpy(x_train)
            # targets = torch.from_numpy(y_train)
            inputs = data
            scores = data_score
            
            # Wrap them in a Variable object
            inputs = Varioable(inputs)
            scores = Variable(scores)
            
            # Set the parameter gradients to zero
            optimizer.zero_grad()
            
            # Forward pass
            # outputs = model(inputs)
            # loss = criterion(outputs, targets)
            outputs = net(inputs)
            loss_size = loss_func(outputs, scores)
            
            # Backward and optimize 
            loss_size.backward()
            optimizer.step()
            
            # Print the statistics
            running_loss += loss_size.data[0]
            total_train_loss += loss_size.data[0]
            
            # Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                    epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()
                
                # If require further validation set 


# To actually train the NN
CNN = CNN()
# trainNet(CNN, batch_size = 32, n_epochs = 5, learning_rate = 0.001)

