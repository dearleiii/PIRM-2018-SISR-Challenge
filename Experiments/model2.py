import os

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

#from apxm import APXM_4conv
from apxm_conv3 import APXM_conv3
import srdata

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt 

# compute on gpu
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assume that we are on a CUDA machine, then this should print a CUDA device:
#print(device)
#print(torch.cuda.current_device())

transformed_dataset = srdata.SRData(score_file = 'F_per_arr.mat',
                                      root_dir = '/usr/project/xtmp/superresoluter/approximater_training_set/',
                                      subdir_edsr = 'EDSR/DIV2K_train/',
                                      subdir_enhanced = 'EhanceNet/DIV2K_train/',
                                      subdir_hr = 'HR/DIV2K_train/',
                                      transform=transforms.Compose([
                                          srdata.Rescale((1024, 2048)),
                                          srdata.ToTensor()
                                      ]))
approximator = APXM_conv3()
#approximator = torch.nn.DataParallel(APXM_4conv, device_ids=[0, 1, 2])
#approximator.to(device)
#approximator.cuda()

device = torch.device("cuda:0")
gpu_list = list(range(0, torch.cuda.device_count()))
if torch.cuda.device_count() > 1:
    print("Let's use::; ", torch.cuda.device_count(), "GPUs!")
    approximator = nn.DataParallel(approximator, gpu_list).cuda()
    print("cuda.current_device=", torch.cuda.current_device())
    #approximator.cuda()
    
print(approximator)

## Data loader
def get_train_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(transformed_dataset, batch_size = batch_size,
                                               shuffle=True, num_workers=4 )# sampler=train_sampler, num_workers=2)
    return(train_loader)

batch_size = 100
train_loader = get_train_loader(batch_size)
n_batches = len(train_loader)    

# Wrap the loss & optimization functions to find CNN the right weights 
optimizer = torch.optim.Adam(approximator.parameters(), lr = 0.001)
loss_func = nn.MSELoss()


# Train the model
# Each time we pass through the loop (called and “epoch”),
# we compute a forward pass on the network and implement backpropagation to adjust the weights
# We’ll also record some other measurements like loss and time passed,
# so that we can analyze them as the net trains itself.

## Training Net

import time
from torch.autograd import Variable

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
    train_losses = []
    epoch_list = []
    for epoch in range(n_epochs):
        running_loss = 0.0
        epoch_train_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0
        
        for i, data in enumerate(train_loader, 0):
            inputs = data['image']
            inputs = inputs.cuda()
            scores = data['score'].float()
            scores = scores.cuda()
            scores = torch.unsqueeze(scores, 1)

            inputs = Variable(inputs, requires_grad=False)
            scores = Variable(scores, requires_grad=False)
            
            # Set the parameter gradients to zero
            optimizer.zero_grad()
            
            # Forward pass
            outputs = net(inputs)

            print(i, 'score:: ', scores.size(),
                  'outputs: ', np.shape(outputs))
            loss_size = loss_func(outputs, scores)
            
            # Backward and optimize 
            loss_size.backward()
            optimizer.step()
            
            # Print the statistics
            running_loss += loss_size.data[0]
            total_train_loss += loss_size.data[0]
            epoch_train_loss += loss_size.data[0]

            # Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                    epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                running_loss = 0.0
                start_time = time.time()

        ## End of Epoch, store loss at this epoch        
        epoch_train_loss_cpu = epoch_train_loss.cpu().numpy()
        
        train_losses.append(epoch_train_loss_cpu.item(0))
#        train_losses.tolist()
        epoch_list.append(epoch+1)
        print("train_loss = ", train_losses)
        print("epoch_list: ", epoch_list)
        figure = plt.figure()
        scores_np = scores.data
        outputs_np = outputs.data
        plt.scatter(scores_np.cpu().numpy(), outputs_np.cpu().numpy(), c='g')
        
        plt.plot([1, 8],[1, 8], 'r--', lw = 2)
        plt.xlabel("Real F_per scores")
        plt.ylabel("Training resulted F_per scores")
        plt.title("Scatter plot of F_per approximator, epoch = {} ".format(epoch+1))
        plt.pause(0.001)
        figure.savefig("scatter_plot_epoch{:d}.png".format(epoch+1))

        if epoch > 5 and epoch % 5 == 0:
            # print the loss plot
            fig_loss = plt.figure()
            plt.plot(epoch_list, train_losses)
            plt.xlabel("Number of Epochs")
            plt.ylabel("Training loss at epoch")
            plt.pause(0.001)
            fig_loss.savefig("training_loss_vs_epoch{:d}.png".format(epoch+1))
        epoch_train_loss = 0.0

            
# To actually train the NN
trainNet(approximator, batch_size = 50, n_epochs = 30, learning_rate = 0.001)
print(time.asctime( time.localtime(time.time()) ))
#approximator.save_state_dict('APXM_4conv.pt')
torch.save(approximator.state_dict, '/home/home2/leichen/SuperResolutor/Approx_discrim/n4conv3.pt')
torch.save(approximator, '/home/home2/leichen/SuperResolutor/Approx_discrim/n4conv3_full.pt')
