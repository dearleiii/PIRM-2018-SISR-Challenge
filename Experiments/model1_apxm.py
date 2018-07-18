import os
from importlib import import_module

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, utils


class APXM_ndf8k3(nn.Module):
    # out batch shape for input x is 3* 1024 * 2048
    def __init__(self, nc=3, ndf=8):
        super(APXM_ndf8k3, self).__init__()        # can be super(CNN), or super(Net), difference?
        self.main = nn.Sequential (
            # Computes the activation of the first convolution
            # Size changes from (3, 1020, 2040) to (18, 510, 2010)
            # input is (nc) x 1024 x 2048
            nn.Conv2d(
                in_channels=nc,             # Input channels = 3            
                out_channels=ndf,           # Output channels = 64 
                kernel_size=3,
                stride = 2,
                padding = 1,               # for same width and length for img after conv2d
                                           
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),                     # activation function

            # 2. state size. (ndf=64) x 512 x 1024
            nn.Conv2d(ndf, ndf * 2, 3, 2, 1),        
            # nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),                     
        
            # 3. state size. (ndf*2=128) x 256 x 512
            nn.Conv2d(ndf * 2, ndf * 2, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 4. state size. (ndf*4=256) x 128 x 256
        )

        self.regressor = nn.Sequential(
            # what's nn.Dropout used for?
            torch.nn.Linear(ndf * 4 * 128 * 256, 256),  
            # what's nn.BatchNormld used for?
            # what's ReLu used for here?
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 1),
        )

    def forward(self, x):
        x = x.float()
        x = self.main(x)

        #Reshape data to input to the input layer of the neural net
        #Size changes from (36, 255, 510) to (1, flatten)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(x.size(0), -1)

        #Computes the activation of the first fully connected layer
        #Size changes from (1, flatten) to (1, 64)
        output = self.regressor(x)
        return output
        
