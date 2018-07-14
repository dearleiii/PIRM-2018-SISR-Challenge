import os
from importlib import import_module

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, utils

#from srdata import SRData 
import srdata

transformed_dataset = srdata.SRData(score_file = 'F_per_arr.mat',
                                      root_dir = '/usr/project/xtmp/superresoluter/approximater_training_set/',
                                      subdir_edsr = 'EDSR/DIV2K_train/',
                                      subdir_enhanced = 'EhanceNet/DIV2K_train/',
                                      subdir_hr = 'HR/DIV2K_train/',
                                      transform=transforms.Compose([
                                          srdata.Rescale((1024, 2048)),
                                          srdata.ToTensor()
                                      ]))

class APXM_4conv(nn.Module):
    # out batch shape for input x is 3* 1024 * 2048
    def __init__(self, nc=3, ndf=64):
        super(APXM_4conv, self).__init__()        # can be super(CNN), or super(Net), difference?
        self.main = nn.Sequential (
            # Computes the activation of the first convolution
            # Size changes from (3, 1020, 2040) to (18, 510, 2010)
            # input is (nc) x 1024 x 2048
            nn.Conv2d(
                in_channels=nc,             # Input channels = 3            
                out_channels=ndf,           # Output channels = 64 
                kernel_size=4,
                stride = 2,
                padding = 1,               # for same width and length for img after conv2d
                                           
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),                     # activation function

            # 2. state size. (ndf=64) x 512 x 1024
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),        
            # nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),                     
        
            # 3. state size. (ndf*2=128) x 256 x 512
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 4. state size. (ndf*4=256) x 128 x 256
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 5. state size. (ndf*8=512) x 64 x 128
        )

        self.regressor = nn.Sequential(
            # what's nn.Dropout used for?
            torch.nn.Linear(ndf * 8 * 64 * 128, 1024),  
            # what's nn.BatchNormld used for?
            # what's ReLu used for here?
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, 256),
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
        return output, x
        

approximator = APXM_4conv()
print(approximator)

## Data loader
def get_train_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(transformed_dataset, batch_size = batch_size,
                                               shuffle=True, num_workers=4 )# sampler=train_sampler, num_workers=2)
    return(train_loader)

batch_size = 100
train_loader = get_train_loader(batch_size)
n_batches = len(train_loader)    
