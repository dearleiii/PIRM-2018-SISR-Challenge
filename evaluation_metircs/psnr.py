from imageio import imread, imwrite
import numpy as np
import sys, os
import torch
from torch import nn
from bicubic import BicubicDownSample
import math

phase_base_dir = '../edsr_bases/'
phase_test_dir = '../demo/'
hr_dir = '../hr/'
lr_dir = '../lr/'
img_name = '0833'
hr_img = imread(os.path.join(hr_dir, img_name+ '.png')).astype(np.float32)
lr_img = imread(os.path.join(lr_dir, img_name+ '.png')).astype(np.float32)
test_img = imread(os.path.join(phase_test_dir, img_name+ '_hr_145.png')).astype(np.floa\
t32)

hr_tensor = torch.tensor(hr_img.reshape(1, *hr_img.shape)).type('torch.DoubleTensor')
lr_tensor = torch.tensor(lr_img.reshape(1, *lr_img.shape)).type('torch.DoubleTensor')
test_tensor = torch.tensor(test_img.reshape(1, *test_img.shape)).type('torch.DoubleTens\
or')

bds = BicubicDownSample()
ds_hr_tensor = bds(hr_tensor, nhwc=True)

l = nn.MSELoss()
ls_loss = l(ds_hr_tensor, lr_tensor)
# hr_loss = l(base_tensor, hr_tensor)
# print(ls_loss): 0.1046

diff = test_tensor - hr_tensor
diff_div = diff.div(255)
#print(diff_div)

mse = diff_div.pow(2).mean()
print(mse)

psnr = -10 * math.log10(mse)
print(psnr)
