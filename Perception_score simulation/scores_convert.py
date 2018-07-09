"""
This file take the .mat file of F_per scores
preprocess as input to CNN training 
"""

import scipy.io as sio
import numpy as np
import torch

mat_content = sio.loadmat('F_per_Enhancednet-DIV2K80.mat')
scores_arr = mat_content['F_per_array']

print(scores_arr.shape) 

# Convert numpy.ndarray to torch
scores_torch = torch.from_numpy(scores_arr)
print(scores_torch)
print(scores_torch.size())
