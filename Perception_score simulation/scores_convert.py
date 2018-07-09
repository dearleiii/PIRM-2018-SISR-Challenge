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


# Add row0 at the very beginning 
scores_arr = np.insert(scores_arr, 0, scores_arr[0])
print('scores_arr new shape', scores_arr.shape)
scores_dataframe = pd.DataFrame(scores_arr)
print('DataFrame: ', scores_dataframe)


# Convert numpy.ndarray to torch
scores_torch = torch.from_numpy(scores_arr)
print(scores_torch)
print(scores_torch.size())
