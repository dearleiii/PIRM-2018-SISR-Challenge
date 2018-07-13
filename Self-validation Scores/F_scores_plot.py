
# coding: utf-8

# In[5]:


import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import colors 

import scipy.io as sio


# In[9]:


edsr_content = sio.loadmat('EDSR_arr.mat')


# In[10]:


edsr_arr = edsr_content['scores_tot']
edsr_arr


# In[32]:


colors = ['skyblue', 'tan', 'lime']

plt.hist(edsr_arr, bins = 40, histtype = 'bar', lw = 1, ec = "tan", label = colors)
plt.title('Histogram of EDSR F_perceptual score distribution')
plt.legend(prop = {'size': 10})
plt.grid(True)
plt.show()


# In[36]:


f_per = sio.loadmat('F_per_arr.mat')
f_per = f_per['scores_tot']
f_per


# In[42]:


colors = ['skyblue', 'red', 'lime']
dataset_name = ['EDSR', 'HR', 'EnhancedNet']
plt.hist(f_per, bins = 40, histtype = 'bar', label = dataset_name)

plt.title('Histogram of F_perceptual score distribution')
plt.legend(prop = {'size': 10})
plt.grid(True)
plt.show()


# In[43]:


f_ma = sio.loadmat('F_ma_arr.mat')
f_ma = f_ma['scores_tot']
f_ma


# In[47]:


#colors = ['skyblue', 'red', 'lime']
dataset_name = ['EDSR', 'HR', 'EnhancedNet']
plt.hist(f_ma, bins = 40, histtype = 'bar', label = dataset_name)

plt.title('Histogram of F_ma score distribution')
plt.legend(prop = {'size': 10})
plt.grid(True)
plt.show()


# In[45]:


f_niqe = sio.loadmat('F_NIQE_arr.mat')
f_niqe = f_niqe['scores_tot']
f_niqe


# In[48]:


dataset_name = ['EDSR', 'HR', 'EnhancedNet']
plt.hist(f_niqe, bins = 40, histtype = 'bar', label = dataset_name)

plt.title('Histogram of F_niqe score distribution')
plt.legend(prop = {'size': 10})
plt.grid(True)
plt.show()

