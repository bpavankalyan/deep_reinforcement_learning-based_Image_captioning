from __future__ import print_function  #you can slowly be accustomed to incompatible changes or to such ones introducing new keywords.
import time, os, json
import numpy as np    #used for working with arrays.
import matplotlib.pyplot as plt #Its functions manipulate elements of a figure, such as creating a figure, creating a plotting area, plotting lines, adding plot labels, etc. 
import nltk  #The Natural Language Toolkit (NLTK) is a Python package for natural language processing.
import random

import numpy as np
#  PyTorch has two main features: Tensor computation (like NumPy) with strong GPU acceleration with cuda
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F

 import load_coco_data, sample_coco_minibatch, decode_captions
 import image_from_url

from torchsummary import summary

%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
%load_ext autoreload
%autoreload 2

#cuda:use GPU to make the neural network training much more faster.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Working on: ", device)

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

max_seq_len = 17

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


import metrics
from PolicyNetwork import PolicyNetwork
from ValueNetwork import ValueNetwork,ValueNetworkRNN
from RewardNetwork import RewardNetwork,RewardNetworkRNN


import h5py
filename = "./coco2014_captions.h5"

with h5py.File(filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]
    print(f[a_group_key])

    # Get the data
    data1 = list(f[a_group_key])



# Load COCO data from disk; this returns a dictionary
# We'll work with dimensionality-reduced features for this notebook, but feel
# free to experiment with the original features by changing the flag below.
data = load_coco_data(pca_features=True)

data["train_captions_lens"] = np.zeros(data["train_captions"].shape[0])
#creating a numpy array all assigned to 0 of length of number of train_captions with train_caption_lens as key
data["val_captions_lens"] = np.zeros(data["val_captions"].shape[0])


for i in range(data["train_captions"].shape[0]):
    data["train_captions_lens"][i] = np.nonzero(data["train_captions"][i] == 2)[0][0] + 1
    #returns the length of the train captions
for i in range(data["val_captions"].shape[0]):
    data["val_captions_lens"][i] = np.nonzero(data["val_captions"][i] == 2)[0][0] + 1
    #returns the length of the validation captions


# Print out all the keys and values from the data dictionary
for k, v in data.items():
    if type(v) == np.ndarray:
        print(k, type(v), v.shape, v.dtype)
    else:
        print(k, type(v), len(v))
