import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
class PolicyNetwork(nn.Module):
    def __init__(self, word_to_idx, input_dimensions=512, wordvec_dimensions=512, hidden_dimensions=512, dtype=np.float32):#building a PolicyNetwork model
        super(PolicyNetwork, self).__init__()
        
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        
        vocab_size = len(word_to_idx)
        print("vocab size",vocab_size)
        
        self.caption_embedding = nn.Embedding(vocab_size, wordvec_dimensions)
        
        self.cnn2linear = nn.Linear(input_dimensions, hidden_dimensions)
        self.lstm = nn.LSTM(wordvec_dimensions, hidden_dimensions, batch_first=True)
        self.linear2vocab = nn.Linear(hidden_dimensions, vocab_size)
        
    def forward(self, features, captions):
        input_captions = self.caption_embedding(captions)
        hidden_initialization = self.cnn2linear(features)
        cell_initialization = torch.zeros_like(hidden_initialization)
        output, _ = self.lstm(input_captions, (hidden_initialization, cell_initialization))
        output = self.linear2vocab(output)
        return output
