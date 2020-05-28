import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torchvision
from torchsummary import summary

class ValueNetworkRNN(nn.Module):
    def __init__(self, word_to_idx, input_dimensions=512, wordvec_dimensions=512, hidden_dimensions=512, dtype=np.float32):
        super(ValueNetworkRNN, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self. hidden_dimensions =  hidden_dimensions
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        vocab_size = len(word_to_idx)
        
        self.hidden_cell = (torch.zeros(1, 1, self. hidden_dimensions).to(device), torch.zeros(1, 1, self. hidden_dimensions).to(device))
        
        self.caption_embedding = nn.Embedding(vocab_size, wordvec_dimensions)
        self.lstm = nn.LSTM(wordvec_dimensions, hidden_dimensions)
        
    def forward(self, captions):
        input_captions = self.caption_embedding(captions)
        output, self.hidden_cell = self.lstm(input_captions.view(len(input_captions) ,1, -1), self.hidden_cell)
        return output
    
class ValueNetwork(nn.Module):
    def __init__(self, word_to_idx):
        super(ValueNetwork, self).__init__()
        self.valuernn = ValueNetworkRNN(word_to_idx)
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 1)
    
    def forward(self, features, captions):
        for t in range(captions.shape[1]):
            valuernn = self.valrnn(captions[:, t])
        valuernn = valuernn.squeeze(0).squeeze(1)
        # concatenation of both image and caption
        state = torch.cat((features, valuernn), dim=1)
        # linear 1 respresents image feature of 512
        output = self.linear1(state)
        output = self.linear2(output)
        return output
