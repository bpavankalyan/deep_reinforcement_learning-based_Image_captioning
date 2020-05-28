import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torchvision
class RewardNetworkRNN(nn.Module):
    def __init__(self, word_to_idx, input_dimensions=512, wordvec_dimensions=512, hidden_dimensions=512, dtype=np.float32):#building a RewardNetworkRNN model
        super(RewardNetworkRNN, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.hidden_dimensions = hidden_dimensions
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        vocab_size = len(word_to_idx)
        
        self.hidden_cell = torch.zeros(1, 1, self.hidden_dimensions).to(device)
        
        self.caption_embedding = nn.Embedding(vocab_size, wordvec_dim)
        self.gru = nn.GRU(wordvec_dimensions, hidden_dimensions)
    
    def forward(self, captions):
        input_captions = self.caption_embedding(captions)
        output, self.hidden_cell = self.gru(input_captions.view(len(input_captions) ,1, -1), self.hidden_cell)
        return output
    
class RewardNetwork(nn.Module):
    def __init__(self, word_to_idx):
        super(RewardNetwork, self).__init__()
        self.rewardrnn = RewardNetworkRNN(word_to_idx)
        self.visual_embedding = nn.Linear(512, 512)
        self.semantic_embedding = nn.Linear(512, 512)
        
    def forward(self, features, captions):
        for t in range(captions.shape[1]):
            rewrnn = self.rewardrnn(captions[:, t])
        rewrnn = rewrnn.squeeze(0).squeeze(1)
        #embedding  of semantic 
        se = self.semantic_embedding(rewrnn)
        ve = self.visual_embeddding(features)
        return ve, se
