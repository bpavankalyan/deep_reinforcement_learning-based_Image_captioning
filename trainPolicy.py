# Training Policy Network


import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import PolicyNetwork

policyNetwork = PolicyNetwork(data["word_to_idx"]).to(device)
#cross entropy loss as loss function
criterion = nn.CrossEntropyLoss().to(device)
# Adam optimizer with learnig rate of 0.0001
optimizer = optim.Adam(policyNetwork.parameters(), lr=0.0001)


batch_size = 100
bestLoss = 0.3
#0.006700546946376562

for epoch in range(100000):
    captions, features, _ = sample_coco_minibatch(small_data, batch_size=batch_size, split='train')
    features = torch.tensor(features, device=device).float().unsqueeze(0)
    captions_in = torch.tensor(captions[:, :-1], device=device).long()
    captions_ou = torch.tensor(captions[:, 1:], device=device).long()
    output = policyNetwork(features, captions_in)
    
    loss = 0
    for i in range(batch_size):
        caplen = np.nonzero(captions[i] == 2)[0][0] + 1
        loss += (caplen/batch_size)*criterion(output[i][:caplen], captions_ou[i][:caplen])
        print("yes") 
    if loss.item() < bestLoss:
        bestLoss = loss.item()
        torch.save(policyNetwork.state_dict(), "./policyNetwork.pt")
        print("epoch:", epoch, "loss:", loss.item())
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
