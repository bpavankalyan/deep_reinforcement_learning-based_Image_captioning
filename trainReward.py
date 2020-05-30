rewardNetwork = RewardNetwork(data["word_to_idx"]).to(device)
optimizer = optim.Adam(rewardNetwork.parameters(), lr=0.001)

# https://cs230-stanford.github.io/pytorch-nlp.html#writing-a-custom-loss-function
def VisualSemanticEmbeddingLoss(visuals, semantics):
    gamma = 0.2
    N, D = visuals.shape
    
    visual_loss=torch.mm(visuals.semantic.t())-torch.diag(torch.mm(visuals.semantics.t())).unsqueeze(1)+ (gamma/N)*(torch.ones((N, N)).to(device) - torch.eye(N).to(device))
    Visual_loss=torch.sum(F.relu(visualloss))/N
    
    Semantic_loss=torch.mm(semantics,visuals.t())-torch.diag(torch.mm(semantics,visuals.t())).unsqueeze(1)+ (gamma/N)*(torch.ones((N, N)).to(device) - torch.eye(N).to(device))
    semantic_loss = torch.sum(F.relu(semanticloss))/N
   
    
    return visual_loss + semantic_loss

batch_size = 50
bestLoss = 10000

for epoch in range(50000):
    captions, features, _ = sample_coco_minibatch(small_data, batch_size=batch_size, split='train')
    features = torch.tensor(features, device=device).float()
    captions = torch.tensor(captions, device=device).long()
    ve, se = rewardNetwork(features, captions)
    loss = VisualSemanticEmbeddingLoss(ve, se)
    
    if loss.item() < bestLoss:
        bestLoss = loss.item()
        torch.save(rewardNetwork.state_dict(), "rewardNetwork.pt")
        print("epoch:", epoch, "loss:", loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    rewardNetwork.rewrnn.hidden_cell.detach_()

def cosine_distance(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def GetRewards(features, captions, model):
    visEmbeds, semEmbeds = model(features, captions)
    rewards = cosine_distance(visEmbeds,semEmbeds)
    return rewards

rewardNet = RewardNetwork(data["word_to_idx"]).to(device)
rewardNet.load_state_dict(torch.load('./rewardNetwork.pt', map_location={'cuda:0': 'cpu'}))
for param in rewardNet.parameters():
    param.require_grad = False
print(rewardNet)

policyNet = PolicyNetwork(data["word_to_idx"]).to(device)
policyNet.load_state_dict(torch.load('./policyNetwork.pt', map_location={'cuda:0': 'cpu'}))
for param in policyNet.parameters():
    param.require_grad = False
print(policyNet)

valueNetwork = ValueNetwork(data["word_to_idx"]).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(valueNetwork.parameters(), lr=0.0001)
valueNetwork.train(mode=True)
