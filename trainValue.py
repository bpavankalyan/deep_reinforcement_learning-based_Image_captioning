batch_size = 50
bestLoss = 10000
max_seq_len = 17

for epoch in range(50000):
    captions, features, _ = sample_coco_minibatch(small_data, batch_size=batch_size, split='train')
    features = torch.tensor(features, device=device).float()
    
    # Generate captions using the policy network
    captions = GenerateCaptions(features, captions, policyNet)
    
    # Compute the reward of the generated caption using reward network
    rewards = GetRewards(features, captions, rewardNet)
    
    # Compute the value of a random state in the generation process
#     print(features.shape, captions[:, :random.randint(1, 17)].shape)
    values = valueNetwork(features, captions[:, :random.randint(1, 17)])
    
    # Compute the loss for the value and the reward
    loss = criterion(values, rewards)
    
    if loss.item() < bestLoss:
        bestLoss = loss.item()
        torch.save(valueNetwork.state_dict(), "valueNetwork.pt")
        print("epoch:", epoch, "loss:", loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    valueNetwork.valrnn.hidden_cell[0].detach_()
    valueNetwork.valrnn.hidden_cell[1].detach_()
    rewardNet.rewrnn.hidden_cell.detach_()

