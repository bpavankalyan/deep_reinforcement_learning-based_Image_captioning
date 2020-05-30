''''

After individual training of each networks we usea2c for joint training.

''''



class AdvantageActorCriticNetwork(nn.Module):
    def __init__(self, valueNet, policyNet):
        super(AdvantageActorCriticNetwork, self).__init__()

        self.valueNet = valueNet #RewardNetwork(data["word_to_idx"]).to(device)
        self.policyNet = policyNet #PolicyNetwork(data["word_to_idx"]).to(device)

    def forward(self, features, captions):
        # Get value from value network
        values = self.valueNet(features, captions)
        # Get action probabilities from policy network
        probs = self.policyNet(features.unsqueeze(0), captions)[:, -1:, :]        
        return values, probs


rewardNet = RewardNetwork(data["word_to_idx"]).to(device)
policyNet = PolicyNetwork(data["word_to_idx"]).to(device)
valueNet = ValueNetwork(data["word_to_idx"]).to(device)

rewardNet.load_state_dict(torch.load('./rewardNetwork.pt', map_location={'cuda:0': 'cpu'}))
policyNet.load_state_dict(torch.load('./policyNetwork.pt', map_location={'cuda:0': 'cpu'}))
valueNet.load_state_dict(torch.load('./valueNetwork.pt', map_location={'cuda:0': 'cpu'}))

a2cNetwork = AdvantageActorCriticNetwork(valueNet, policyNet)
optimizer = optim.Adam(a2cNetwork.parameters(), lr=0.0001)
