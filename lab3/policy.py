import torch
import torch.nn as nn
import torch.nn.functional as F

# A simple, but generic, policy network with one hidden layer.
class PolicyNet(nn.Module):
    def __init__(self, env, hs):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], hs)
        self.fc2 = nn.Linear(hs, hs)
        self.fc3 = nn.Linear(hs, env.action_space.n)
        self.relu = nn.ReLU()
        
    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        s = F.softmax(self.fc3(s), dim=-1)
        return s