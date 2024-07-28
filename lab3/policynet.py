# Deep Learning Applications 2023 course, held by Professor Andrew David Bagdanov - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/DLA_Labs/tree/main/lab3

import torch.nn as nn
import torch.nn.functional as F


# Policy network for REINFORCE and DEEP Q-LEARNING algorithms
class PolicyNet(nn.Module):
    def __init__(self, env, hs):
        super(PolicyNet, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], hs),
            nn.ReLU(),
            nn.Linear(hs, hs),
            nn.ReLU(),
            nn.Linear(hs, env.action_space.n))

        self.critic = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], hs),
            nn.ReLU(),
            nn.Linear(hs, hs),
            nn.ReLU(),
            nn.Linear(hs, 1))

    def forward(self, s):
        return F.softmax(self.actor(s), dim=-1)
    
    def get_value(self, s):
        return self.critic(s)
