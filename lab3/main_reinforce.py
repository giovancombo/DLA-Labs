# Deep Learning Applications 2023 course, held by Professor Andrew David Bagdanov - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/DLA_Labs/tree/main/lab3

# Code for EXERCISES 3.1, 3.2 and 3.3 using REINFORCE or PPO algorithms.

import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb

from baseREINFORCE import reinforce
from myREINFORCE import Baseline, combo_reinforce, combo_reinforce_baseline


# Choose the algorithm to run: if myREINFORCE is False, baseREINFORCE will be run
myREINFORCE = True
baseline_ok = False

env_name = 'CartPole-v1'     # 'gym_navigation:NavigationGoal-v0' or 'LunarLander-v2' or 'CartPole-v1' ...
wandb_log = True
capture_video = False
render_mode = 'rgb_array'

NUM_EPISODES = 1500
LEARNING_RATE = 1e-3                # baseREINFORCE: 1e-4, myREINFORCE: 1e-4
LR_BASELINE = 1e-3                  # learning rate baseline, implemented only in myREINFORCE
GAMMA = 0.9                        # Discount factor; baseREINFORCE: 0.99, myREINFORCE: 0.99
hidden_size = 128

REWARD_THRESHOLD = 0
AVG_SCORE_THRESHOLD = 400
SCORE_THRESHOLD = 500

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

name_env = 'gymnav' if env_name == 'gym_navigation:NavigationGoal-v0' else env_name
name_alg = 'myREINFORCE' if myREINFORCE else 'myREINFORCEbaseline' if baseline_ok else 'baseREINFORCE'
config = {
    "env_name": env_name,
    "episodes": NUM_EPISODES,
    "lr": LEARNING_RATE,
    "lr_baseline": LR_BASELINE,
    "gamma": GAMMA,
}

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


# Create the environment
env = gym.make(env_name, render_mode = render_mode)
if capture_video:
    os.makedirs(f"videos/{name_env}/{name_alg}_lr{LEARNING_RATE}_gamma{GAMMA}", exist_ok = True)
    env = gym.wrappers.RecordVideo(env, f"videos/{name_env}/{name_alg}_lr{LEARNING_RATE}_gamma{GAMMA}", episode_trigger = lambda t: t % 200 == 0)
env_render = gym.make(env_name, render_mode = 'rgb_array')

policy = PolicyNet(env, hidden_size).to(device)

with wandb.init(project = "DLA_Lab3_DRL", config = config, monitor_gym = True, save_code = True, name = f"{name_alg}-{name_env}_lr{LEARNING_RATE}_gamma{GAMMA}"):
    config = wandb.config

    if myREINFORCE:
        if baseline_ok:
            baseline = Baseline(env.observation_space.shape[0])
            running = combo_reinforce_baseline(policy, baseline, env, name_env, GAMMA, LEARNING_RATE,
                                               LR_BASELINE, NUM_EPISODES, AVG_SCORE_THRESHOLD, SCORE_THRESHOLD,
                                               device='cpu', wandb_log=False)
            plt.plot(running)
        else:
            running = combo_reinforce(policy, env, name_env, gamma = GAMMA, lr = LEARNING_RATE,
                                    episodes = NUM_EPISODES, avg_thresh = AVG_SCORE_THRESHOLD,
                                    scorethresh = SCORE_THRESHOLD, device = device, wandb_log = wandb_log)
            plt.plot(running)

    else:
        running  = reinforce(policy, env, env_render = env_render, name_env = name_env, gamma = GAMMA, lr = LEARNING_RATE,
                            num_episodes = NUM_EPISODES, avg_thresh = AVG_SCORE_THRESHOLD,
                            scorethresh = SCORE_THRESHOLD, device = device, wandb_log = wandb_log)
        plt.plot(running)

env_render.close()
env.close()
