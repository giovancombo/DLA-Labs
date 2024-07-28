# Deep Learning Applications 2023 course, held by Professor Andrew David Bagdanov - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/DLA_Labs/tree/main/lab3

# Code for EXERCISES 3.1, 3.2 and 3.3 using REINFORCE or PPO algorithms.

import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
import yaml

from policynet import PolicyNet
from base_reinforce import reinforce
from my_reinforce import combo_reinforce, combo_reinforce_with_baseline
from my_ppo import PPOAgent, PPOTrainer, train_ppo
from my_ppo2 import Agent, train_ppo2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Choose the algorithm to run
base_reinforce = True
my_reinforce = False
my_ppo = False
my_ppo2 = False

# Load parameters from YAML file
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)
name = 'gymnavigation' if params['env_name'] == 'gym_navigation:NavigationGoal-v0' else params['env_name']

# Create the environment
env = gym.make(params['env_name'], render_mode = 'rgb_array' if params['capture_video'] else None)
if params['capture_video']:
    env = gym.wrappers.RecordVideo(env, f"videos/{name}/video.mp4", episode_trigger = lambda t: t % 25 == 0)
env_render = gym.make(params['env_name'], render_mode = 'human')

policy = PolicyNet(env, params['hidden_size']).to(device)


with wandb.init(project = "DLA_Lab3_DRL", config = params, monitor_gym = True, save_code = True):
    config = wandb.config

    if base_reinforce:
        running  = reinforce(policy, env, env_render, device = device, lr = params['lr'],
                             num_episodes = params['episodes'], wandb_log = params['wandb_log'])
        plt.plot(running)
        torch.save(policy, f"checkpoints/basereinforce_{name}_lr{params['lr']}_gamma{params['gamma']}.pth")

    elif my_reinforce:
        # Alternative: combo_reinforce_with_baseline, for REINFORCE algorithm with baseline
        running = combo_reinforce(env, policy, lr = params['lr'], gamma = params['gamma'],
                                  episodes = params['episodes'], device = device, wandb_log = params['wandb_log'])
        plt.plot(running)
        torch.save(policy, f"checkpoints/comboreinforce_{name}_lr{params['lr']}_gamma{params['gamma']}.pth")

    elif my_ppo:
        agent = PPOAgent(env.observation_space.shape[0], env.action_space.n, params['hidden_size']).to(device)
        ppo = PPOTrainer(agent, policy_lr = params['policy_lr'],
                                value_lr = params['value_lr'],
                                target_kl_div = params['target_kl_div'],
                                max_policy_train_iters = params['max_policy_train_iters'],
                                value_train_iters = params['value_train_iters'])
    
        train_ppo(env, agent, ppo, params['episodes'], params['max_steps'], params['log_freq'], device,
                    params['wandb_log'], params)
        
        torch.save(agent, f"checkpoints/comboppo_{name}_plr{params['policy_lr']}_vlr{params['value_lr']}_kl{params['target_kl_div']}.pth")

    elif my_ppo2:
        agent = Agent(n_actions = env.action_space.n, batch_size = params['batch_size'], alpha = params['lr'],
                      N = params['N'], n_epochs = params['n_epochs'], input_dims = env.observation_space.shape)
        
        train_ppo2(env, agent, params['N'], params['n_games'], params['wandb_log'])

        torch.save(agent, f"checkpoints/comboppo2_{name}_bs{params['batch_size']}_n{params['N']}_lr{params['lr']}.pth")

env_render.close()
env.close()
