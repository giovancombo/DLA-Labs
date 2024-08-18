# Deep Learning Applications 2023 course, held by Professor Andrew David Bagdanov - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/DLA_Labs/tree/main/lab3

# My implementation of the Deep Q-Learning algorithm

import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
from collections import deque, namedtuple
import wandb


env_name = 'CartPole-v1'     # 'gym_navigation:NavigationGoal-v0' or 'LunarLander-v2' or 'CartPole-v1' ...
wandb_log = True
render_mode = None

SCORE_THRESHOLD = 501           # CartPole: 500; LunarLander: 250; gymnav: 150
AVG_SCORE_THRESHOLD = 195       # CartPole: 105; LunarLander: 200; gymnav: 100

NUM_EPISODES = 3000
LEARNING_RATE = 1e-2
START_EPS = 1.0                 # the starting value of epsilon
MIN_EPS = 0.001
EPS_DECAY = 0.995               # controls the rate of exponential decay of epsilon, higher means a slower decay
GAMMA = 0.99                    # Discount Factor
TAU = 1e-3
BATCH_SIZE = 128                # number of transitions randomly sampled from the replay buffer
BUFFER_SIZE = 100000
UPDATE_TARGET = 4               # how often to update the target network
hidden_sizes = [64, 64]

name_env = 'gymnav' if env_name == 'gym_navigation:NavigationGoal-v0' else env_name
config = {
    "env_name": env_name,
    "episodes": NUM_EPISODES,
    "eps_decay": EPS_DECAY,
    "gamma": GAMMA,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "update_target": UPDATE_TARGET,
}

PATH = f'./checkpoints/{name_env}/myDQN_eps{EPS_DECAY}_lr{LEARNING_RATE}_gamma{GAMMA}' + str(time.time())[-6:]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Define the Deep Q-Learning Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=[64, 64]):
        super(QNetwork, self).__init__()

        layers = [nn.Linear(state_size, hidden_sizes[0]), nn.ReLU()]
        for i in range(len(hidden_sizes) - 1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]), nn.ReLU()])
        layers.append(nn.Linear(hidden_sizes[-1], action_size))

        self.model = nn.Sequential(*layers)

    def forward(self, state):
        return self.model(state)
    
class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, device):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class Agent():
    def __init__(self, state_size, action_size, hidden_sizes=[64, 64], learning_rate=5e-4, 
                 buffer_size=100000, batch_size=64, gamma=0.99, tau=1e-3, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        # Q-Networks
        self.qnetwork_local = QNetwork(state_size, action_size, hidden_sizes).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, hidden_sizes).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, device)
        
        self.t_step = 0
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_TARGET
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = nn.functional.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


if wandb_log:
    wandb.login()
    wandb.init(project = "DLA_Lab3_DRL", monitor_gym = True, config = config, save_code = True, name = f"myDQN-{name_env}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_eps{EPS_DECAY}_gamma{GAMMA}")
    config = wandb.config

print("START Deep Q-Learning")

env = gym.make(env_name, render_mode = render_mode)
agent = Agent(env.observation_space.shape[0], env.action_space.n,
              hidden_sizes, LEARNING_RATE, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, device)
eps = START_EPS

solved = False
last_scores = []
partial_best_score = float('-inf')
total_best_score = float('-inf')
os.makedirs(PATH, exist_ok=True)

for episode in range(NUM_EPISODES):
    done = False
    state, _ = env.reset()
    rewards = []
    score = 0
    while not done:
        action = agent.act(state, eps)
        next_state, reward, truncated, terminated, _ = env.step(action)
        done = terminated or truncated
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        rewards.append(reward)
        if done:
            break

    eps = max(MIN_EPS, EPS_DECAY*eps)       # decrease epsilon

    last_scores.append(score)
    if len(last_scores) > 100:
        last_scores.pop(0)
    avg_score = sum(last_scores) / len(last_scores)

    if avg_score >= AVG_SCORE_THRESHOLD and len(last_scores) == 100 and not solved:
        solved = True
        torch.save(agent.qnetwork_local.state_dict(), PATH + f'/solved_episode{episode}.pth')
        print(f"ENVIRONMENT SOLVED in {episode} episodes!!!")
        print(f"Checkpoint saved at episode {episode} with average score of {avg_score:.2f}")
        break

    if score > SCORE_THRESHOLD and score > partial_best_score:
        partial_best_score = score
        torch.save(agent.qnetwork_local.state_dict(), PATH + f'/episode{episode}.pth')
        if score > total_best_score:
            total_best_score = score
            torch.save(agent.qnetwork_local.state_dict(), PATH + f'/best_score_model.pth')
        print(f"New best model saved at episode {episode}")

    if episode % 500 == 0:
        partial_best_score = float('-inf')

    print(f"Episode {episode} - {len(rewards)} steps. Score: {score:.2f}; Last 100 runs avg score: {avg_score:.2f}")
    if wandb_log:
        wandb.log({"score": score}, step = episode)

torch.save(agent.qnetwork_local.state_dict(), PATH + f'/episode{episode}.pth')

if wandb_log:
    wandb.finish()

env.close()
print('COMPLETE')
