# Deep Learning Applications 2023 course, held by Professor Andrew David Bagdanov - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/DLA_Labs/tree/main/lab3

# My implementation of the Deep Q-Learning algorithm

import torch
import torch.nn as nn
import gymnasium as gym
from collections import deque
import itertools
import numpy as np
import random


def combo_dqn():    
    GAMMA = 0.99                    # Discount rate for computing TD target
    LEARNING_RATE = 5e-4
    BATCH_SIZE = 32                 # Number of experiences to sample in each training step from the buffer
    BUFFER_SIZE = 50000             # Max number of experiences stored
    MIN_REPLAY_SIZE = 1000          # Min number of experiences stored before computind gradients and doing training
    EPSILON_START = 1.0
    EPSILON_END = 0.02
    EPSILON_DECAY = 10000           # Number of steps to decay epsilon from EPSILON_START to EPSILON_END
    TARGET_UPDATE_FREQ = 1000

    env_name = 'CartPole-v1'


    class DQN(nn.Module):
        def __init__(self, env):
            super().__init__()

            in_features = int(np.prod(env.observation_space.shape))
            self.net = nn.Sequential(
                nn.Linear(in_features, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, env.action_space.n))
        
        def forward(self, x):
            return self.net(x)
        
        # Selecting action with greedy approach
        def act(self, obs):
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            q_values = self(obs_t.unsqueeze(0))
            max_q_index = torch.argmax(q_values, dim=1)[0]
            action = max_q_index.detach().item()
            return action

    env = gym.make(env_name)

    replay_buffer = deque(maxlen = BUFFER_SIZE)
    rew_buffer = deque([0.0], maxlen = 100)
    episode_reward = 0.0

    online_net = DQN(env)
    target_net = DQN(env)
    # Initialize target network with the same weights of the online network
    target_net.load_state_dict(online_net.state_dict())
    optimizer = torch.optim.Adam(online_net.parameters(), lr = LEARNING_RATE)

    # Initialize Replay Buffer
    obs, _ = env.reset()
    for _ in range(MIN_REPLAY_SIZE):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        transition = (obs, action, reward, done, next_obs)
        replay_buffer.append(transition)
        obs = next_obs

        if done:
            obs, _ = env.reset()

    # Training Loop
    obs, _ = env.reset()
    for step in itertools.count():
        # Implementing epsilon-greedy policy
        epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
        if random.random() <= epsilon:
            action = env.action_space.sample()
        else:
            action = online_net.act(obs)
            
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        transition = (obs, action, reward, done, next_obs)
        replay_buffer.append(transition)
        obs = next_obs
        episode_reward += reward

        if done:
            obs, _ = env.reset()
            rew_buffer.append(episode_reward)
            episode_reward = 0.0

    # Start Gradient Step
    transitions = random.sample(replay_buffer, BATCH_SIZE)

    # Put in a list every information about the transitions (transition = experience)
    obses = np.asarray([t[0] for t in transitions])
    actions = np.asarray([t[1] for t in transitions])
    rews = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    new_obses = np.asarray([t[4] for t in transitions])

    obses_t = torch.as_tensor(obses, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

    # Compute Targets
    target_q_values = target_net(new_obses_t)
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
    targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

    # Compute Loss
    q_values = online_net(obses_t)
    action_q_values = torch.gather(input = q_values, dim = 1, index = actions_t)
    loss = nn.functional.smooth_l1_loss(action_q_values, targets)

    # Gradient Descent Step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update Target Network parameters
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    # Logging
    if step % 50 == 0:
        print(f'Step {step}: Avg Reward: {np.mean(rew_buffer)}')


if __name__ == '__main__':
    combo_dqn()
