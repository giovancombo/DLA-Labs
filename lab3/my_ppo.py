# Deep Learning Applications 2023 course, held by Professor Andrew David Bagdanov - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/DLA_Labs/tree/main/lab3

# An Off-the-Shelf implementation of the Proximal Policy Optimization algorithm

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import wandb


PPO_CLIP_VAL = 0.2
TARGET_KL_DIV = 0.01
MAX_POLICY_TRAIN_ITERS = 80
VALUE_TRAIN_ITERS = 80
POLICY_LR = 3e-4
VALUE_LR = 1e-2
GAMMA = 0.99
DECAY = 0.97
MAX_STEPS = 1000

def layer_init(layer, std = np.sqrt(2), bias_const = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

# PPO Agent
class PPOAgent(nn.Module):
  def __init__(self, obs_space_size, action_space_size, hidden_size):
    super().__init__()

    self.shared_layers = nn.Sequential(
        layer_init(nn.Linear(obs_space_size, hidden_size)),
        nn.ReLU(),
        layer_init(nn.Linear(hidden_size, hidden_size)),
        nn.ReLU())
    
    self.policy_layers = nn.Sequential(
        layer_init(nn.Linear(hidden_size, hidden_size)),
        nn.ReLU(),
        layer_init(nn.Linear(hidden_size, action_space_size)))
    
    self.value_layers = nn.Sequential(
        layer_init(nn.Linear(hidden_size, hidden_size)),
        nn.ReLU(),
        layer_init(nn.Linear(hidden_size, 1), std = 1.))
    
  def value(self, obs):
    z = self.shared_layers(obs)
    value = self.value_layers(z)
    return value
        
  def policy(self, obs):
    z = self.shared_layers(obs)
    policy_logits = self.policy_layers(z)
    return policy_logits

  def forward(self, obs):
    z = self.shared_layers(obs)
    policy_logits = self.policy_layers(z)
    value = self.value_layers(z)
    return policy_logits, value


class PPOTrainer():
  def __init__(self, actor_critic, ppo_clip_val = PPO_CLIP_VAL, target_kl_div = TARGET_KL_DIV,
                                   max_policy_train_iters = MAX_POLICY_TRAIN_ITERS,
                                   value_train_iters = VALUE_TRAIN_ITERS, policy_lr = POLICY_LR, value_lr = VALUE_LR):
    self.ac = actor_critic
    self.ppo_clip_val = ppo_clip_val
    self.target_kl_div = target_kl_div
    self.max_policy_train_iters = max_policy_train_iters
    self.value_train_iters = value_train_iters

    policy_params = list(self.ac.shared_layers.parameters()) + list(self.ac.policy_layers.parameters())
    self.policy_optim = torch.optim.Adam(policy_params, lr = policy_lr)

    value_params = list(self.ac.shared_layers.parameters()) + list(self.ac.value_layers.parameters())
    self.value_optim = torch.optim.Adam(value_params, lr = value_lr)

  def train_policy(self, obs, acts, old_log_probs, gaes):
    for _ in range(self.max_policy_train_iters):
      self.policy_optim.zero_grad()

      new_logits = self.ac.policy(obs)
      new_logits = Categorical(logits=new_logits)
      new_log_probs = new_logits.log_prob(acts)

      policy_ratio = torch.exp(new_log_probs - old_log_probs)
      clipped_ratio = policy_ratio.clamp(
          1 - self.ppo_clip_val, 1 + self.ppo_clip_val)
      
      clipped_loss = clipped_ratio * gaes
      full_loss = policy_ratio * gaes
      policy_loss = -torch.min(full_loss, clipped_loss).mean()

      policy_loss.backward()
      self.policy_optim.step()

      kl_div = (old_log_probs - new_log_probs).mean()
      if kl_div >= self.target_kl_div:
        break

  def train_value(self, obs, returns):
    for _ in range(self.value_train_iters):
      self.value_optim.zero_grad()

      values = self.ac.value(obs)
      value_loss = (returns - values) ** 2
      value_loss = value_loss.mean()

      value_loss.backward()
      self.value_optim.step()


def discount_rewards(rewards, gamma = GAMMA):
    """
    Return discounted rewards based on the given rewards and gamma param.
    """
    new_rewards = [float(rewards[-1])]
    for i in reversed(range(len(rewards)-1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
    return np.array(new_rewards[::-1])


def calculate_gaes(rewards, values, gamma = GAMMA, decay = DECAY):
    """
    Return the General Advantage Estimates from the given rewards and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    next_values = np.concatenate([values[1:], [0]])
    deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas) - 1)):
        gaes.append(deltas[i] + decay * gamma * gaes[-1])

    return np.array(gaes[::-1])


def rollout(model, env, device, max_steps = MAX_STEPS):
    """
    Performs a single rollout.
    Returns training data in the shape (n_steps, observation_shape)
    and the cumulative reward.
    """
    ### Create data storage
    train_data = [[], [], [], [], []] # obs, act, reward, values, act_log_probs
    obs, _ = env.reset()

    ep_reward = 0
    for _ in range(max_steps):
        logits, val = model(torch.tensor(obs, dtype = torch.float32, device = device))
        act_distribution = Categorical(logits = logits)
        act = act_distribution.sample()
        act_log_prob = act_distribution.log_prob(act).item()

        act, val = act.item(), val.item()
        next_obs, reward, terminated, truncated, _ = env.step(act)
        done = terminated or truncated

        for i, item in enumerate((obs, act, reward, val, act_log_prob)):
          train_data[i].append(item)

        obs = next_obs
        ep_reward += reward
        if done:
            break

    train_data = [np.asarray(x) for x in train_data]
    ### Do train data filtering
    train_data[3] = calculate_gaes(train_data[2], train_data[3])

    return train_data, ep_reward


# Training loop
def train_ppo(env, model, ppo_trainer, episodes, max_steps, log_freq, device, wandb_log, config):

    if wandb_log:
      wandb.init(project = 'DLA_Lab3_DRL', config = config, name = f"myPPO_{env.spec.id}_plr{POLICY_LR}_vlr{VALUE_LR}_gamma{GAMMA}")
      config = wandb.config
      wandb.watch(model, log = 'all')

    ep_rewards = []
    for episode_idx in range(episodes):
        # Perform rollout
        train_data, reward = rollout(model, env, device, max_steps=max_steps)
        ep_rewards.append(reward)
        # Shuffle
        permute_idxs = np.random.permutation(len(train_data[0]))
        # Policy data
        obs = torch.tensor(train_data[0][permute_idxs], dtype=torch.float32, device=device)
        acts = torch.tensor(train_data[1][permute_idxs], dtype=torch.int32, device=device)
        gaes = torch.tensor(train_data[3][permute_idxs], dtype=torch.float32, device=device)
        act_log_probs = torch.tensor(train_data[4][permute_idxs], dtype=torch.float32, device=device)
        # Value data
        returns = discount_rewards(train_data[2])[permute_idxs]
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        # Train model
        ppo_trainer.train_policy(obs, acts, act_log_probs, gaes)
        ppo_trainer.train_value(obs, returns)

        if wandb_log:
          wandb.log({"score": reward}, step = episode_idx)

        if (episode_idx + 1) % log_freq == 0:
            print('Episode {} | Avg Reward {:.1f}'.format(episode_idx + 1, np.mean(ep_rewards[-log_freq:])))
    
    if wandb_log:
      wandb.unwatch(model)
      wandb.finish()