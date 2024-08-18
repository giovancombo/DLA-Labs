# Deep Learning Applications 2023 course, held by Professor Andrew David Bagdanov - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/DLA_Labs/tree/main/lab3

# An Off-the-Shelf implementation of the Proximal Policy Optimization algorithm

import os
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import wandb

env_name = 'LunarLander-v2'     # 'gym_navigation:NavigationGoal-v0' or 'LunarLander-v2' or 'CartPole-v1' ...
wandb_log = True
render_mode = None

SCORE_THRESHOLD = 250
AVG_SCORE_THRESHOLD = 200

NUM_EPISODES = 2000
PPO_CLIP_VAL = 0.2
TARGET_KL_DIV = 0.01
MAX_POLICY_TRAIN_ITERS = 15
VALUE_TRAIN_ITERS = 15
POLICY_LR = 3e-4
VALUE_LR = 5e-4
GAMMA = 0.99
DECAY = 0.95

name_env = 'gymnav' if env_name == 'gym_navigation:NavigationGoal-v0' else env_name
config = {
    "env_name": env_name,
    "episodes": NUM_EPISODES,
    "policy_lr": POLICY_LR,
    "value_lr": VALUE_LR,
    "gamma": GAMMA,
    "decay": DECAY,
    "ppo_clip": PPO_CLIP_VAL,
    "kl_div": TARGET_KL_DIV,
}

PATH = f"checkpoints/{name_env}/myPPO_plr{POLICY_LR}_vlr{VALUE_LR}_kldiv{TARGET_KL_DIV}_gamma{GAMMA}_decay{DECAY}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# PPO Agent
class PPOAgent(nn.Module):
  def __init__(self, obs_space_size, action_space_size, hidden_size):
    super().__init__()

    self.shared_layers = nn.Sequential(
        self.layer_init(nn.Linear(obs_space_size, hidden_size)),
        nn.ReLU(),
        self.layer_init(nn.Linear(hidden_size, hidden_size)),
        nn.ReLU())
    
    self.policy_layers = nn.Sequential(
        self.layer_init(nn.Linear(hidden_size, hidden_size)),
        nn.ReLU(),
        self.layer_init(nn.Linear(hidden_size, action_space_size)))
    
    self.value_layers = nn.Sequential(
        self.layer_init(nn.Linear(hidden_size, hidden_size)),
        nn.ReLU(),
        self.layer_init(nn.Linear(hidden_size, 1), std = 1.))
    
  @staticmethod
  def layer_init(layer, std = np.sqrt(2), bias_const = 0.0):
      nn.init.orthogonal_(layer.weight, std)
      nn.init.constant_(layer.bias, bias_const)
      return layer
    
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
  def __init__(self, actor_critic, plr = 3e-4, vlr = 3e-4, ppo_clip_val = PPO_CLIP_VAL, target_kl_div = TARGET_KL_DIV,
                                   max_policy_train_iters = MAX_POLICY_TRAIN_ITERS,
                                   value_train_iters = VALUE_TRAIN_ITERS):
    self.ac = actor_critic
    self.ppo_clip_val = ppo_clip_val
    self.target_kl_div = target_kl_div
    self.max_policy_train_iters = max_policy_train_iters
    self.value_train_iters = value_train_iters

    policy_params = list(self.ac.shared_layers.parameters()) + list(self.ac.policy_layers.parameters())
    self.policy_optim = torch.optim.Adam(policy_params, lr = plr)

    value_params = list(self.ac.shared_layers.parameters()) + list(self.ac.value_layers.parameters())
    self.value_optim = torch.optim.Adam(value_params, lr = vlr)

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

def rollout(model, env, device):
    """
    Performs a single rollout.
    Returns training data in the shape (n_steps, observation_shape)
    and the cumulative reward.
    """
    ### Create data storage
    train_data = [[], [], [], [], []] # obs, act, reward, values, act_log_probs
    obs, _ = env.reset()
    score = 0
    done = False
    while not done:
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
      score += reward

      if done:
          break

    train_data = [np.asarray(x) for x in train_data]
    ### Do train data filtering
    train_data[3] = calculate_gaes(train_data[2], train_data[3])

    return train_data, score


if wandb_log:
  wandb.login()
  wandb.init(project = "DLA_Lab3_DRL", config = config, monitor_gym = True, save_code = True, name = f"myPPO-{name_env}_plr{POLICY_LR}_vlr{VALUE_LR}_kldiv{TARGET_KL_DIV}_gamma{GAMMA}_decay{DECAY}")
  config = wandb.config

print("START Proximal Policy Optimization")

env = gym.make(env_name, render_mode = render_mode)
agent = PPOAgent(env.observation_space.shape[0], env.action_space.n, hidden_size).to(device)
ppo = PPOTrainer(agent, plr = POLICY_LR, vlr = VALUE_LR, target_kl_div = TARGET_KL_DIV,
                 max_policy_train_iters = MAX_POLICY_TRAIN_ITERS, value_train_iters = VALUE_TRAIN_ITERS)

solved = False
last_scores = []
partial_best_score = float('-inf')
total_best_score = float('-inf')
os.makedirs(PATH, exist_ok=True)

for episode in range(NUM_EPISODES):
  # Perform rollout
  train_data, score = rollout(agent, env, device)
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
  ppo.train_policy(obs, acts, act_log_probs, gaes)
  ppo.train_value(obs, returns)

  last_scores.append(score)
  if len(last_scores) > 100:
    last_scores.pop(0)
  avg_score = sum(last_scores) / len(last_scores)

  if avg_score >= AVG_SCORE_THRESHOLD and len(last_scores) == 100 and not solved:
    solved = True
    torch.save(agent.state_dict(), PATH + f'/solved_episode{episode}.pth')
    print(f"ENVIRONMENT SOLVED in {episode} episodes!!!")
    print(f"Checkpoint saved at episode {episode} with average score of {avg_score:.2f}")
    break

#  if score > SCORE_THRESHOLD and score > partial_best_score:
#      partial_best_score = score
#      torch.save(agent.qnetwork_local.state_dict(), PATH + f'/episode{episode}.pth')
#      if score > total_best_score:
#          total_best_score = score
#          torch.save(agent.qnetwork_local.state_dict(), PATH + f'/best_model.pth')
#      print(f"New best model saved at episode {episode}")

#  if episode % 500 == 0:
#      partial_best_score = float('-inf')
  
  print(f"Episode {episode} - Score: {score:.2f}; Last 100 runs avg score: {avg_score:.2f}")
  if wandb_log:
    wandb.log({"score": score}, step = episode)

if wandb_log:
  wandb.finish()
