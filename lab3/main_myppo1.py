# Deep Learning Applications 2023 course, held by Professor Andrew David Bagdanov - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/DLA_Labs/tree/main/lab3

# An Off-the-Shelf implementation of the Proximal Policy Optimization algorithm

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import gymnasium as gym
import wandb

env_name = 'LunarLander-v2'     # 'gym_navigation:NavigationGoal-v0' or 'LunarLander-v2' or 'CartPole-v1' ...
wandb_log = True
render_mode = None

SCORE_THRESHOLD = 250
AVG_SCORE_THRESHOLD = 200

NUM_EPISODES = 2000
HIDDEN_SIZE = 64
LEARNING_RATE = 5e-4
GAMMA = 0.99
N_TRIALS = 25
PPO_STEPS = 5
PPO_CLIP = 0.2

name_env = 'gymnav' if env_name == 'gym_navigation:NavigationGoal-v0' else env_name
config = {
    "env_name": env_name,
    "episodes": NUM_EPISODES,
    "policy_lr": LEARNING_RATE,
    "value_lr": LEARNING_RATE,
    "gamma": GAMMA,
}

PATH = f"checkpoints/{name_env}/myPPO_pvlr{LEARNING_RATE}_gamma{GAMMA}" + str(time.time())[-4:]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        x = self.net(x)
        return x
    
class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        
        self.actor = actor
        self.critic = critic
        
    def forward(self, state):
        
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        
        return action_pred, value_pred


def train(env, policy, optimizer, discount_factor, ppo_steps, ppo_clip):
    policy.train()
    state, _ = env.reset()
    states, actions, log_prob_actions, values, rewards = [], [], [], [], []
    done = False
    score = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        #append state here, not after we get the next state from env.step()
        states.append(state)
        action_pred, value_pred = policy(state)
        action_prob = F.softmax(action_pred, dim = -1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        actions.append(action)
        log_prob_actions.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)

        score += reward

        if done:
            break
    
    states = torch.cat(states)
    actions = torch.cat(actions)    
    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)
    
    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(returns, values)
    policy_loss, value_loss = update_policy(policy, states, actions, log_prob_actions, advantages, returns, optimizer, ppo_steps, ppo_clip)

    return policy_loss, value_loss, score

def calculate_returns(rewards, discount_factor, normalize = True):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
    returns = torch.tensor(returns)
    if normalize:
        returns = (returns - returns.mean()) / returns.std()
    
    return returns

def calculate_advantages(returns, values, normalize = True):
    advantages = returns - values
    if normalize:
        advantages = (advantages - advantages.mean()) / advantages.std()
        
    return advantages

def update_policy(policy, states, actions, log_prob_actions, advantages, returns, optimizer, ppo_steps, ppo_clip):
    total_policy_loss = 0 
    total_value_loss = 0
    states = states.detach()
    actions = actions.detach()
    log_prob_actions = log_prob_actions.detach()
    advantages = advantages.detach()
    returns = returns.detach()
    for _ in range(ppo_steps):   
        #get new log prob of actions for all input states
        action_pred, value_pred = policy(states)
        value_pred = value_pred.squeeze(-1)
        action_prob = F.softmax(action_pred, dim = -1)
        dist = distributions.Categorical(action_prob)
        #new log prob using old actions
        new_log_prob_actions = dist.log_prob(actions)
        policy_ratio = (new_log_prob_actions - log_prob_actions).exp()
                
        policy_loss_1 = policy_ratio * advantages
        policy_loss_2 = torch.clamp(policy_ratio, min = 1.0 - ppo_clip, max = 1.0 + ppo_clip) * advantages
        policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()
        value_loss = F.smooth_l1_loss(returns, value_pred).mean()
    
        optimizer.zero_grad()
        policy_loss.backward()
        value_loss.backward()
        optimizer.step()
    
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
    
    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps

def evaluate(env, policy):
    policy.eval()
    state, _ = env.reset()
    score = 0
    done = False
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_pred, _ = policy(state)
            action_prob = F.softmax(action_pred, dim = -1)    
        action = torch.argmax(action_prob, dim = -1)
        state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        score += reward
        
        if done:
            break

    return score



if wandb_log:
  wandb.login()
  wandb.init(project = "DLA_Lab3_DRL", config = config, monitor_gym = True, save_code = True, name = f"myPPO-{name_env}_pvlr{LEARNING_RATE}_gamma{GAMMA}")
  config = wandb.config

print("START Proximal Policy Optimization")

env = gym.make(env_name, render_mode = render_mode)
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

actor = MLP(observation_space, HIDDEN_SIZE, action_space)
critic = MLP(observation_space, HIDDEN_SIZE, 1)
policy = ActorCritic(actor, critic)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)
policy.apply(init_weights)

optimizer = optim.Adam(policy.parameters(), lr = LEARNING_RATE)

solved = False
last_scores = []
partial_best_score = float('-inf')
total_best_score = float('-inf')
os.makedirs(PATH, exist_ok=True)

for episode in range(NUM_EPISODES):
    policy_loss, value_loss, score = train(env, policy, optimizer, GAMMA, PPO_STEPS, PPO_CLIP)
    #test_reward = evaluate(test_env, policy)
    
    last_scores.append(score)
    if len(last_scores) > 100:
        last_scores.pop(0)
    avg_score = sum(last_scores) / len(last_scores)

    if avg_score >= AVG_SCORE_THRESHOLD and len(last_scores) == 100 and not solved:
        solved = True
        torch.save(policy.state_dict(), PATH + f'/solved_episode{episode}.pth')
        print(f"ENVIRONMENT SOLVED in {episode} episodes!!!")
        print(f"Checkpoint saved at episode {episode} with average score of {avg_score:.2f}")
        break
    
    print(f"Episode {episode} - Score: {score:.2f}; Last 100 runs avg score: {avg_score:.2f}")
    if wandb_log:
        wandb.log({"score": score}, step = episode)

if wandb_log:
    wandb.finish()
