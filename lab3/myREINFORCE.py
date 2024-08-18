# Deep Learning Applications 2023 course, held by Professor Andrew David Bagdanov - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/DLA_Labs/tree/main/lab3

# My implementation of the REINFORCE algorithm, starting from the baseline code from professor Bagdanov

import os
import torch
import torch.nn as nn
from torch.distributions import Categorical
import wandb
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Given an environment, observation, and policy, sample from pi(a | obs). Returns the selected action.
def select_action(state, policy):
    action_probs = policy(torch.tensor(state, dtype = torch.float32, device = device))
    dist = Categorical(probs = action_probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob.reshape(1)

# Computes the discounted total rewards for each episode
def compute_rewards(rewards, gamma):
    discountedRewards = []
    for t in range(len(rewards)):
        G = 0.0
        for k, r in enumerate(rewards[t:]):
            G += (gamma ** k) * r
        discountedRewards.append(G)
    return discountedRewards

# Computes the REINFORCE loss function
def optimize_policy(log_probs, discountedRewards): 
    log_probs = torch.cat(log_probs).to(device)
    discountedRewards = torch.tensor(discountedRewards, dtype = torch.float32, device = device)
    discountedRewards = ((discountedRewards - discountedRewards.mean()) / (discountedRewards.std() + 1e-6))
    policy_loss = (-log_probs * discountedRewards).sum()
    
    return policy_loss

# The actual REINFORCE algorithm
def combo_reinforce(policy, env, name_env, gamma = 0.99, lr = 1e-3, episodes = 10, avg_thresh = 0, scorethresh = 0, device = 'cpu', wandb_log = False):
    optimizer = torch.optim.Adam(policy.parameters(), lr = lr)

    PATH = f"checkpoints/{name_env}/myREINFORCE_lr{lr}_gamma{gamma}" + str(time.time())[-6:]
    os.makedirs(PATH, exist_ok=True) 

    partial_best_score = float('-inf')
    total_best_score = float('-inf')
    solved = False
    running_rewards = [0.0]
    last_scores = []

    if wandb_log:
        wandb.watch(policy, log = "all", log_freq = 1)

    policy.train()
    for episode in range(episodes):
        (state, _) = env.reset()
        terminated, truncated = False, False
        states, actions, log_probs, rewards = [], [], [], []
        score = 0
     
        while True:
            #env.render()
            action, log_prob = select_action(state, policy)
            states.append(state)
            log_probs.append(log_prob)
            state, reward, terminated, truncated, _ = env.step(action)
            actions.append(torch.tensor(action, dtype = torch.long, device = device))
            rewards.append(reward)
            score += reward
            if terminated or truncated:
                states.append(state)
                break

        discountedRewards = compute_rewards(rewards, gamma)
        policy_loss = optimize_policy(log_probs, discountedRewards)

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        running_rewards.append(0.005 * discountedRewards[0] + 0.995 * running_rewards[-1])

        last_scores.append(score)
        if len(last_scores) > 100:
            last_scores.pop(0)
        avg_score = sum(last_scores) / len(last_scores)

        if avg_score >= avg_thresh and len(last_scores) == 100 and not solved:
            solved = True
            torch.save(policy.state_dict(), PATH + f'/solved_episode{episode}.pth')
            print("ENVIRONMENT SOLVED!!!")
            print(f"Checkpoint saved at episode {episode} with average score of {avg_score:.2f}")

        if avg_score >= avg_thresh and score > scorethresh and score > partial_best_score:
                partial_best_score = score
                torch.save(policy.state_dict(), PATH + f'/episode{episode}.pth')
                if score > total_best_score:
                    total_best_score = score
                    torch.save(policy.state_dict(), PATH + f'/best_model.pth')
                print(f"New best model saved at episode {episode}")
        
        if episode % 50 == 0:
            solved = False

        if episode % 500 == 0:
            partial_best_score = float('-inf')

        if wandb_log:
            wandb.log({"score": score,
                       "policy_loss": policy_loss,
                       "running_reward": running_rewards[-1]}, step = episode)
        print(f'Episode {episode+1}; {len(rewards)}\tScore: {score:.2f}; Running reward: {running_rewards[-1]:.2f}; Avg score: {avg_score:.2f}')
    
    torch.save(policy.state_dict(), PATH + f'/episode{episode}.pth')

    if wandb_log:
        wandb.unwatch(policy)

    return running_rewards


# My implementation of the REINFORCE algorithm with adaptive baseline
class Baseline(nn.Module):
    def __init__(self, state_dim):
        super(Baseline, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)

def compute_advantages(rewards, values, gamma):
    advantages = []
    returns = []
    R = 0
    for r, v in zip(reversed(rewards), reversed(values)):
        R = r + gamma * R
        returns.insert(0, R)
        advantage = R - v
        advantages.insert(0, advantage)
    return advantages, returns

def combo_reinforce_baseline(policy, baseline, env, name_env, gamma=0.99, lr_policy=1e-3, lr_baseline=1e-4, episodes=10, avg_thresh=0, scorethresh=0, device='cpu', wandb_log=False):
    baseline = baseline.to(device)
    optimizer_policy = torch.optim.Adam(policy.parameters(), lr=lr_policy)
    optimizer_baseline = torch.optim.Adam(baseline.parameters(), lr=lr_baseline)

    PATH = f"checkpoints/{name_env}/myREINFORCEbaseline_lr{lr_policy}_gamma{gamma}" + str(time.time())[-6:]
    os.makedirs(PATH, exist_ok=True) 

    partial_best_score = float('-inf')
    total_best_score = float('-inf')
    solved = False
    running_rewards = [0.0]
    last_scores = []

    if wandb_log:
        wandb.watch((policy, baseline), log="all", log_freq=1)

    policy.train()
    baseline.train()
    for episode in range(episodes):
        state, _ = env.reset()
        log_probs, rewards, values, states = [], [], [], []
        score = 0
     
        while True:
            states.append(state)  # Salva lo stato corrente
            action, log_prob = select_action(state, policy)
            value = baseline(torch.tensor(state, dtype=torch.float32, device=device))
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value.item())
            
            score += reward
            state = next_state
            
            if terminated or truncated:
                break

        advantages, returns = compute_advantages(rewards, values, gamma)
        
        # Converti log_probs e advantages in tensori
        log_probs = torch.cat(log_probs).to(device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        
        # Assicurati che log_probs e advantages abbiano la stessa lunghezza
        min_length = min(log_probs.shape[0], advantages.shape[0])
        log_probs = log_probs[:min_length]
        advantages = advantages[:min_length]

        # Calcola la policy loss
        policy_loss = -(log_probs * advantages).mean()
        
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()
        
        # Aggiornamento della baseline
        states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
        
        predicted_values = baseline(states_tensor).squeeze()
        baseline_loss = nn.MSELoss()(predicted_values, returns_tensor)
        
        optimizer_baseline.zero_grad()
        baseline_loss.backward()
        optimizer_baseline.step()

        last_scores.append(score)
        if len(last_scores) > 100:
            last_scores.pop(0)
        avg_score = sum(last_scores) / len(last_scores)

        if avg_score >= avg_thresh and len(last_scores) == 100 and not solved:
            solved = True
            torch.save(policy.state_dict(), PATH + f'/solved_episode{episode}.pth')
            print("ENVIRONMENT SOLVED!!!")
            print(f"Checkpoint saved at episode {episode} with average score of {avg_score:.2f}")

        if avg_score >= avg_thresh and score > scorethresh and score > partial_best_score:
                partial_best_score = score
                torch.save(policy.state_dict(), PATH + f'/episode{episode}.pth')
                if score > total_best_score:
                    total_best_score = score
                    torch.save({
                        'policy': policy.state_dict(),
                        'baseline': baseline.state_dict()
                    }, PATH + f'/best_model.pth')
                print(f"New best model saved at episode {episode}")
        
        if episode % 50 == 0:
            solved = False

        if episode % 500 == 0:
            partial_best_score = float('-inf')

        if wandb_log:
            wandb.log({
                "score": score,
                "policy_loss": policy_loss.item(),
                "baseline_loss": baseline_loss.item(),
                "running_reward": running_rewards[-1]
            }, step=episode)
        
        print(f'Episode {episode+1}; {len(rewards)}\tScore: {score:.2f}; Running reward: {running_rewards[-1]:.2f}; Avg score: {avg_score:.2f}')
    
    torch.save({
        'policy': policy.state_dict(),
        'baseline': baseline.state_dict()
    }, PATH + f'/episode{episode}.pth')

    if wandb_log:
        wandb.unwatch((policy, baseline))

    return running_rewards