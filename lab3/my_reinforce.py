# Deep Learning Applications 2023 course, held by Professor Andrew David Bagdanov - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/DLA_Labs/tree/main/lab3

# My implementation of the REINFORCE algorithm, starting from the baseline code from professor Bagdanov

import torch
from torch.distributions import Categorical
import wandb

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
def combo_reinforce(env, policy, lr = 1e-2, gamma = 0.999, episodes = 10, device = 'cpu', wandb_log = False):
    optimizer = torch.optim.Adam(policy.parameters(), lr = lr)
    running_rewards = [0.0]

    if wandb_log:
        wandb.watch(policy, log = "all", log_freq = 1)

    policy.train()
    for episode in range(episodes):
        (state, _) = env.reset()
        terminated, truncated = False, False
        states, actions, log_probs, rewards = [], [], [], []
        score = 0
     
        while True:
            env.render()
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
        running_rewards.append(0.010 * discountedRewards[0] + 0.990 * running_rewards[-1])

        if wandb_log:
            wandb.log({"score": score,
                       "policy_loss": policy_loss,
                       "running_reward": running_rewards[-1]}, step = episode)
        print(f'Episode {episode+1}, {len(rewards)}\tScore: {score:.2f}; Running reward: {running_rewards[-1]:.2f}')
    
    if wandb_log:
        wandb.unwatch(policy)

    return running_rewards


# The actual REINFORCE algorithm with baseline
def combo_reinforce_with_baseline(env, policy, lr = 1e-2, gamma = 0.999, episodes = 10, device = 'cpu', wandb_log = False):
    optimizer = torch.optim.Adam(policy.parameters(), lr = lr)
    running_rewards = [0.0]

    if wandb_log:
        wandb.watch(policy, log = "all", log_freq = 1)

    policy.train()
    for episode in range(episodes):
        (state, _) = env.reset()
        terminated, truncated = False, False
        states, actions, log_probs, rewards = [], [], [], []
        score = 0
     
        while True:
            env.render()
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

        if wandb_log:
            wandb.log({"score": score,
                       "policy_loss": policy_loss,
                       "running_reward": running_rewards[-1]}, step = episode)
        print(f'Episode {episode+1}, {len(rewards)}\tScore: {score:.2f}; Running reward: {running_rewards[-1]:.2f}')
    
    if wandb_log:
        wandb.unwatch(policy)

    return running_rewards
