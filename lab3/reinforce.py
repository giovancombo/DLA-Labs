# Raw implementation of the REINFORCE algorithm from professor Andrew David Bagdanov

# Standard imports.
import numpy as np
import torch
# Plus one non standard one -- we need this to sample from policies.
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Given an environment, observation, and policy, sample from pi(a | obs). Returns the
# selected action and the log probability of that action (needed for policy gradient).
def select_action(env, obs, policy):
    dist = Categorical(policy(obs))
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return (action.item(), log_prob.reshape(1))

# Utility to compute the discounted total reward. Torch doesn't like flipped arrays, so we need to
# .copy() the final numpy array. There's probably a better way to do this.
def compute_returns(rewards, gamma):
    return np.flip(np.cumsum([gamma**(i+1)*r for (i, r) in enumerate(rewards)][::-1]), 0).copy()

# Given an environment and a policy, run it up to the maximum number of steps.
def run_episode(env, policy, maxlen=500, device='cpu'):
    # Collect just about everything.
    observations = []
    actions = []
    log_probs = []
    rewards = []
    
    # Reset the environment and start the episode.
    (obs, info) = env.reset()
    for i in range(maxlen):
        # Get the current observation, run the policy and select an action.
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
        (action, log_prob) = select_action(env, obs, policy)
        observations.append(obs)
        actions.append(action)
        log_probs.append(log_prob)
        
        # Advance the episode by executing the selected action.
        (obs, reward, term, trunc, info) = env.step(action)
        rewards.append(reward)
        if term or trunc:
            break
    return (observations, actions, torch.cat(log_probs), rewards)
    

# A direct, inefficient, and probably buggy implementation of the REINFORCE policy gradient algorithm.
def reinforce(policy, env, env_render=None, gamma=0.999, lr=1e-2, num_episodes=10, device='cpu'):
    # The only non-vanilla part: we use Adam instead of SGD.
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    # Track episode rewards in a list.
    running_rewards = [0.0]
    
    # The main training loop.
    policy.train()
    for episode in range(num_episodes):
        # Run an episode of the environment, collect everything needed for policy update.
        (observations, actions, log_probs, rewards) = run_episode(env, policy, device=device)
        
        # Compute the discounted reward for every step of the episode. 
        returns = torch.tensor(compute_returns(rewards, gamma), dtype=torch.float32)
        
        # Keep a running average of total discounted rewards for the whole episode.
        running_rewards.append(0.005 * returns[0].item() + 0.995 * running_rewards[-1])
        
        # Standardize returns.
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        
        # Make an optimization step
        log_probs = log_probs.to(device)
        returns = returns.to(device)
        opt.zero_grad()
        loss = (-log_probs * returns).sum()
        loss.backward()
        opt.step()
        
        # Render an episode after every 50 policy updates.
        if episode % 50:
            policy.eval()
            (obs, _, _, _) = run_episode(env_render, policy, device=device)
            policy.train()
            print(f'Running reward: {running_rewards[-1]}')
    
    # Return the running rewards.
    policy.eval()
    return running_rewards
