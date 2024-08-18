# Deep Learning Applications 2023 course, held by Professor Andrew David Bagdanov - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/DLA_Labs/tree/main/lab3

# NOTE: My contributions here were cleansing code, adding checkpoint saving and Weights & Biases compatibility for performance tracking.

import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gym_navigation.memory.replay_memory import ReplayMemory, Transition
import numpy as np
import math
import random
import wandb
import time
torch.set_default_dtype(torch.float32)


env_name = 'LunarLander-v2'     # 'gym_navigation:NavigationGoal-v0' or 'LunarLander-v2' or 'CartPole-v1' ...
wandb_log = True

TRAIN = True
checkpoint_loaded = None

SCORE_THRESHOLD = 250           # CartPole: 500; LunarLander: 250; gymnav: 150
AVG_SCORE_THRESHOLD = 150       # CartPole: 350; LunarLander: 200; gymnav: 100

NUM_EPISODES = 3000
START_EPS = 1.0                 # the starting value of epsilon
EPS_DECAY = 0.995               # controls the rate of exponential decay of epsilon, higher means a slower decay
MIN_EPS = 0.1
GAMMA = 0.99                    # Discount Factor
BATCH_SIZE = 128                 # number of transitions randomly sampled from the replay buffer
LEARNING_RATE = 1e-4
TEST_EPISODES = 500
REPLAY_BUFFER_CAPACITY = 1000000
UPDATE_TARGET = 20
steps_done = 0
hidden_sizes = [256, 256]

name_env = 'gymnav' if env_name == 'gym_navigation:NavigationGoal-v0' else env_name
config = {
    "env_name": env_name,
    "episodes": NUM_EPISODES,
    "eps_decay": EPS_DECAY,
    "gamma": GAMMA,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE
}

PATH = f'./checkpoints/{name_env}/baseDQN_eps{EPS_DECAY}_lr{LEARNING_RATE}_gamma{GAMMA}' + str(time.time())[-6:]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQLN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQLN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_sizes[0]).to(torch.float32)
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1]).to(torch.float32)
        self.layer3 = nn.Linear(hidden_sizes[1], n_actions).to(torch.float32)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

MAX_DISTANCE = 15.0
MIN_DISTANCE = 0.2

def normalize(state: np.ndarray) -> np.ndarray:
    normalized_state = np.zeros(shape=state.shape, dtype=np.float32)
    for i in range(len(state)):
        if i < 17:
            normalized_state[i] = (state[i] - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
        else:
            normalized_state[i] = state[i] / math.pi
    return normalized_state

# Training loop
if TRAIN:
    env = gym.make(env_name, render_mode = None)
    env.action_space.seed(1492)

    state_observation, info = env.reset(seed=1492)
    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    n_observations = len(state_observation)
    # Instantiate policy network.
    q_function = DQLN(n_observations, n_actions).to(device)

    # Instantiate target network.
    target_q_function = DQLN(n_observations, n_actions).to(device)
    target_q_function.load_state_dict(q_function.state_dict())

    # Use Adam to optimize.
    optimizer = optim.Adam(q_function.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    # Using a larger replay buffer.
    replay_buffer = ReplayMemory(REPLAY_BUFFER_CAPACITY)
    EPS = START_EPS

    # Epsilon-greedy action sampling.
    def select_action_epsilon(state):
        sample = random.random()
        if sample > EPS:
            with torch.no_grad():
                # return index of action with the best Q value in the current state
                return q_function(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

    # Do one batch of gradient descent.
    def optimize_model():
        # Make sure we have enough samples in replay buffer.
        if len(replay_buffer) < BATCH_SIZE:
            return

        # Sample uniformly from replay buffer.
        transitions = replay_buffer.sample(BATCH_SIZE)
        # This converts batch-arrays of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Concatenate into tensors for batch update.
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=device)

        # policy_net computes Q(state, action taken)
        state_action_values = q_function(state_batch).gather(1, action_batch)

        # Compute the expected Q values with BELLMAN OPTIMALITY Q VALUE EQUATION:
        # Q(state,action) = reward(state,action) + GAMMA * max(Q(next_state, actions), action)
        expected_state_action_values = reward_batch + GAMMA * (1 - done_batch) * target_q_function(next_state_batch).max(1)[0]

        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

        return loss

    def run_validation(env, policy_net, num=10):
        running_rewards = [0.0] * num
        for i in range(num):
            state_observation, _ = env.reset()
            while True:
                #state_observation = normalize(state_observation)
                state_observation = torch.tensor(state_observation, dtype=torch.float32, device=device).unsqueeze(0)
                action = policy_net(state_observation).max(1)[1].view(1, 1)
                state_observation, reward, terminated, truncated, _ = env.step(action.item())
                running_rewards[i] += reward

                if terminated or truncated:
                    break
        return running_rewards

    if torch.cuda.is_available():
        num_episodes = NUM_EPISODES
    else:
        num_episodes = NUM_EPISODES

    if wandb_log:
        wandb.login()
        wandb.init(project = "DLA_Lab3_DRL", monitor_gym = True, config = config, save_code = True, name = f"baseDQN-{name_env}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_eps{EPS_DECAY}_gamma{GAMMA}")
        config = wandb.config
        wandb.watch(q_function, log = "all", log_freq = 1)

    print("START Deep Q-Learning Navigation Goal")
    
    running_reward = 0
    last_scores = []
    solved = False
    partial_best_score = float('-inf')
    total_best_score = float('-inf')
    os.makedirs(PATH, exist_ok=True)

    # Sample experience, save in Replay Buffer.
    for i_episode in range(0, num_episodes, 1):
        state_observation, info = env.reset()
        #state_observation = normalize(state_observation)
        state_observation = torch.tensor(state_observation, dtype=torch.float32, device=device).unsqueeze(0)
        steps = 0
        score = 0
        # Run one episode.
        while True:
            action = select_action_epsilon(state_observation)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            score += reward
            #observation = normalize(observation)  # Normalize in [0,1]
            reward = torch.tensor([reward], device=device, dtype=torch.float32)
            done = terminated or truncated
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            # Store the transition in memory
            replay_buffer.push(state_observation, action, next_state, reward, done)
            # Move to the next state
            state_observation = next_state
            steps += 1

            if done:
                break

        # Perform one step of the optimization (on the policy network)
        loss = optimize_model()
        # Epsilon decay
        EPS = max(MIN_EPS, EPS * EPS_DECAY)
        running_reward = (1 - 0.005) * running_reward + 0.005 * score

        # Update target network.
        if not i_episode % UPDATE_TARGET:
            policy_net_state_dict = q_function.state_dict()
            target_q_function.load_state_dict(policy_net_state_dict)

        # Every 50 episodes, validate.
        if i_episode % 50 == 0:
            validation_rewards = run_validation(env, q_function)
            avg_validation_reward = sum(validation_rewards) / len(validation_rewards)
            if wandb_log:
                wandb.log({"avg_validation_reward": avg_validation_reward}, step=i_episode)

        last_scores.append(score)
        if len(last_scores) > 100:
            last_scores.pop(0)
        avg_score = sum(last_scores) / len(last_scores)

        if avg_score >= AVG_SCORE_THRESHOLD and len(last_scores) == 100 and not solved:
            solved = True
            torch.save(q_function.state_dict(), PATH + f'/solved_episode{i_episode}.pth')
            print("ENVIRONMENT SOLVED!!!")
            print(f"Checkpoint saved at episode {i_episode} with average score of {avg_score:.2f}")

        if avg_score >= AVG_SCORE_THRESHOLD and score > SCORE_THRESHOLD and score > partial_best_score:
                partial_best_score = score
                torch.save(q_function.state_dict(), PATH + f'/episode{i_episode}.pth')
                if score > total_best_score:
                    total_best_score = score
                    torch.save(q_function.state_dict(), PATH + f'/best_score_model.pth')
                print(f"New best model saved at episode {i_episode}")
        
        if i_episode % 100 == 0:
            solved = False

        if i_episode % 500 == 0:
            partial_best_score = float('-inf')

        print(f"Episode: {i_episode}; Score: {score:.2f}; Running Reward: {running_reward:.2f}; Avg score: {avg_score:.2f}")

        if wandb_log:
            wandb.log({"score": score,
                       "policy_loss": loss,
                       "running_reward": running_reward}, step = i_episode)
            
    torch.save(q_function.state_dict(), PATH + f'/episode{i_episode}.pth')

    if wandb_log:
        wandb.finish()

    env.close()
    print('COMPLETE')

else:
    # For accuracy check
    # env = gym.make('gym_navigation:NavigationGoal-v0', render_mode=None, track_id=1)
    # For visible check
    env = gym.make(env_name, render_mode='human')

    env.action_space.seed(42)
    state_observation, info = env.reset(seed=42)

    n_actions = env.action_space.n
    n_observations = len(state_observation)

    q_function = DQLN(n_observations, n_actions).to(device)
    q_function.load_state_dict(torch.load(PATH + f'/{checkpoint_loaded}'))
    not_terminated = 0
    success = 0
    for _ in range(TEST_EPISODES):
        steps = 0
        while True:
            #state_observation = normalize(state_observation)
            state_observation = torch.tensor(state_observation, dtype=torch.float32, device=device).unsqueeze(0)
            action = q_function(state_observation).max(1)[1].view(1, 1)
            state_observation, reward, terminated, truncated, info = env.step(action.item())
            steps += 1
            if steps >= 200:
                not_terminated += 1
                truncated = True

            if terminated or truncated:
                if not truncated and reward == 500:
                    success += 1
                state_observation, info = env.reset()
                break

    env.close()
    print("Executed " + str(TEST_EPISODES) + " episodes:\n" + str(success) + " successes\n" + str(
        not_terminated) + " episodes not terminated\n" + str(
        TEST_EPISODES - (success + not_terminated)) + " failures\n")
