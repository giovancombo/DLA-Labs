# Deep Learning Applications: Laboratory #3 - DRL

In this Laboratory I will explore the realm of Deep Reinforcement Learning and its applications to navigation problems and simulations. Many of the key theoretical concepts about this Lab were taken from [this excellent blog post by Andrej Karpathy](http://karpathy.github.io/2016/05/31/rl/).

For this Lab, I will create a new *conda* environment named **DRL**, which will allow me to work with the numerous libraries specifically dedicated to Deep Reinforcement Learning tasks.

## Exercise 1: Testing the Environment

Part of this experience will involve "hacking" an implementation of a navigation environment, provided by Francesco Fantechi from Ingegneria Informatica. The setup is relatively straightforward:

+ The environment is a simple 2D space containing a limited number of *obstacles* (black squares) and a single *goal* (green circle). The agent (blue dot) must learn to navigate to the goal while avoiding obstacles.
+ The agent *observes* the environment through 16 uniformly cast rays, which return the distance to the first obstacle encountered, as well as the distance and direction to the goal.
+ The agent can perform three possible actions: `ROTATE LEFT`, `ROTATE RIGHT`, or `MOVE FORWARD`.

For each step of an episode, the agent receives a reward based on the following criteria:
+ -100 if it hits an obstacle (ending the episode).
+ -100 if it reaches one hundred steps without finding the goal (time's up, ending the episode).
+ +100 if it reaches the goal (ending the episode).
+ A small *positive* reward if the distance to the goal is *reduced*.
+ A small *negative* reward if the distance to the goal is *increased*.

The environment implementation and setup is available in the `gym_navigation` folder.

The default **Deep Q-Learning** implementation and training code for solving this exercise are available in the `main_basedqn.py` script. My approach consists of two main steps:
- Training Phase: I train the agent for 2000 episodes and save the trained model.
- Testing Phase: I run `main_basedqn.py` again, this time with `TRAIN = False`, and load the last saved checkpoint to evaluate the agent's performance.

<p float="center" align="center">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab3/images/gymnav.png" width="20%">
</p>
<p align="center"><i><b>Figure 1</b> | Interface of the <b>gymnav</b> environment</i></p><br>

---

## Exercise 2: Stabilizing Q-Learning

The *Deep Q-Network* **baseDQN** provided in the `main_basedqn.py` script required some hyperparameter tuning to stabilize the training process and achieve performance capable of solving the *gymnav* environment.  In my optimization efforts, I focused on evaluating the effects of three key parameters:
- `EPS_DECAY`: The decay rate of the exploration probability.
- `LEARNING_RATE`: The step size at each iteration while moving toward a minimum of the loss function.
- `GAMMA`: The discount factor for future rewards.

<p float="center" align="center">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab3/images/gymnav_basedqn_eps.png" width="60%">
</p>
<p align="center"><i><b>Figure 2</b> | Comparison between runs on the gymnav environment using <b>baseDQN</b> with different <b>EPS_DECAY</b></i></p>

It appears that the smaller the `EPS_DECAY` value, the better the performance. However, I found it's crucial to be cautious when lowering this hyperparameter too much, because excessively small values can lead to performances that overly favor *exploitation* over *exploration*, potentially causing the agent to become stuck in a particular behavior too soon in the training.

<br><p float="center" align="center">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab3/images/gymnav_basedqn_lr.png" width="60%">
</p>
<p align="center"><i><b>Figure 3</b> | Comparison between runs on the gymnav environment using <b>baseDQN</b> with different <b>LEARNING_RATE</b></i></p>

We observe optimal performance for `LEARNING_RATE` in the range of 0.01 to 0.001, where the model effectively balances learning speed and stability. Learning Rates below 0.0001 lead to training failure, as the weight updates become too small for effective learning.

<br><p float="center" align="center">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab3/images/gymnav_basedqn_gamma.png" width="60%">
</p>
<p align="center"><i><b>Figure 4</b> | Comparison between runs on the gymnav environment using <b>baseDQN</b> with different <b>GAMMA</b></i></p>

There is consistent performance across a wide range of `GAMMA` values, from 0.9 to 0.9999. This stability suggests that both short-term and long-term strategies are equally effective in achieving the navigation goal.

<br><p float="center" align="center">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab3/images/gymnav_basedqn.png" width="60%">
</p>
<p align="center"><i><b>Figure 5</b> | Best runs on the gymnav environment using <b>baseDQN</b></i></p>

Setting `render = 'human'` enables a qualitative evaluation of the agent's navigation improvement. Over multiple runs, I observed the following patterns:
- During the initial phase of training (the first 300-400 episodes), the agent often wanders randomly in the environment.
- Usually, the agent begins to reach the goal more frequently, achieving a positive running average score.
- Despite there are clear improvements, the agent's performance remains inconsistent: it can successfully reach the goal for several consecutive episodes, and then begin to fluctuate or move in the opposite direction of the goal in subsequent episodes.
- Often, the agent collides with walls or obstacles without attempting to change direction, even immediately after spawning. In some instances, the agent navigates to the goal but stops just short of reaching it, then changes direction.

---

## Exercise 3: Going Deeper

### Exercise 3.1: Solving the environment with REINFORCE

**REINFORCE** is one of the pioneering Policy Gradient algorithms in DRL. Given its simplicity, I decided to utilize not only professor Bagdanov's implementation, available in the `baseREINFORCE.py` script, but also to develop my own version, inspired by a tutorial, which I've implemented in the `myREINFORCE.py` script. To facilitate experimentation, I created a `main_reinforce.py` script that allows for launching runs using either implementation.

<p float="center" align="center">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab3/images/gymnav_reinforce.png" width="60%">
</p>
<p align="center"><i><b>Figure 6</b> | Best runs on the gymnav environment using <b>baseREINFORCE</b></i></p>

After conducting multiple runs using REINFORCE in the Navigation environment, I observed an unexpected behavior pattern in the agents. After some first episodes in which agents struggled to reach the goal, but their movements across the map maintained at least a logical pattern, agents adopted a peculiar strategy involving circular movements. This behavior allowed them to balance their total episode reward without incurring significant losses: by moving in circles, agents alternated between slightly negative rewards when moving away from the goal, and slightly positive rewards when moving towards the goal. This kind of strategy is sub-optimal for the agent, but it represents a form of *mode collapse*, where the agent becomes trapped in a local minimum of the reward function. This behavior prevents the agent from learning the correct policy for consistently succeeding in each episode.

Another notable behavior is the strong dependence of episode outcomes on the initial policy initialization. Cases where the agent radically changes its direction to actively go towards the goal are rare. To address this, I experimented with several modifications on the environment reward system, as a hint in the exercise suggested that the *gymnav* environment suffered from a design flaw, which was causing issues when using REINFORCE:
- In the `gym_navigation/envs/navigation_track.py` script, modifying `FORWARD_REWARD` from 2 to -0.1 (from slightly positive to slightly negative).
- In the `gym_navigation/envs/navigation_goal.py` script, modifying `BACKWARD_REWARD` from -1 to -2.5.
- In the `gym_navigation/enums/action.py` script, removing the *linear shift* from the rotation actions, as every slight rotation would result in a very little positive reward.

However, removing the *linear shift* from the rotation actions resulted in an agent that moved through the environment in a jerky manner, without any improvement in performance. This modification, while altering the agent's movement pattern, failed to address the underlying issues or enhance the navigation capabilities.

Modifying the reward system for getting closer to or further from the goal only slightly improved performance. The agent remained inconsistent, often alternating between successfully reaching the goal and demonstrating inability to maintain a clear policy.

The hyperparameter tuning for REINFORCE focused on `LEARNING_RATE` and `GAMMA`. I could observed that optimal values of `GAMMA` range between 0.8 and 0.99 (*Figure 6*), however an important trend emerged with lower gamma values: the agent achieved positive scores but rarely reached the goal. Instead, it developed a strategy of rotating in circles, allowing the episode to time out. This behavior maximized short-term rewards without achieving the actual objective.

<p float="center" align="center">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab3/images/gymnav_confronto.png" width="60%">
</p>
<p align="center"><i><b>Figure 7</b> | Comparison between runs on the gymnav environment using <b>baseREINFORCE</b> and <b>baseDQN</b></i></p>

*Figure 7* demonstrates how Deep Q-Learning achieves better performances than REINFORCE in the *gymnav* task. This superiority can be attributed to DQL's ability to learn from past experiences, leading to faster and more stable learning. The *epsilon-greedy* strategy and directly learning an *action-value function* are features that provide a more long-term stable learning compared to REINFORCE's policy gradient approach.

---
### Exercise 3.2: Solving another environment

After working with the custom *gymnav* environment, it is now time to explore some of the environments available in the [Gymnasium](https://gymnasium.farama.org/) framework, which offers a consistent interface to a broad range of Reinforcement Learning environments, making it an ideal platform for comparative studies.

In this section, I will perform a comparative analysis of the **REINFORCE** and **Deep Q-Learning** algorithms' performances in solving two of the most popular OpenAI Gymnasium environments:
+ [Lunar Lander-v2](https://gymnasium.farama.org/environments/box2d/lunar_lander/): the environment is considered *solved* when the average score of the latest 100 episodes reaches **200**.
+ [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/): the environment is considered *solved* when the average score of the latest 100 episodes reaches **190**.

<p float="center" align="center">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab3/images/lunarlander_interface.png" width="25%">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab3/images/cartpole_interface.png" width="25%">
</p>
<p align="center"><i><b>Figure 8</b> | <b>LunarLander-v2</b> (left) and <b>CartPole-v1</b> (right) interfaces</i></p>

Since the **baseDQN** architecture provided in the `main_basedqn.py` script was specifically tailored for the *gymnav* environment, I recognized that I needed a more flexible solution. So, I decided to develop my own version of the DQL algorithm (inspired by a tutorial), **myDQN**, which is designed to be more adaptable to various *Gymnasium* environments. It is implemented in the `main_mydqn.py` script.

#### LunarLander-v2

<p float="center" align="center">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab3/images/lunarlander_reinforce.png" width="49%">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab3/images/lunarlander_dqn.png" width="49%">
</p>
<p align="center"><i><b>Figure 9</b> | Runs on the LunarLander-v2 environment using <b>REINFORCE</b> (left) and <b>myDQN</b> (right) algorithms</i></p>

#### CartPole-v1

<p float="center" align="center">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab3/images/cartpole_reinforce.png" width="49%">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab3/images/cartpole_dqn.png" width="49%">
</p>
<p align="center"><i><b>Figure 10</b> | Runs on the CartPole-v1 environment using <b>REINFORCE</b> (left) and <b>myDQN</b> (right) algorithms</i></p>

Surprisingly, here **myREINFORCE** algorithm solved the CartPole-v1 environment faster than **myDQN**.

---
### Exercise 3.3: Advanced techniques 

While REINFORCE and Deep Q-Learning are foundational approaches in Deep Reinforcement Learning (DRL), they are no longer considered state-of-the-art. Currently, one of the most powerful methods for solving DRL environments is [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347).

Intrigued by its potential, I decided to implement PPO to enhance our comparative analysis. My implementation, which I've named **myPPO**, takes inspiration from [this off-the-shelf implementation of PPO](https://github.com/bentrevett/pytorch-rl/blob/master/5a%20-%20Proximal%20Policy%20Optimization%20(PPO)%20%5BLunarLander%5D.ipynb). The goal is to solve the *LunarLander-v2* environment and compare its performance with the previously implemented Deep Q-Learning and REINFORCE algorithms. **myPPO** implementation can be found in the `main_myppo1.py` script.

*Proximal Policy Optimization* employs an Actor-Critic approach. This methodology uses two distinct models, one called *Actor* and the other called *Critic*.
+ The *Actor* model earns the optimal action to take in a given observed state of the environment. In the *LunarLander-v2* case, it takes in input a list of eight values that represent the current state of the rocket (position, velocity, orientation), and outputs the specific action indicating which engine to fire.
+ The *Critic* model evaluates the value of being in a particular state.

This dual-model structure allows PPO to simultaneously learn both the policy (via the *Actor*) and the value function (via the *Critic*), leading to more stable and efficient learning.

#### LunarLander-v2

<p float="center" align="center">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab3/images/lunarlander_confronto.png" width="49%">
</p>
<p align="center"><i><b>Figure 11</b> | Comparison between runs on the LunarLander-v2 environment using <b>REINFORCE</b>, <b>myDQN</b> and <b>myPPO</b> algorithms</i></p>

#### CartPole-v1

<p float="center" align="center">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab3/images/cartpole_confronto.png" width="49%">
</p>
<p align="center"><i><b>Figure 12</b> | Comparison between runs on the CartPole-v1 environment using <b>REINFORCE</b>, <b>myDQN</b> and <b>myPPO</b> algorithms</i></p>
