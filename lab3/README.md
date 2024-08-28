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
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab3/images/gymnav.png" width="25%">
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

<p float="center" align="center">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab3/images/gymnav_confronto.png" width="60%">
</p>
<p align="center"><i><b>Figure 7</b> | Comparison between runs on the gymnav environment using <b>baseREINFORCE</b> and <b>baseDQN</b></i></p>

*Figure 7* demonstrates how Deep Q-Learning achieves better performances than REINFORCE.

After conducting multiple runs using REINFORCE in the Navigation environment, I observed an unexpected behavior pattern in the agents. After some first episodes in which agents struggled to reach the goal, but their movements across the map maintained at least a logical pattern, agents adopted a peculiar strategy involving circular movements. This behavior allowed them to balance their total episode reward without incurring significant losses: by moving in circles, agents alternated between slightly negative rewards when moving away from the goal, and slightly positive rewards when moving towards the goal. This kind of strategy is sub-optimal for the agent, but it represents a form of *mode collapse*, where the agent becomes trapped in a local minimum of the reward function. This behavior prevents the agent from learning the correct policy for consistently succeeding in each episode.

Another notable behavior is the strong dependence of episode outcomes on the initial policy initialization. Cases where the agent radically changes its direction to actively go towards the goal are rare. To address this, I experimented with several modifications on the environment reward system, as a hint in the exercise suggested that the *gymnav* environment suffered from a design flaw, which was causing issues when using REINFORCE.



Provo a fare qualche modifica:
- Modifying `FORWARD_REWARD` from 2 to -0.1 (from slightly positive to slightly negative)
- Modifying `BACKWARD_REWARD` from -1 to -2.5
- Removing the *linear shift* from the rotation actions, as every slight rotation would result in a very little positive reward

La configurazione fw = -1 e bw = -2.5 con rotazioni inalterate sembra funzionare abbastanza bene, tuttavia molto spesso si nota il comportamento dell'agente che punta diretto verso il goal per poi sterzare a evitarlo a pochissimi passi da esso.

**Il potenziale design flaw potrebbe essere nel sistema di ricompense, in particolare:**

Ricompensa sparsa: La ricompensa principale (+100) viene data solo quando si raggiunge l'obiettivo. Questo può rendere difficile per l'agente imparare, specialmente all'inizio dell'addestramento quando raggiungere l'obiettivo è raro.
Penalità per passi eccessivi: La penalità di -100 dopo 100 passi potrebbe incoraggiare l'agente a terminare rapidamente l'episodio, anche se ciò significa collidere.
Ricompense intermedie basate su distanza e angolo: Queste ricompense potrebbero non essere sufficientemente informative o potrebbero essere troppo complesse per essere apprese efficacemente.

Il comportamento strano potrebbe manifestarsi come:

L'agente che gira in cerchio o si muove casualmente, evitando di esplorare efficacemente l'ambiente.
L'agente che preferisce collidere rapidamente piuttosto che rischiare la penalità per passi eccessivi.
**L'agente che non riesce a imparare una politica coerente per raggiungere l'obiettivo.**

After checking Fantechi's code, I found that a possible explanation for this weird behavior lies in the fact that forward reward and backward rewards are too different. Another thing is that rotations are associated with small forward linear shift --> rewards! That brings the agent sometimes at moving in circle.

There are many other things that can be improved in this example:

1. **Replay**. In the current implementation we execute an episode, and then immediately run an optimization step on all of the steps of the episode. Not only are we using *correlated* samples from a single episode, we are decidedly *not* taking advantage of parallelism via batch gradient descent. Note that `REINFORCE` does **not** require entire trajectories, all we need are the discounted rewards and log probabilities for *individual transitions*.

2. **Exploration**. The model is probably overfitting (or perhaps remaining too *plastic*, which can explain the unstable convergence). Our policy is *always* stochastic in that we sample from the output distribution. It would be interesting to add a temperature parameter to the policy so that we can control this behavior, or even implement a deterministic policy sampler that always selects the action with max probability to evaluate the quality of the learned policy network.

3. **Discount Factor**: The discount factor (default $\gamma = 0.99$) is an important hyperparameter that has an effect on the stability of training. Try different values for $\gamma$ and see how it affects training. Can you think of other ways to stabilize training?

---
### Exercise 3.2: Solving another environment

After working in a custom environment, I really wanted to try some of the environments available in the [Gymnasium](https://gymnasium.farama.org/) framework, which provides a consistent interface to a broad range of Reinforcement Learning environments. I thought this could be a good time to compare some DRL architectures on different environments.

So, in this section, I will compare the performances of REINFORCE and Deep Q-Learning algorithms in solving two of the most common OpenAI Gymnasium environments: [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/), and [Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/).

To set things up, I firstly implement a lander that takes totally random actions at each time tick. Obviously, the total reward will be very bad.

**random run on Lander**

Now I will try to use the REINFORCE algorithm: running the two versions of it made me find out that the episode-wise REINFORCE is the only one that "works" in this task, as opposite to the CartPole environment, which showed better results with the interaction-wise REINFORCE.

**REINFORCE on Lander**

And finally, the Deep Q-Learning technique.

**DQN on Lander**

To complete the experience, let's try to solve also the CartPole environment.

**REINFORCE and DQN on CartPole**

---
### Exercise 3.3: Advanced techniques 

The `REINFORCE` and Q-Learning approaches, though venerable, are not even close to the state-of-the-art. Nowadays, one of the most powerful approaches for solving DRL environments is [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347).

Curious about its implementation, I decided to use [this off-the-shelf implementation of PPO]() to solve the Lunar Lander environment and compare my results with those of Q-Learning and REINFORCE.

PPO uses the Actor-Critic approach for the agent. This means that it uses two models, one called the Actor and the other called Critic. The Actor model performs the task of learning what action to take under a particular observed state of the environment. In the LunarLander case, it takes eight values list of the game as input which represents the current state of our rocket and gives a particular action what engine to fire as output.
