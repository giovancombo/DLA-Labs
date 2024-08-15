# Deep Learning Applications: Laboratory #3 - DRL

In this Laboratory I will explore the land of Deep Reinforcement Learning and its application to Navigation problems and simulations. Some of the most important theoretical concepts about this Lab were learned from [this excellent Andrej Karpathy blog post](http://karpathy.github.io/2016/05/31/rl/).

For this Lab, I will create a new *conda* environment named **DRL**, in order to work with the great number of libraries that are specifically dedicated to DRL tasks.

## Exercise 1: Testing the Environment
Part of the experience will include "hacking" a provided implementation (provided by Francesco Fantechi, from Ingegneria Informatica) of a navigation environment. The setup is fairly simple:

+ A simple 2D environment with a (limited) number of *obstacles* (yellow squares) and a single *goal* (blue circle) is presented to the agent (green dot), which must learn how to navigate to the goal without hitting any obstacles.
+ The agent *observes* the environment via a set of 16 rays cast uniformly which return the distance to the first obstacle encountered, as well as the distance and direction to the goal.
+ The agent has three possible actions: `ROTATE LEFT`, `ROTATE RIGHT`, or `MOVE FORWARD`.

For each step of an episode, the agent receives a reward of:
+ -100 if hitting an obstacle (the episode ends).
+ -100 if one hundred steps are reached without hitting the goal (time's up, the episode ends).
+ +100 if hitting the goal (the episode ends)
+ A small *positive* reward if the distance to the goal is *reduced*.
+ A small *negative* reward if the distance to the goal is *increased*.

The **Deep Q-Learning** implementation for solving this exercise can be found in the `main.py` script.

Running the script will start running episodes using a pre-trained agent. Fantechi's clever implementation allows me to change mode from Training to Testing only by modifying the `TRAIN` flag at the top.

Firstly, I train and save the trained agent using DQL setting `TRAIN = True`. I initially decide to train the agent for 2000 episodes, but soon I discover that the training is very slow, so I reduced them to 1000.

Then, I run `main.py` again with `TRAIN = False` and the last saved checkpoint loaded to see how the agent performs.

**RUNS**

Qualitatively, it's possible to see that the agents looks like it has not learned so well to find the goal. Many times, the agent hits the walls or obstables without even trying to change direction, even just after the spawn. Some other times, the agent finds its way to the goal, until it stops right in front of it and changes direction.

---
## Exercise 3: Going Deeper

### Exercise 3.1: Solving the environment with `REINFORCE`

REINFORCE is one of the first Policy Gradient DRL algorithms for training an agent. Since it is also one of the most simple ones, I decided not only to use professor Bagdanov's implementation of REINFORCE, but to implement my own version following a tutorial.

**Note**: There is a *design flaw* in the environment implementation that will lead to strange (by explainable) behavior in agents trained with `REINFORCE`. See if you can figure it out and fix it.

After multiple runs using REINFORCE in the Navigation environment, I could see a strange behavior in agents. After some first tries in which agents struggled to reach the goal, but at least maintained a logic in their movements across the map, agents just started to employ a new particular strategy that involved moving in circle in order to balance their total episode reward with no more losses: moving in circle, in fact, allowed agents to get slightly negative rewards while moving away from the goal, compensated by slightly positive rewards for moving towards the goal. This kind of strategy is sub-optimal for the agent, but it's actually a mode collapse that leads to no improvement in learning the correct policy for succeeding every episode.

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
