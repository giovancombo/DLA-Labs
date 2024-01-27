# Deep Reinforcement Learning Laboratory

In this laboratory session we will hack one of your colleague's (Francesco Fantechi, from Ingegneria Informatica) implementation of a navigation environment for Deep Reinforcement Learning. The setup is fairly simple:

+ A simple 2D environment with a (limited) number of *obstacles* and a single *goal* is presented to the agent, which must learn how to navigate to the goal without hitting any obstacles.
+ The agent *observes* the environment via a set of 16 rays cast uniformly which return the distance to the first obstacle encountered, as well as the distance and direction to the goal.
+ The agent has three possible actions: `ROTATE LEFT`, `ROTATE RIGHT`, or `MOVE FORWARD`.

For each step of an episode, the agent receives a reward of:
+ -100 if hitting an obstacle (episode ends).
+ -100 if one hundred steps are reached without hitting the goal.
+ +100 if hitting the goal (episode ends)
+ A small *positive* reward if the distance to the goal is *reduced*.
+ A small *negative* reward if the distance to the goal is *increased*.

In the file `main.py` you will find an implementation of **Deep Q-Learning**.

## Exercise 1: Testing the Environment

The first thing to do is verify that the environment is working in your Anaconda virtual environment. I had a weird problem with Tensorboard and had to downgrade it using:

    conda install -c conda-forge tensorboard=2.11.2
    
In any case, you should be able to run:

    python main.py
    
from the repository root and it will run episodes using a pretrained agent. To train an agent from scratch, you must modify `main.py` setting `TRAIN = True` at the top. Then running `main.py` again will train an agent for 2000 episodes of training. To run the trained agent you will again have to modify `main.py` on line 225 to load the last saved checkpoint:

    PATH = './checkpoints/last.pth'
    
and then run the script again (after setting `TRAIN = False` !).

Make sure you can at run the demo agent and train one from scratch. If you don't have a GPU you can set the number of training episodes to a smaller number.

## Exercise 2: Stabilizing Q-Learning



## Exercise 3: Going Deeper

As usual, pick **AT LEAST ONE** of the following exercises to complete.

### Exercise 3.1: Solving the environment with `REINFORCE`

Use my (or even better, improve on my) implementation of `REINFORCE` to solve the environment.

**Note**: There is a *design flaw* in the environment implementation that will lead to strange (by explainable) behavior in agents trained with `REINFORCE`. See if you can figure it out and fix it.

### Exercise 3.2: Solving another environment

The [Gymnasium](https://gymnasium.farama.org/) framework has a ton of interesting and fun environments to work with. Pick one and try to solve it using any technique you like. The [Lunar Landar](https://gymnasium.farama.org/environments/box2d/lunar_lander/) environment is a fun one. 

### Exercise 3.3: Advanced techniques 

The `REINFORCE` and Q-Learning approaches, though venerable, are not even close to the state-of-the-art. Try using an off-the-shelf implementation of [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) to solve one (or more) of these environments. Compare your results with those of Q-Learning and/or REINFORCE.





# Getting up to speed with DRL

In this notebook I provide a simple example of implementing a policy gradient Deep Reinforcement Learning algorithm to solve a control problem with continuous state space and discrete action space -- the venerable [CartPole environment](https://gymnasium.farama.org/environments/classic_control/cart_pole/). You should study the implementation in this notebook in preparation for the laboratory next Wednesday.

This notebook should run in an environment with at least the following packages installed (the gpu version of PyTorch is not mandatory):

     conda create -n DRL -c conda-forge gymnasium pytorch-gpu matplotlib pygame jupyterlab
     
Some background reading to get you started:

1. We will be using the [Gymnasium](https://gymnasium.farama.org/) framework for all of our experiments. This framework provides a consistent interface to a broad range of reinforcement learning environments (including CartPole). You should familiarize yourself with how it works, how environments are specified, how to instantiate them, and how to interact with them.

2. [This excellent blog post](http://karpathy.github.io/2016/05/31/rl/) is a great introduction to policy gradients, where they come from and how they work. Give it a read and I am sure it will help understand better what is going on in this notebook.

## Preliminaries

We start with our standard imports... And also some utility functions useful for what comes next.

## The Policy network

Here I provide a simple policy network which should work with any environment with continuous observations and discrete action spaces. Note how it uses the *specification* of the environment to configure its input and output spaces. 

## The `REINFORCE` Algorithm

This is a very simple implementation of the most basic policy gradient DRL algorithm: `REINFORCE`. It is a very direct implementation of the policy gradient update (although I use Adam instead of SGD).

## For your consideration

There are many things that can be improved in this example. Some things you can think about:

1. **Replay**. In the current implementation we execute an episode, and then immediately run an optimization step on all of the steps of the episode. Not only are we using *correlated* samples from a single episode, we are decidedly *not* taking advantage of parallelism via batch gradient descent. Note that `REINFORCE` does **not** require entire trajectories, all we need are the discounted rewards and log probabilities for *individual transitions*.

2. **Exploration**. The model is probably overfitting (or perhaps remaining too *plastic*, which can explain the unstable convergence). Our policy is *always* stochastic in that we sample from the output distribution. It would be interesting to add a temperature parameter to the policy so that we can control this behavior, or even implement a deterministic policy sampler that always selects the action with max probability to evaluate the quality of the learned policy network.

3. **Discount Factor**: The discount factor (default $\gamma = 0.99$) is an important hyperparameter that has an effect on the stability of training. Try different values for $\gamma$ and see how it affects training. Can you think of other ways to stabilize training?


