# Deep Learning Applications: Laboratory #1 - CNNs

In this first laboratory we will work relatively simple architectures to get a feel for working with Deep Models. This notebook is designed to work with PyTorch.

## Exercise 1: Warming Up
In this series of exercises I will duplicate (on a small scale) the results of the ResNet paper:

> [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, CVPR 2016.

I will do this in steps, firstly using a Multilayer Perceptron on MNIST.

What's important to recall is that the main message of the ResNet paper is that **deeper networks do not guarantee** more reduction in training loss (or in validation accuracy).
Below, I will incrementally build a sequence of experiments to verify this for different architectures, starting with an *MLP*.

The Laboratory requires me to compare multiple training runs, so I took this as a great opportunity to learn to use [Weights and Biases](https://wandb.ai/site) for performance monitoring.

### Exercise 1.1: A baseline MLP

I will now implement a *simple Multilayer Perceptron* to classify the 10 digits of MNIST, and (hopefully) train it to convergence, monitoring Training and Validation losses and accuraces with W&B.

The exercise wants me to think in an *abstract* way: I'll have to instantiate multiple models, with different hyperparameters configurations each, and train them on different datasets.
It could be a good idea to try to generalize the most possible the instantiation of every object of the training workflow. That's why I decided to try to build a single file `config.yaml`, where I put almost every variable that can help me building any model I want.

I then define a `load` function, that passes the dictionary `config` (obtained from my `.yaml` file) as an argument, in order to load the dataset we want (between MNIST and CIFAR10), transformed accordingly, and splitted into *Train*, *Validation* and *Test* sets.

The script file `models.py` contains all the model classes used for this Laboratory:
+ **MLP**, for instantiating a *Multilayer Perceptron*
+ **ResidualMLP**, for instantiating an MLP that implements *Residual Connections*
+ **CNN**, for instantiating *Convolutional Network*, with the possibility of tuning almost every possible parameter
+ **ResidualCNN**, for instantiating a ConvNet that implements *Residual Connections*
+ **ResNet**, for instantiating an actual *ResNet* as defined in the [Paper](https://arxiv.org/abs/1512.03385), in its *[9, 18, 34, 50, 101, 152]* versions.

The `build_model` function instantiates Model, Loss Function and Optimizer chosen with the `config` file, and sends it to `device`, that can be `cuda` (in my case, a *Nvidia GeForce RTX 3060 Laptop*) or `cpu`.

The training loop lies in the `train` function, that takes all the objects instantiated in the previous steps and uses them to train the model.

The *forward* and *backward* passes are performed batch-wise through the `train_batch` function, that implements a tweak to reshape the input images' sizes accordingly to the model used. Same thing is done in the `validation` and `test` functions.

The `load`, `build_model`, `train` and `test` functions are all contained in a single function, `model_pipeline`, that allows me to wrap all my workflow into a *Weights & Biases* run more efficiently.

### Exercise 1.2: Rinse and Repeat

I will now repeat the verification I did above, but with **Convolutional** Neural Networks.
This specific part of the exercise focuses on revealing that **deeper** CNNs *without* residual connections do not always work better, and **even deeper** ones *with* residual connections.

**Note**: MNIST is *very* easy to work on (at least up to about 99% accuracy), so I will work on **CIFAR10** from now on.

Launching the `model_pipeline` function with its proper configuration allows me to observe the performance of multiple kinds of Convolutional architectures.

The focus, here, is on playing with the total **depth** (i.e. the number of layers) of the network, while maintaining the general architecture untouched, in order to show that a **deeper** ConvNet provides better performances, **up to a certain depth (!)**.

All logs and trackings of my runs are available on Weights & Biases, at [this link](https://wandb.ai/giovancombo/DLA_Lab1_CNN?workspace=user-giovancombo).

...Well, as previously said, reaching a very high Validation Accuracy on **MNIST** is *very* easy, and doesn't allow us to appreciate at the fullest the differences between different models.
Let's try then to train some models on the **CIFAR10** dataset.

---
## Exercise 2: A Deeper Understanding on Visual Tasks

Let's now deepen our understanding of Deep Networks for visual recognition.

+ Firstly, I will find a quantitative answer about *how* and *why* Redidual Networks learn more efficiently than their Convolutional counterparts.
+ Secondly, I will become a *network surgeon*, trying to fully-convolutionalize a network by acting on its final layers.
+ Thirdly, I will try to implement *Class Activation Maps*, in order to see which parts of an image were the most decisive for its classification.

### Exercise 2.1: Explain why Residual Connections are so effective

The question *"Why Residual Networks learn more efficiently than Convolutional Networks?"* can find an answer by looking at the gradient magnitudes passing through the networks, during backpropagation.

`wandb.watch(log = "all")` tells *Weights & Biases* to log *gradients* and *parameters*' evolution in all the layers of the network. This functionality is useful to graphically visualize the concept of **Vanishing Gradients**.

For this exercise, I firstly tried to run a basic *MLP*, and then an *MLP with Residual Connections*. Honestly, at the time, I didn't think that this could be a very clever idea, since I've always seen Residuals been added only on Convolutional Networks, but... I decided to give it a try anyway.

As mentioned before, I compared these two architectures by challenging them on their performance over their **depth** (i.t. their number of layers).

A basic **10-layer MLP** is seen suffering from Vanishing Gradients, with its accuracy dropping all the way down to 10%, that means picking a class **by chance**.

As mentioned in the original [ResNet paper](https://arxiv.org/abs/1512.03385), a higher number of layers leads to not only higher validation loss, but also a *higher training loss*: this means that we are not facing overfitting, but in the "weird" behavior that a deeper model shows itself.

On the contrary, the **10-layer Residual MLP** performed well, confirming the explanation of ResNet authors: Residual Connections allow a network to go **a lot** deeper (with the only limitation of reaching overfitting).

The results can be quantitatively checked by observing the *W&B* logs about gradient magnitudes. The basic **MLP** shows gradients that are very close to zero, meaning that the model is not making any real progress.

Conversely, the **Residual MLP** showed gradients that did not vanish nor explode, and progressively diminishing their magnitude during training, meaning that the model is proceeding towards convergence on a (local, hopefully global) optimum.

---
### Exercise 2.2: Fully-convolutionalize a network.
Take one of your trained classifiers and **fully-convolutionalize** it. That is, turn it into a network that can predict classification outputs at *all* pixels in an input image. Can you turn this into a **detector** of handwritten digits? Give it a try.

**Hint 1**: Sometimes the process of fully-convolutionalization is called "network surgery".

**Hint 2**: To test your fully-convolutionalized networks you might want to write some functions to take random MNIST samples and embed them into a larger image (i.e. in a regular grid or at random positions).

The ConvNets built in the previous exercise have a global Average Pooling layer and a Fully Connected Layer at the end, in order to merge all infro from the convolutions in a single prediction for all the image, on the 10 MNIST/CIFAR10 classes.

In a Fully Convolutional Network, we need instead to produce a prediction for every single one of the 28x28 (32x32) pixels of an image. I then proceed to do a "network surgery", removing the two layers mentioned above and rearranging the net to have the dimension of the input image as output.

---
### Exercise 2.3: *Explain* the predictions of a CNN

Use the CNN model you trained in Exercise 1.2 and implement [*Class Activation Maps*](http://cnnlocalization.csail.mit.edu/#:~:text=A%20class%20activation%20map%20for,decision%20made%20by%20the%20CNN.):

> B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, and A. Torralba. Learning Deep Features for Discriminative Localization. CVPR'16 (arXiv:1512.04150, 2015).

Use your implementation to demonstrate how your trained CNN *attends* to specific image features to recognize *specific* classes.
