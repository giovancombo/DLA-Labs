# Deep Learning Applications: Laboratory #1 - CNNs

In this first Laboratory I will work with relatively simple architectures to get a feel for working with Deep Models. Convolutional Networks are fundamental tools for image processing, thanks to their ability to analyse input data through different levels of abstraction.

This Lab is essentially a brief analysis of how ConvNets work, which are their drawbacks, how can they be improved, and employed for the most common visual tasks.

## Exercise 1: Warming Up
In this series of exercises I will duplicate (on a small scale) the results of the notorious ResNet paper, by Kaiming He et al.:

> [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun; CVPR 2016.

It's important to recall that the main message of the ResNet paper is **deeper networks do not guarantee more reduction in Training Loss**. While the primary focus of many researchers was trying to build the deepest network possible, He et al. demonstrated that Training and Validation Accuracies increase accordingly with the model depth, *but only until a certain point*, after which they start to decrease.

I will incrementally build a sequence of experiments to verify and explain this behaviour for different architectures, firstly using a *Multilayer Perceptron* on MNIST, and finally *Convolutional Neural Networks* with Residual Connections on CIFAR-10.

Since this Lab requires me to compare multiple training runs, I took this as a great opportunity to learn [Weights & Biases](https://wandb.ai/site) for performance monitoring. A report of this Lab will soon be available on my Weights & Biases profile.

### Exercise 1.1: A baseline MLP

I will now implement a *simple* Multilayer Perceptron to classify the 10 digits of MNIST, and (hopefully) train it to convergence, monitoring Training and Validation Losses and Accuracies using *Weights & Biases*.

A fundamental skill in programming is being able to think in an *abstract* way: I'll have to instantiate multiple models, with different hyperparameters configurations each, and train them on different datasets. So, it could be a good idea to generalize the most possible the instantiation of every object of the training workflow. That's why I decided to create a `config.yaml` file, where I put every hyperparameter I need to set about architectures, datasets and training stuff. I just found it easier for me.

To facilitate the future reuse of code for different architectures, I implemented the `trainer.py` script, which defines a **Trainer** object that contains `train` and `evaluate` functions for running the training loop.

**Note:** For all models, I decided to fix the layer width to **64**, so, an MLP will always have layers 64 nodes wide, a CNN will have 64 kernels in every feature map, and so on. I used **ReLU** wherever there's an activation function. I use the **Adam** optimizer with **1e-4** learning rate for **20** epochs of training, and a dropout factor of **0.2**. The batch size is always set at **128**. The `device` used for computation is **cuda** (in my case, a *Nvidia GeForce RTX 3060 Laptop*).

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/21_runs/mnist_mlp_valacc.png" width="49%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/21_runs/mnist_mlp_valloss.png" width="49%" />
</p>
<p align="center"><i><b>Figure 1</b> | Comparison between 1, 5, 10 and 20 layers deep MLPs on MNIST: Validation Accuracy <b>(a)</b>, Validation Loss <b>(b)</b></i></p>

*Figure 1* shows the comparison between **1, 5, 10** and **20** layers deep Multilayer Perceptrons: here we can already see how *deeper* models do not reach better results. Indeed, the 10-layer MLP performs worse than its 1 and 5-layer counterpart, while the 20-layer MLP totally fails to learn! It takes just a shallower MLP to easily reach convergence to relatively high values of accuracy on MNIST.

### Exercise 1.2: Rinse and Repeat

I will now make a step forward, training some **Convolutional Neural Networks**.

After building the simple MLP used previously, I updated the `models.py` script adding all the model classes used for this Laboratory:
+ **MLP**, that instantiates a *Multilayer Perceptron*
+ **ResidualMLP**, that instantiates an MLP with *Residual Connections*
+ **CNN**, that instantiates a *Convolutional Network*
+ **ResidualCNN**, that instantiates a ConvNet with *Residual Connections*
+ **ResNet**, that instantiates an actual *ResNet* as proposed in the [Paper](https://arxiv.org/abs/1512.03385), available in its *[9, 18, 34, 50, 101, 152]* versions.

This specific part of the exercise focuses on revealing that **deeper** CNNs *without* Residual Connections do not always work better, and **even deeper** ones *with* Residual Connections. But since MNIST is a *very* easy dataset to work on (at least up to about 99% accuracy), I will soon start to work on the **CIFAR10** dataset.

The focus, here, is on playing with the total **depth** (i.e. the number of layers) of the network, while maintaining the general architecture untouched, in order to show that **deeper** ConvNet provides better performances, **only up to a certain depth**. So, I decided to compare **1, 5, 10, 20, 30** and **50** layers deep ConvNets, every single layer having the same width of **64**.

All logs and trackings of my runs are available on Weights & Biases, at [this link](https://wandb.ai/giovancombo/DLA_Lab1_CNN?workspace=user-giovancombo).

I launch new runs with CNNs chenging the *dataset* and *architecture* parameters in the `config.yaml` file and calling the `train` and `evaluate` functions of the **Trainer** object, as previously made with the MLPs.

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/21_runs/mnist_cnn_valacc.png" width="49%">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/21_runs/mnist_cnn_valloss.png" width="49%">
</p>
<p align="center"><i><b>Figure 2</b> | Comparison between 1, 5, 10, 20 and 30 layers deep ConvNets on MNIST: Validation Accuracy <b>(a)</b>, Validation Loss <b>(b)</b></i></p>

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/21_runs/mnist_valacc.png" width="49%">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/21_runs/mnist_valloss.png" width="49%">
</p>
<p align="center"><i><b>Figure 3</b> | Comparison between ConvNets and MLPs (dashed) with different depths on MNIST: Validation Accuracy <b>(a)</b>, Validation Loss <b>(b)</b></i></p>

Well... As previously said, reaching a very high Validation Accuracy on **MNIST** is *very* easy for almost every kind of model regardless of depth: all those similar results close to 99% (*Figure 2*) don't allow me to appreciate to the fullest how differently models perform their learning on data.

Let's try then to train some models on the **CIFAR-10** dataset.

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/21_runs/cifar_cnn_valacc.png" width="49%">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/21_runs/cifar_cnn_valloss.png" width="49%">
</p>
<p align="center"><i><b>Figure 4</b> | Comparison between 1, 5, 10, 20, 30 and 50 layers deep ConvNets on CIFAR10: Validation Accuracy <b>(a)</b>, Validation Loss <b>(b)</b></i></p>

**CIFAR10** is a more complex dataset than MNIST to learn, so reaching very high accuracies is not that easy. This allows us to appreciate the *mantra* of Kaiming He paper on Convolutional models for the first time.

*Figure 4* shows how a 1-layer deep ConvNet is too simple to handle this kind of images, and adding new parameters and feature maps to the model (i.e. going deeper!) crucially improves performance, peaking at the 10-layer deep ConvNet. Things start to change beyond that depth, as deeper networks (20, 30 and 50 layers deep) perform progressively worse. Interestingly, not only *Validation* metrics get worse, but also *Training* metrics, suggesting that this behavior is caused by something else than overfitting.

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/21_runs/cifar_valacc.png" width="49%">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/21_runs/cifar_valloss.png" width="49%">
</p>
<p align="center"><i><b>Figure 5</b> | Comparison between ConvNets and MLPs (dashed) with different depths on CIFAR10: Validation Accuracy <b>(a)</b>, Validation Loss <b>(b)</b></i></p>

#### Implementing ResidualCNNs: CNNs with Residual Connections

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/21_runs/cifar_rescnn_valacc.png" width="49%">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/21_runs/cifar_rescnn_valloss.png" width="49%">
</p>
<p align="center"><i><b>Figure 6</b> | Comparison between 1, 5, 10, 20, 30 and 50 layers deep ResidualCNNs on CIFAR10: Validation Accuracy <b>(a)</b>, Validation Loss <b>(b)</b></i></p>

The **ResidualCNN** class of models imitates the **CNN** class structure, but implements residual connections. *Figure 6* shows that Convolutional models that implement residual connections improve their performance accordingly with depth.

---
## Exercise 2: A Deeper Understanding on Visual Tasks

Classification is just one of the many tasks Convolutional Networks can address. So, I will now deepen my understanding of Deep Networks for visual recognition.

+ Firstly, I will quantitative discover and explain *how* and *why* Redidual Networks learn more efficiently than their Convolutional counterparts.
+ Secondly, I will become a *network surgeon*, fully-convolutionalizing a network by acting on its final layers, in order to perform Segmentation.
+ Thirdly, I will implement *Class Activation Maps*, that allow me to see which parts of an image were the most informative for its classification.

### Exercise 2.1: Explain why Residual Connections are so effective

*"Why Residual Networks learn more efficiently than Convolutional Networks?"*

This question can find an answer by looking at the **Gradient Magnitudes** passing through the layers of the networks, during Backpropagation.

As mentioned in the original [ResNet paper](https://arxiv.org/abs/1512.03385), a higher number of layers leads to not only higher Validation Loss, but also a *higher Training Loss*. The fact that deeper networks lead to higher **Training** Loss (not only Validation Loss!) proves that this particular phenomenon is not simple *overfitting*, but it's related to the actual excessive model complexity.

`wandb.watch(log = "all")` tells *Weights & Biases* to log the evolution of *gradients* and *parameters* in all the layers of a network.

<p float="center" align="center">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/21_runs/grad_cnn1.png" width="49%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/21_runs/grad_rescnn1.png" width="49%" />
</p>
<p align="center"><i><b>Figure 7</b> | Gradient evolution in the only layer of <b>CNN-1</b> and <b>ResidualCNN-1</b></i></p>

<p float="center" align="center">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/21_runs/grad_cnn50.png" width="49%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/21_runs/grad_rescnn50.png" width="49%" />
</p>
<p align="center"><i><b>Figure 8</b> | Gradient evolution in the first layer of <b>CNN-50</b> and <b>ResidualCNN-50</b></i></p>

The **Vanishing Gradient problem** in deep CNNs occurs when gradients become extremely small as they're backpropagated through many layers, making it difficult for earlier layers to learn effectively. In standard *CNNs*, this can lead to degraded performance as network depth increases. *ResidualCNNs* mitigate this issue by introducing skip connections that allow gradients to flow directly through the network, maintaining stronger gradient signals even in very deep architectures.

While *Figure 7* shows that shallow CNNs can perfectly compete with (and even perform better than) shallow ResidualCNNs, *Figure 8* shows the central issue that differentiates CNNs and ResidualCNNs. The gradient magnitudes of the CNNs (on the left) start with very high values at the beginning of training, and then rapidly tend to zero, falling in the Vanishing Gradient Problem.

On the contrary, ResidualCNNs (on the right) does not suffer from any Vanishing nor Exploding problems, as gradients are stable through all the training process, guaranteeing continuous learning.

---
### Exercise 2.2: Fully-convolutionalize a network (WORK IN PROGRESS)

**Fully-convolutionalizing** a model means turning it into a network that can predict classification outputs at *all* pixels in an input image. This reframing unlocks many new visual tasks that can be addressed, such as *Semantic Segmentation*, *Object Detection* and *Object Recognition*.

The ConvNets built in the previous exercise have a global Average Pooling layer and a Fully Connected Layer (e.g. the *Classification Head*) at the end, in order to merge all info from the convolutions in a single output prediction for all the image, on the 10 MNIST/CIFAR-10 classes.

In a Fully Convolutional Network, we need instead to produce a prediction for every single one of the 28x28 (32x32) pixels of an image. I then proceed to do a "network surgery", removing the two layers mentioned above and rearranging the net to have the dimension of the input image as output.

In this section, I will turn one of the ConvNets I trained into a **detector** of handwritten MNIST digits. In order to test my new network, I need to write some functions to take random MNIST samples and embed them into a larger image (i.e. in a regular grid or at random positions).

---
### Exercise 2.3: *Explain* the predictions of a CNN

> [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150), B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, and A. Torralba; CVPR 2016.
> 
[*Class Activation Maps*](http://cnnlocalization.csail.mit.edu/#:~:text=A%20class%20activation%20map%20for,decision%20made%20by%20the%20CNN.) are very powerful tools for understanding how Neural Networks learn in order to classify objects in an image.

The main focus, here, is to see how one of the previously trained CNNs *attends* to specific image features to recognize *specific* classes.

Class Activation Maps were here implemented taking inspiration from [this tutorial](https://medium.com/intelligentmachines/implementation-of-class-activation-map-cam-with-pytorch-c32f7e414923).

For this task, I trained a 50-layer ResidualCNN for 50 epochs, in order to reach convergence to a higher Validation Accuracy than all models that were trained in Exercise 1 (*Figure 10*).

<p float="center" align="center">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/model_for_CAMs_valacc.png" width="48%" />
</p>
<p align="center"><i><b>Figure 10</b> | Validation Accuracy of the <b>ResidualCNN</b> trained for this Exercise</i></p>

#### CAMs on CIFAR10 images

Firstly, I did some evaluations on the 3-channels 32x32 CIFAR10 images (here I show some of them, other can be found in the `images/23_cam` folder).

<p float="center">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_data/cifar_idx0_cat.jpg" width="12%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_cam/CAM_cifar_idx0_bird_probs0.4601.jpg" width="12%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_data/cifar_idx1_ship.jpg" width="12%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_cam/CAM_cifar_idx1_automobile_probs0.9972.jpg" width="12%" /> 
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_data/cifar_idx4_frog.jpg" width="12%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_cam/CAM_cifar_idx4_frog_probs1.0000.jpg" width="12%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_data/cifar_idx6_automobile.jpg" width="12%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_cam/CAM_cifar_idx6_automobile_probs0.6085.jpg" width="12%" /> 
</p>
<p align="center"><i><b>(1)</b> <b>Cat</b> pred <b>Bird</b> (0.4601);&emsp;<b>(2)</b> <b>Ship</b> pred <b>Automobile</b> (0.9972);&emsp;<b>(3)</b> <b>Frog</b> pred <b>Frog</b> (1.0000);&emsp;<b>(4)</b> <b>Automobile</b> pred <b>Automobile</b> (0.6085)</i></p>

<p float="center">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_data/cifar_idx8_cat.jpg" width="12%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_cam/CAM_cifar_idx8_dog_probs0.5319.jpg" width="12%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_data/cifar_idx9_automobile.jpg" width="12%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_cam/CAM_cifar_idx9_automobile_probs0.9821.jpg" width="12%" /> 
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_data/cifar_idx10_airplane.jpg" width="12%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_cam/CAM_cifar_idx10_airplane_probs0.9438.jpg" width="12%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_data/cifar_idx11_truck.jpg" width="12%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_cam/CAM_cifar_idx11_truck_probs1.0000.jpg" width="12%" /> 
</p>
<p align="center"><i><b>(5)</b> <b>Cat</b> pred <b>Dog</b> (0.5319);&emsp;<b>(6)</b> <b>Automobile</b> pred <b>Automobile</b> (0.9821);&emsp;<b>(7)</b> <b>Airplane</b> pred <b>Airplane</b> (0.9438);&emsp;<b>(8)</b> <b>Truck</b> pred <b>Truck</b> (1.0000)</i></p>

<p float="center">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_data/cifar_idx34_truck.jpg" width="12%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_cam/CAM_cifar_idx34_truck_probs1.0000.jpg" width="12%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_data/cifar_idx35_bird.jpg" width="12%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_cam/CAM_cifar_idx35_frog_probs0.6957.jpg" width="12%" /> 
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_data/cifar_idx40_deer.jpg" width="12%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_cam/CAM_cifar_idx40_deer_probs0.5564.jpg" width="12%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_data/cifar_idx39_dog.jpg" width="12%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_cam/CAM_cifar_idx39_dog_probs0.9996.jpg" width="12%" /> 
</p>
<p align="center"><i><b>(9)</b> <b>Truck</b> pred <b>Truck</b> (1.0000);&emsp;<b>(10)</b> <b>Bird</b> pred <b>Frog</b> (0.6957);&emsp;<b>(11)</b> <b>Deer</b> pred <b>Deer</b> (0.5564);&emsp;<b>(12)</b> <b>Dog</b> pred <b>Dog</b> (0.9996)</i></p>

<p float="center">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_data/cifar_idx17_horse.jpg" width="12%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_cam/CAM_cifar_idx17_horse_probs0.4215.jpg" width="12%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_data/cifar_idx26_deer.jpg" width="12%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_cam/CAM_cifar_idx26_bird_probs0.6547.jpg" width="12%" /> 
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_data/cifar_idx33_dog.jpg" width="12%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_cam/CAM_cifar_idx33_frog_probs0.7481.jpg" width="12%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_data/cifar_idx48_horse.jpg" width="12%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/cifar_cam/CAM_cifar_idx48_horse_probs0.8355.jpg" width="12%" /> 
</p>
<p align="center"><i><b>(13)</b> <b>Horse</b> pred <b>Horse</b> (0.4215);&emsp;<b>(14)</b> <b>Deer</b> pred <b>Bird</b> (0.6547);&emsp;<b>(15)</b> <b>Dog</b> pred <b>Frog</b> (0.7481);&emsp;<b>(16)</b> <b>Horse</b> pred <b>Horse</b> (0.8355)</i></p>

#### CAMs on my photographs

Secondly, as a photographer I couldn't resist to compute CAMs on some of my photographs! But as they are 4000x3000 (huge) images, I had to adjust scaling factors between images and CAMs: I resized every image to a width of 32 (I kept their rectangular shapes) to get the prediction from the model (trained on 32x32 CIFAR10 images), before resizing again CAM and image to width 256.

In order to test how well the model performs, I chose images with "obvious" classes (like 1, 2, 9, 16, 17 and 18), and images that, instead, I thought could have been more challenging. Some results are very interesting:

+ **(3 and 4)**: vintage cars are predicted as *trucks*, probably because the model sees something with four tyres but with a very unconventional shape.
+ **(5)**: these are *goats*, so they are objects that don't fit in any of the CIFAR10 classes. Yet, antlers are decisive in classifying them as *deers*.
+ **(10)**: the model recognizes an airplane even if you are inside the airplane, the wing is sufficient.
+ **(11)**: again a subject that doesn't fit in any CIFAR10 category: this macaque was predicted as *deer*.
+ **(12)**: I thought the model would have predicted this statue as *dog*, but eventually something went wrong.
+ **(14)**: I was sure the model would have correctly classified this image due to the sailboat, not those yachts in the background.

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my1horse.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my1horse_horse_probs0.9252.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my2automobile.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my2automobile_automobile_probs0.9999.jpg" width="24%" /> 
</p>
<p align="center"><i><b>(1)</b> Class: <b>Horse</b>; Predicted: <b>Horse</b> (0.9252)&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<b>(2)</b> Class: <b>Automobile</b>; Predicted: <b>Automobile</b> (0.9999)</i></p>

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my3automobile.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my3automobile_truck_probs0.6790.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my4automobile.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my4automobile_truck_probs0.9844.jpg" width="24%" /> 
</p>
<p align="center"><i><b>(3)</b> Class: <b>Automobile</b>; Predicted: <b>Truck</b> (0.6790)&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<b>(4)</b> Class: <b>Automobile</b>; Predicted: <b>Truck</b> (0.9844)</i></p>

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my5nd.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my5nd_deer_probs0.8761.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my6bird.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my6bird_horse_probs0.6518.jpg" width="24%" /> 
</p>
<p align="center"><i><b>(5)</b> Class: <b>XXX</b>; Predicted: <b>Deer</b> (0.8761)&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<b>(6)</b> Class: <b>Bird</b>; Predicted: <b>Horse</b> (0.6518)</i></p>

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my7cat.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my7cat_airplane_probs0.7187.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my8ship.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my8ship_airplane_probs0.6052.jpg" width="24%" /> 
</p>
<p align="center"><i><b>(7)</b> Class: <b>Cat</b>; Predicted: <b>Airplane</b> (0.7187)&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<b>(8)</b> Class: <b>Ship</b>; Predicted: <b>Airplane</b> (0.6052)</i></p>

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my9truck.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my9truck_truck_probs0.9999.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my10airplane.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my10airplane_airplane_probs0.3945.jpg" width="24%" /> 
</p>
<p align="center"><i><b>(9)</b> Class: <b>Truck</b>; Predicted: <b>Truck</b> (0.9999)&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<b>(10)</b> Class: <b>Airplane</b>; Predicted: <b>Airplane</b> (0.3945)</i></p>

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my11nd.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my11nd_deer_probs0.9759.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my12dog.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my12dog_frog_probs0.8742.jpg" width="24%" /> 
</p>
<p align="center"><i><b>(11)</b> Class: <b>XXX</b>; Predicted: <b>Deer</b> (0.9759)&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<b>(12)</b> Class: <b>Dog</b>; Predicted: <b>Frog</b> (0.8742)</i></p>

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my13dog.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my13dog_bird_probs0.3523.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my14ship.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my14ship_ship_probs0.8284.jpg" width="24%" /> 
</p>
<p align="center"><i><b>(13)</b> Class: <b>Dog</b>; Predicted: <b>Bird</b> (0.3523)&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<b>(14)</b> Class: <b>Ship</b>; Predicted: <b>Ship</b> (0.8284)</i></p>

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my15automobile.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my15automobile_frog_probs0.7401.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my16bird.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my16bird_bird_probs0.9999.jpg" width="24%" /> 
</p>
<p align="center"><i><b>(15)</b> Class: <b>Automobile</b>; Predicted: <b>Frog</b> (0.7401)&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<b>(16)</b> Class: <b>Bird</b>; Predicted: <b>Bird</b> (0.9999)</i></p>

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my17cat.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my17cat_cat_probs0.5190.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my18ship.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my18ship_ship_probs0.9126.jpg" width="24%" /> 
</p>
<p align="center"><i><b>(17)</b> Class: <b>Cat</b>; Predicted: <b>Cat</b> (0.5190)&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<b>(18)</b> Class: <b>Ship</b>; Predicted: <b>Ship</b> (0.9126)</i></p>
