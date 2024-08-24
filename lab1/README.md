# Deep Learning Applications: Laboratory #1 - CNNs

In this first Laboratory I will work with relatively simple architectures to get a feel for working with Deep Models. Convolutional Networks are fundamental tools for image processing, thanks to their ability to analyse input data through different levels of abstraction.

This Lab is essentially a brief analysis of how ConvNets work, which are their drawbacks, how can they be improved, and employed for the most common visual tasks.

## Exercise 1: Warming Up
In this series of exercises I will duplicate (on a small scale) the results of the notorious ResNet paper, by Kaiming He et al.:

> [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, CVPR 2016.

What's important to recall is that the main message of the ResNet paper is that **deeper networks do not guarantee more reduction in Training Loss**. While the primary focus of many researchers was trying to build the deepest network possible, He et al. demonstrated that Training and Validation Accuracies increase accordingly with the model depth, *but only until a certain point*, after which they start to decrease.

I will incrementally build a sequence of experiments to verify and explain this behaviour for different architectures, firstly using a *Multilayer Perceptron* on MNIST, and finally *Convolutional Neural Networks* with Residual Connections on CIFAR-10.

Since this Lab requires me to compare multiple training runs, I took this as a great opportunity to learn [Weights & Biases](https://wandb.ai/site) for performance monitoring. A report of this Lab will soon be available on my Weights & Biases profile.

### Exercise 1.1: A baseline MLP

I will now implement a *simple* Multilayer Perceptron to classify the 10 digits of MNIST, and (hopefully) train it to convergence, monitoring Training and Validation Losses and Accuraces with Weights & Biases.

A fundamental skill in programming is being able to think in an *abstract* way: I'll have to instantiate multiple models, with different hyperparameters configurations each, and train them on different datasets. So, it could be a good idea to generalize the most possible the instantiation of every object of the training workflow. That's why I decided to create a `config.yaml` file, where I will put every hyperparameter I will need to set in the Lab regarding architecture, dataset and training stuff. I just found it easier for me.

To facilitate the future reuse of code, I implemented the `trainer.py` script, which defines a **Trainer** object that contains `train` and `evaluate` functions that perform my training loop.

The `device` used for computation can be `cuda` (in my case, an *Nvidia GeForce RTX 3060 Laptop*) or `cpu`.

The `models.py` script contains all the model classes used for this Laboratory:
+ **MLP**, that instantiates a *Multilayer Perceptron*
+ **ResidualMLP**, that instantiates an MLP with *Residual Connections*
+ **CNN**, that instantiates a *Convolutional Network*
+ **ResidualCNN**, that instantiates a ConvNet with *Residual Connections*
+ **ResNet**, that instantiates an actual *ResNet* as proposed in the [Paper](https://arxiv.org/abs/1512.03385), available in its *[9, 18, 34, 50, 101, 152]* versions.

<p float="left">
    <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/21_runs/mnist_mlp_valacc.png" width="49%" />
    <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/21_runs/mnist_mlp_valloss.png" width="49%" />
</p>

### Exercise 1.2: Rinse and Repeat

I will now repeat the verification I did above, but with **Convolutional Neural Networks**.
This specific part of the exercise focuses on revealing that **deeper** CNNs *without* Residual Connections do not always work better, and **even deeper** ones *with* Residual Connections.

**Note**: MNIST is a *very* easy dataset to work on (at least up to about 99% accuracy), so I will soon start to work on the **CIFAR-10** dataset.

Launching the `model_pipeline` function with its proper configuration allows me to observe the performance of multiple architectures. The focus, here, is on playing with the total **depth** (i.e. the number of layers) of the network, while maintaining the general architecture untouched, in order to show that a **deeper** ConvNet provides better performances, **up to a certain depth**. So, I decided to compare *1, 5, 10, 15, 20, 30 and 50* layer-deep ConvNets, every single layer having the same architecture.

All logs and trackings of my runs are interactingly available on Weights & Biases, at [this link](https://wandb.ai/giovancombo/DLA_Lab1_CNN?workspace=user-giovancombo).

Well... As previously said, reaching a very high Validation Accuracy on **MNIST** is *very* easy for almost every type of model, so this setting doesn't allow me to appreciate at the fullest how different models perform their learning: even MLPs perform the same as CNNs.

Let's try then to train some models on the **CIFAR-10** dataset.

---
## Exercise 2: A Deeper Understanding on Visual Tasks

Classification is just one of the many tasks Convolutional Networks can address. So, I will now deepen my understanding of Deep Networks for visual recognition.

+ Firstly, I will quantitative discover and explain *how* and *why* Redidual Networks learn more efficiently than their Convolutional counterparts.
+ Secondly, I will become a *network surgeon*, fully-convolutionalizing a network by acting on its final layers, in order to perform Segmentation.
+ Thirdly, I will implement *Class Activation Maps*, that allow me to see which parts of an image were the most informative for its classification.

### Exercise 2.1: Explain why Residual Connections are so effective

*"Why Residual Networks learn more efficiently than Convolutional Networks?"*

This question can find an answer by looking at the **Gradient Magnitudes** passing through the layers of the networks, during Backpropagation.

`wandb.watch(log = "all")` tells *Weights & Biases* to log *gradients* and *parameters*' evolution in all the layers of a network. This functionality is useful to graphically visualize the concept of **Vanishing Gradients**.

For this exercise, I firstly tried to run a basic *MLP*, and then an *MLP with Residual Connections*. Honestly, at the time, I didn't think that this could be a very clever idea, since I've always seen Residuals been added only on Convolutional Networks, but... I decided to give it a try anyway.

As mentioned before, I compared these two architectures by challenging them on their performance over their **depth** (i.t. their number of layers).

A basic **10-layer MLP** already suffers from Vanishing Gradients, with its Accuracy dropping all the way down to 10%, that means picking a class literally **by chance**: the learning process is not working.

As mentioned in the original [ResNet paper](https://arxiv.org/abs/1512.03385), a higher number of layers leads to not only higher Validation Loss, but also a *higher Training Loss*. The fact that deeper networks lead to higher **Training** Loss (not only Validation Loss!) proves that this particular phenomenon is not simple *overfitting*, but it's related to the actual excessive model complexity.

On the contrary, the **10-layer Residual MLP** performed well, confirming the explanation of ResNet authors: Residual Connections allow a network to go **dramatically** deeper (with the only limitation of reaching *overfitting*).

Results can be quantitatively checked by observing the *Weights & Biases* logs for Gradient Magnitudes. The basic **MLP** shows gradients that are very close to *zero* for most of the training time, meaning that the model is not making any real learning progress.

Conversely, the **Residual MLP** showed gradients that did not vanish nor explode, and progressively diminishing their magnitude during training, meaning that the model is proceeding towards convergence on a (local, hopefully global) optimum.

---
### Exercise 2.2: Fully-convolutionalize a network

**Fully-convolutionalizing** a model means turning it into a network that can predict classification outputs at *all* pixels in an input image. This reframing unlocks many new visual tasks that can be addressed, such as *Semantic Segmentation*, *Object Detection* and *Object Recognition*.

The ConvNets built in the previous exercise have a global Average Pooling layer and a Fully Connected Layer (e.g. the *Classification Head*) at the end, in order to merge all info from the convolutions in a single output prediction for all the image, on the 10 MNIST/CIFAR-10 classes.

In a Fully Convolutional Network, we need instead to produce a prediction for every single one of the 28x28 (32x32) pixels of an image. I then proceed to do a "network surgery", removing the two layers mentioned above and rearranging the net to have the dimension of the input image as output.

In this section, I will turn one of the ConvNets I trained into a **detector** of handwritten MNIST digits. In order to test my new network, I need to write some functions to take random MNIST samples and embed them into a larger image (i.e. in a regular grid or at random positions).

---
### Exercise 2.3: *Explain* the predictions of a CNN

> B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, and A. Torralba. Learning Deep Features for Discriminative Localization. CVPR'16 (arXiv:1512.04150, 2015).
> 
[*Class Activation Maps*](http://cnnlocalization.csail.mit.edu/#:~:text=A%20class%20activation%20map%20for,decision%20made%20by%20the%20CNN.) are very powerful tools for understanding how Neural Networks learn in order to classify objects in an image.

The main focus, here, is to see how one of the previously trained CNNs *attends* to specific image features to recognize *specific* classes.

Class Activation Maps were here implemented following [this tutorial](https://medium.com/intelligentmachines/implementation-of-class-activation-map-cam-with-pytorch-c32f7e414923).

For this task, I trained a 50-layer ResidualCNN for 50 epochs, in order to reach a higher accuracy than all models that were trained in Exercise 1.

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my1horse.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my1horse_horse_probs0.9252.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my2automobile.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my2automobile_automobile_probs0.9999.jpg" width="24%" /> 
</p>

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my3automobile.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my3automobile_truck_probs0.6790.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my4automobile.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my4automobile_truck_probs0.9844.jpg" width="24%" /> 
</p>

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my5nd.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my5nd_deer_probs0.8761.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my6bird.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my6bird_horse_probs0.6518.jpg" width="24%" /> 
</p>

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my7cat.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my7cat_airplane_probs0.7187.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my8ship.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my8ship_airplane_probs0.6052.jpg" width="24%" /> 
</p>

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my9truck.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my9truck_truck_probs0.9999.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my10airplane.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my10airplane_airplane_probs0.3945.jpg" width="24%" /> 
</p>

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my11nd.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my11nd_deer_probs0.9759.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my12dog.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my12dog_frog_probs0.8742.jpg" width="24%" /> 
</p>

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my13dog.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my13dog_bird_probs0.3523.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my14ship.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my14ship_ship_probs0.8284.jpg" width="24%" /> 
</p>

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my15automobile.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my15automobile_frog_probs0.7401.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my16bird.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my16bird_bird_probs0.9999.jpg" width="24%" /> 
</p>

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my17cat.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my17cat_cat_probs0.5190.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my18ship.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my18ship_ship_probs0.9126.jpg" width="24%" /> 
</p>

---

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my10airplane.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my10airplane_airplane_probs0.3945.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my11nd.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my11nd_deer_probs0.9759.jpg" width="24%" /> 
</p>

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my10airplane.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my10airplane_airplane_probs0.3945.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my11nd.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my11nd_deer_probs0.9759.jpg" width="24%" /> 
</p>

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my10airplane.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my10airplane_airplane_probs0.3945.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my11nd.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my11nd_deer_probs0.9759.jpg" width="24%" /> 
</p>

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my10airplane.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my10airplane_airplane_probs0.3945.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my11nd.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my11nd_deer_probs0.9759.jpg" width="24%" /> 
</p>

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my10airplane.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my10airplane_airplane_probs0.3945.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my11nd.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my11nd_deer_probs0.9759.jpg" width="24%" /> 
</p>

<p float="left">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my10airplane.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my10airplane_airplane_probs0.3945.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_data/my11nd.jpg" width="24%" />
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab1/images/23_cam/my_cam/CAM_my11nd_deer_probs0.9759.jpg" width="24%" /> 
</p>
