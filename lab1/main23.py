# Deep Learning Applications 2023 course, held by Professor Andrew David Bagdanov - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/DLA_Labs/tree/main/lab1

# Code for EXERCISE 2.3

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from PIL import Image
import os

import my_utils
from models import ResidualCNN


MY_IMAGE_PATH = 'images/23_cam/my_data/my1horse.jpg'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = 'checkpoints/CIFAR10/ResidualCNN_depth50_ep50_bs128_lr0.001_wd0.0001_dr0.2_17243924/bestmodel.pth'
loaded_model = ResidualCNN((3, 32, 32), [64], 10, 50, 0.2).to(device)
model, _, _ = my_utils.load_checkpoint(MODEL_PATH, loaded_model, device)

use_cifar = False            # Set to False for custom images
image_idxs = [0,1,2,3,4,5,6,7,8,9]
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CUSTOM_IMAGE_SIZE = 32
CIFAR_CAM_SIZE = 256

# CIFAR-10 mean and std for normalization
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)


def returnCAM(feature_conv, weight_softmax, class_idx):
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        output_cam.append(cam_img)
    return output_cam

def show_cam(CAMs, orig_image, image_file, label, image_idx, probs):
    for i, cam in enumerate(CAMs):
        if use_cifar:
            cam_resized = cv2.resize(cam, (CIFAR_CAM_SIZE, CIFAR_CAM_SIZE))
            orig_image_resized = cv2.resize(orig_image, (CIFAR_CAM_SIZE, CIFAR_CAM_SIZE))
        else:
            cam_resized = cv2.resize(cam, (orig_image.shape[1], orig_image.shape[0]))
            orig_image_resized = orig_image

        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + orig_image_resized * 0.7        
        cv2.imshow('CAM', result/255.)
        if use_cifar:
            cv2.imwrite(f'images/23_cam/CAM_cifar_idx{image_idx}_{classes[label]}_probs{probs[0]:.4f}.jpg', result)
        else:
            filename = os.path.basename(image_file)
            cv2.imwrite(f'images/23_cam/CAM_{filename[:-4]}_{classes[label]}_probs{probs[0]:.4f}.jpg', result)


def process_image(image, image_idx = None, label = None):
    global model
    # Hook the feature extractor
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())
    model.features.register_forward_hook(hook_feature)

    # Get the softmax weight
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

    if use_cifar:
        image_file = f'images/23_cam/cifar_data/cifar_idx{image_idx}_{classes[label]}.jpg'
        resized_img = transforms.Resize((CIFAR_CAM_SIZE, CIFAR_CAM_SIZE))(image)
        resized_img.save(image_file)  

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std)
        ])
        tensor_image = transform(image)
    else:
        image_file = MY_IMAGE_PATH
        print(f"Using custom image: {image_file}")
    
        transform = transforms.Compose([
            transforms.Resize((CUSTOM_IMAGE_SIZE, CUSTOM_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std)
        ])
        tensor_image = transform(image)

    var_image = tensor_image.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = F.softmax(model(var_image), dim=1).data.squeeze()
    probs, idx = output.sort(0, True)
    probs = probs.cpu().numpy()
    idx = idx.cpu().numpy()
    predicted_label = idx[0]
    print(f"Predicted label: {classes[predicted_label]}")

    # Generate Class Activation Mapping for the Top-1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [predicted_label])
    
    # Convert PIL Image to numpy array for OpenCV
    orig_image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # For CIFAR images, resize to CIFAR_CAM_SIZE for visualization
    if use_cifar:
        orig_image_cv = cv2.resize(orig_image_cv, (CIFAR_CAM_SIZE, CIFAR_CAM_SIZE))
    
    show_cam(CAMs, orig_image_cv, image_file, predicted_label, image_idx, probs)

def main():
    if use_cifar:
        test_set = datasets.CIFAR10(root='./data', train=False, download=True)
        for idx in image_idxs:
            image, label = test_set[idx]
            print(f"\nProcessing CIFAR image index: {idx}")
            print(f"Real label: {classes[label]}")
            process_image(image, idx, label)
    else:
        image = Image.open(MY_IMAGE_PATH)
        process_image(image)

if __name__ == '__main__':
    main()
