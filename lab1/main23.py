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
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True


def returnCAM(feature_conv, weight_softmax, class_idx, width, height):
    # Generate the Class Activation Maps upsample to (width x height)
    size_upsample = (width, height)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        # Dot product weights and feature map
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w) 
        cam = cam - np.min(cam)              
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def show_cam(CAMs, width, height, orig_image, image_file):
    for i, cam in enumerate(CAMs):
        heatmap = cv2.applyColorMap(cv2.resize(cam, (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.25 + orig_image * 0.75
        cv2.imshow('CAM', result/255.)
        if cifar:
            cv2.imwrite('images/CAM/CAM_cifar_' + str(classes[label]) + '_idx' + str(image_idx) + '_probs' + str(probs[0]) + '.jpg', result)
        else:
            cv2.imwrite('images/CAM/CAM_' + {image_file.split('/')[-1]} + '_pred_' + str(probs[0]) + '.jpg', result)


cifar = True
image_idx = 7
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    if cifar:
        test_set = datasets.CIFAR10(root = './data', train = False, download = True)
        image, label = test_set[image_idx]
        image_file = 'images/cifar_' + str(classes[label]) + '.jpg'
        image.save(image_file)
        print("real label:", classes[label])
    else:
        # Load the image
        image_file = 'images/cam/my_horse.jpg'
        print("using hd image:", image_file)


    # Load the model
    model = torch.load("models/CIFAR10/residualcnn-depth5-ep10-lr0.0001-bs256-dr0.2-1708516763.2621446model.pt")

    # Hook the feature extractor
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())
    model._modules.get('features').register_forward_hook(hook_feature)
    # Get the softmax weight
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())


    # Predict the image class
    image = Image.open(image_file)
    pil_image = transforms.Resize((256))(image)
    tensor_image = transforms.ToTensor()(pil_image)
    var_image = Variable(tensor_image.unsqueeze(0))
    resized_width, resized_height = pil_image.size

    output = F.softmax(model(var_image.to(device)), dim = 1).data.squeeze()
    probs, idx = output.sort(0, True)
    probs = probs.cpu().numpy()
    idx = idx.cpu().numpy()
    print("predicted label:", classes[idx[0]])

    # Generate Class Activation Mapping for the Top-1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]], resized_width, resized_height)
    show_cam(CAMs, resized_width, resized_height, cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR), image_file)
