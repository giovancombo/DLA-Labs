import torch
import torchvision.utils as vutils
import cv2
import numpy as np

import utils


def cam_test(model, test_loader, epoch):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    classes = ['airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck']

    params = [param for param in model.parameters()]
    
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:

            images, labels = images.to(device), labels.to(device)
            oututs, b_gap, a_gap = model(images)
            
            _, predicated = torch.max(oututs.data, 1)
            total += labels.size(0)
            correct += (predicated == labels).sum().item()
            
            image_labels, image_paths = [], []

            for i in range(5):

                k = i
                vutils.save_image(images[k], f"Lab1/img/image{i}.jpg")

                # for cam using only the weights of the class predicted
                 
                #weights = params[-2][predicated[i].item()].detach()
                #c = torch.sum(b_gap[k]*weights[:,None, None], dim = 0)

                #using global average pooling parameters

                c = torch.sum(b_gap[k]*a_gap[k][:,None, None], dim = 0)
                
                c = (c-torch.min(c))/(torch.max(c)-torch.min(c))
            
                cam_img = np.uint8(255 * c.cpu().numpy())

                hm = cv2.applyColorMap(cv2.resize(cam_img, (96, 96)), cv2.COLORMAP_JET)
            
                re = hm*0.3+(images[k].permute(1,2,0).cpu().numpy()*255 )*0.4

                cv2.imwrite(f"Lab1/img/CAM{i}.jpg", re)

                image_labels.append(classes[labels[k]]+"-"+classes[predicated[k]])
                image_paths.append(f"Lab1/img/CAM{i}.jpg")


            utils.plot_images(image_paths, image_labels, epoch)

            break

    model.train()
    return