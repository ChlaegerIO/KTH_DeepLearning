import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt


# load CIFAR-10 dataset
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# # plot some images
# def imshow(img):
#     img = img / 2 + 0.5 # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1,2,0)))
#     plt.show()

# imshow(trainloader.dataset.data[0])


# We take the column unconditional EDM-G++ model (page 21)

############################ Dataloader: Lisa ############################
# load pretrained generator model (for conditional and unconditional case - do we want both?)
    # ${project_page}/DG/
    # ├── checkpoints
    # │   ├── pretrained_score/edm-cifar10-32x32-uncond-vp.pkl
    # │   ├── pretrained_score/edm-cifar10-32x32-cond-vp.pkl
    # ├── ...

# generate samples (conditional and unconditional) to check the model is working

# prepare real data (CIFAR-10, MINST later, simple toy 2-dimensional Case)
    # ${project_page}/DG/
    # ├── data
    # │   ├── true_data.npz
    # │   ├── true_data_label.npz
    # ├── ...


############################ Dataloader: Timo ############################
# load pretrained classifier model (ADM) --> fix weights
    # ${project_page}/DG/
    # ├── checkpoints
    # │   ├── ADM_classifier/32x32_classifier.pt
    # ├── ...


# load pretrained discriminator model (U-Net?), or make a own one
    # ${project_page}/DG/
    # ├── checkpoints/discriminator
    # │   ├── cifar_uncond/discriminator_60.pt
    # │   ├── cifar_cond/discriminator_250.pt
    # ├── ...



############################ Next step ############################

# train the discriminator (conditional and unconditional) for discriminator guiding



# generator-quided sample generator to check the model is working


# evaluation (FID, IS, etc.)
    # ${project_page}/DG/
    # ├── stats
    # │   ├── cifar10-32x32.npz
    # ├── ...

############################ Dataloader: Vik ############################


# SDE (Stochastic differential equation) layer
    # correction term c
    # g(t) --> volatility function
    # f(x,t) --> drift function
    # forward Euler method


# helper functions
    # - Euler, RK45 solver
    # 





# questions: 
    # - what do we want to show with the MINST dataset? --> compare with state of the art GAN, VAE, etc.?
    # - How to achieve NFE? - I'm not sure about that
    # - Why did the paper uses Euler 1st order with step size 0.001 and not RK4 with step size 0.1 --> faster?
            
