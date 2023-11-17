import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary
# import numpy as np
# import sys
import pickle
# import matplotlib.pyplot as plt

from utils import load_classifier, load_discriminator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)


# We take the column unconditional EDM-G++ model (page 21)

############################ Dataloader: Lisa ############################
# load pretrained generator model (for conditional and unconditional case - do we want both?)
    # ${project_page}/DG/
    # ├── checkpoints
    # │   ├── pretrained_score/edm-cifar10-32x32-uncond-vp.pkl
    # │   ├── pretrained_score/edm-cifar10-32x32-cond-vp.pkl
    # ├── ...
print("Load pretrained diffusion score model...")
with open('./model/edm-cifar10-32x32-uncond-vp.pkl', 'rb') as f:
    diffusion_model = pickle.load(f)['ema'].to(DEVICE)  # TODO: does not work yet?

print("\nDiffusion model:", diffusion_model)

# generate samples (conditional and unconditional) to check the model is working

# prepare real data (CIFAR-10, MINST later, simple toy 2-dimensional Case)
    # ${project_page}/DG/
    # ├── data
    # │   ├── true_data.npz
    # │   ├── true_data_label.npz
    # ├── ...


############################ Discriminator: Timo ############################
# load pretreined classifier
classifier_model = load_classifier(img_size=32, device=DEVICE)
print("\nClassifier:",classifier_model)


# load pretrained or own discriminator
discriminator_model = load_discriminator(model_type="pretrained", in_size=8, in_channels=512, device=DEVICE, eval=True)
print("\nDiscriminator:", discriminator_model)

# test classifier/discriminator
# Batch = 128
# nbr_timesteps = torch.randn(Batch, device=DEVICE)   # optimal 1000
# input = torch.randn(Batch,3,32,32, device=DEVICE)
# summary(classifier_model, input_data=[input, nbr_timesteps])    # summary of model does not work, line 37: AttributeError: 'tuple' object has no attribute 'float'


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
            
