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
from diffusion import Diffusion, Discriminator

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
print("Load pretrained diffusion score model, classifier and discriminator...")
with open('./model/edm-cifar10-32x32-uncond-vp.pkl', 'rb') as f:
    diffusion_model = pickle.load(f)['ema'].to(DEVICE)  # TODO: does not work yet?

print("\nDiffusion model:", diffusion_model)

# load pretreined classifier
classifier_model = load_classifier(img_size=32, device=DEVICE)
print("\nClassifier:",classifier_model)
# freeze classifier parameters
for param in classifier_model.parameters():
    param.requires_grad = False


# load pretrained or own discriminator
discriminator_model = load_discriminator(model_type="pretrained", in_size=8, in_channels=512, device=DEVICE, eval=True)
print("\nDiscriminator:", discriminator_model)

    
entire_dis_model = Discriminator(classifier_model, discriminator_model)

# generate samples (conditional and unconditional) to check the model is working
x_latent = torch.randn(1, 3, 32, 32, device=DEVICE)
time_min = 0.01 # [0,1]
time_max = 1.0  # [0,1]
boosting = True
diffusion = Diffusion(diffusion_model, entire_dis_model, nbr_diff_steps=35, min_dis=10e-5, max_dis=1-10e-5, img_size=32, dg_weight_1order=2.0, dg_weight_2order=0, device=DEVICE)
diffusion.sample(x_latent, boosting, time_min, time_max)


# prepare data loader (CIFAR-10, MINST later, simple toy 2-dimensional Case)
    # ${project_page}/DG/
    # ├── data
    # │   ├── true_data.npz
    # │   ├── true_data_label.npz
    # ├── ...


############################ Discriminator: Timo ############################



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



# questions: 
    # - what do we want to show with the MINST dataset? --> compare with state of the art GAN, VAE, etc.?
    # - How to achieve NFE? - I'm not sure about that
    # - Why did the paper uses Euler 1st order with step size 0.001 and not RK4 with step size 0.1 --> faster?
            
