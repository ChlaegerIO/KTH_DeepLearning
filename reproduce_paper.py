import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

from adm_model_definition import create_adm_classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


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


############################ Discriminator: Timo ############################
# load pretrained classifier model (ADM) --> fix weights
    # ${project_page}/DG/
    # ├── checkpoints
    # │   ├── ADM_classifier/32x32_classifier.pt
    # ├── ...

# !!! für MNIST 28x28 Auflösung -->  !!!
def load_classifier():
    classifier_adm = create_adm_classifier(
        image_size=32,
        classifier_use_fp16=False,
        classifier_width=128,
        classifier_depth=4,
        classifier_attention_resolutions="32,16,8",
        classifier_use_scale_shift_norm=True,
        classifier_resblock_updown=True,
        classifier_pool="attention",
    )
    classifier_adm.load_state_dict(torch.load('./model/32x32_classifier_adm_pretrained.pt'), map_location=torch.device(device))     # does not work with 'cpu'
    classifier_adm.eval()
    return classifier_adm

# test classifier
classifier_model = load_classifier()
print(classifier_model)


# load pretrained discriminator model (U-Net?), or make a own one
    # ${project_page}/DG/
    # ├── checkpoints/discriminator
    # │   ├── cifar_uncond/discriminator_60.pt
    # │   ├── cifar_cond/discriminator_250.pt
    # ├── ...

def load_discriminator():
    pass


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
            
