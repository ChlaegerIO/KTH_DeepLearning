import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary
import numpy as np
# import sys
import pickle
from tqdm import tqdm
import PIL.Image
# import matplotlib.pyplot as plt

from utils import load_classifier, load_discriminator#, get_discriminator
from diffusion import Diffusion, Discriminator
import params

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)


# load pretrained generator model (for conditional and unconditional case - do we want both?)
print("Load pretrained diffusion score model, classifier and discriminator...")
with open(params.diffusion_mPath, 'rb') as f:
    diffusion_model = pickle.load(f)['ema'].to(DEVICE)  # TODO: does not work yet?

print("\nDiffusion model:", diffusion_model)


# load pretreined classifier
classifier_model = load_classifier(img_size=32, device=DEVICE)
print("\nClassifier:",classifier_model)
# freeze classifier parameters
# for param in classifier_model.parameters():
#     param.requires_grad = False

# load pretrained or own discriminator
discriminator_model = load_discriminator(model_type="pretrained", in_size=8, in_channels=512, device=DEVICE, eval=True)
print("\nDiscriminator:", discriminator_model)

entire_dis_model = Discriminator(classifier_model, discriminator_model, enable_grad=True)

# loop over batches to generate samples
diffusion = Diffusion(diffusion_model, entire_dis_model, nbr_diff_steps=params.nbr_diff_steps, min_dis=params.min_dis, max_dis=params.max_dis, 
                    img_size=params.img_size, dg_weight_1order=params.dg_weight_1order, dg_weight_2order=params.dg_weight_2order, device=DEVICE)
nbr_batches = params.nbr_samples // params.batch_size + 1
os.makedirs(params.outdir_gen, exist_ok=True)
for i in tqdm(range(nbr_batches)):
    # sample from latent space (8, 3, 32, 32
    x_latent = torch.randn(params.batch_size, diffusion_model.img_channels, diffusion_model.img_resolution, diffusion_model.img_resolution, device=DEVICE)

    # generate samples images
    images = diffusion.sample(x_latent, params.boosting, params.time_min, params.time_max)

    # save generated samples images
    images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    count = 0
    for image_np in images_np:
        image_path = os.path.join(params.outdir_gen, f'{i*params.batch_size+count:06d}.png')
        count += 1
        PIL.Image.fromarray(image_np, 'RGB').save(image_path)



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
            
