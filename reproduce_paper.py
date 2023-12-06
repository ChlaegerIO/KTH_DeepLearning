# library imports
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary
import numpy as np
import pickle
from tqdm import tqdm
import PIL.Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

# own files imports
from utils import load_classifier, load_discriminator#, get_discriminator
from unconditional_dataloader import get_dataloader
from diffusion import Diffusion, Discriminator
import params

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)


# load pretrained generator model (for conditional and unconditional case - do we want both?)
print("Load pretrained diffusion score model, classifier and discriminator...")
with open(params.diffusion_mPath, 'rb') as f:
    diffusion_model = pickle.load(f)['ema'].to(DEVICE)  # TODO: does not work yet?
# print("\nDiffusion model:", diffusion_model)

# load pretreined classifier
classifier_model = load_classifier(img_size=32, device=DEVICE)
# print("\nClassifier:",classifier_model)

# load pretrained or own discriminator
discriminator_model = load_discriminator(model_type="own", in_size=8, in_channels=512, device=DEVICE, eval=True)
# print("\nDiscriminator:", discriminator_model)

entire_dis_model = Discriminator(classifier_model, discriminator_model, enable_grad=True)

# loop over batches to generate samples
diffusion = Diffusion(diffusion_model, entire_dis_model, nbr_diff_steps=params.nbr_diff_steps, min_dis=params.min_dis, max_dis=params.max_dis, 
                    img_size=params.img_size, dg_weight_1order=params.dg_weight_1order, dg_weight_2order=params.dg_weight_2order, device=DEVICE)
nbr_batches = params.nbr_samples // params.batch_size + 1

if params.task_generate_samples:
    print("\nGenerate samples...")
    os.makedirs(params.outdir_gen, exist_ok=True)
    for i in tqdm(range(nbr_batches)):
        # sample from latent space (batch_size, 3, 32, 32)
        x_latent = torch.randn(params.batch_size, diffusion_model.img_channels, diffusion_model.img_resolution, diffusion_model.img_resolution, device=DEVICE)

        # generate samples images
        images = diffusion.sample(x_latent, params.boosting, params.time_min, params.time_max)

        # save generated samples images
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        count = 0
        for image_np in images_np:
            image_path = os.path.join(params.outdir_gen, f'{i*params.batch_size+count:06d}.png')
            PIL.Image.fromarray(image_np, 'RGB').save(image_path)

            # save generated samples as .npz
            image_path = os.path.join(params.outdir_gen, f'{i*params.batch_size+count:06d}.npz')
            np.savez_compressed(image_path, samples=image_np)

            count += 1

# prepare data loader (CIFAR-10, MINST later, simple toy 2-dimensional Case)
    # ${project_page}/DG/
    # ├── data
    # │   ├── true_data.npz
    # │   ├── true_data_label.npz
    # ├── ...



# test classifier/discriminator
# Batch = 128
# nbr_timesteps = torch.randn(Batch, device=DEVICE)   # optimal 1000
# input = torch.randn(Batch,3,32,32, device=DEVICE)
# summary(classifier_model, input_data=[input, nbr_timesteps])    # summary of model does not work, line 37: AttributeError: 'tuple' object has no attribute 'float'


############################ Next step ############################

# train the discriminator (conditional and unconditional) for discriminator guiding

if params.task_train_discriminator:
    print("\nTrain discriminator...")
    
    # load data
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(batch_size=params.batch_size)

    # train discriminator
    optimizer = optim.Adam(discriminator_model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    bce_loss = nn.BCELoss()
    scaler = lambda x: (x / 127.5) - 1

    loss_list =  []  
    accuracy_list = []
    loss_val_list = []
    accuracy_val_list = []       
    classifier_model.eval()
    for epoch in tqdm(range(params.nbr_epochs)):
        accuracy_epoch = []
        loss_epoch = []
        discriminator_model.train()
        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            # get data
            images, labels = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # scale data [0, 255] to [-1,1]
            images = scaler(images)

            # sample time, diffuse data
            t, _ = diffusion.get_diffusion_time(images.shape[0], images.device, t_min=params.min_diff_time, importance_sampling=params.importance_sampling)
            mean, std = diffusion.marginal_prob(t)
            z = torch.randn_like(images)
            perturbed_inputs = mean[:, None, None, None] * images + std[:, None, None, None] * z

            ## Forward
            with torch.no_grad():
                pretrained_feature = classifier_model(perturbed_inputs, timesteps=t, feature=True)
            label_prediction = discriminator_model(pretrained_feature, t, sigmoid=True).view(-1)

            ## Backward
            loss_net = bce_loss(label_prediction, labels)
            loss_net.backward()
            optimizer.step()

            # compute average loss, accuracy            
            accuracy = ((label_prediction > 0.5) == labels).float().mean().item()
            loss_epoch.append(loss_net.item())
            accuracy_epoch.append(accuracy)
        
        accuracy_average = np.mean(accuracy_epoch)
        accuracy_list.append(accuracy_average)

        loss_average = np.mean(loss_epoch)
        loss_list.append(loss_average)

        
        # validation loop
        discriminator_model.eval()
        loss_val = 0
        accuracy_val = 0
        for val in val_dataloader:
            # get data
            images, labels = val
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # scale data [0, 255] to [-1,1]
            images = scaler(images)

            t, _ = diffusion.get_diffusion_time(images.shape[0], images.device, importance_sampling=params.importance_sampling)
            mean, std = diffusion.marginal_prob(t)
            z = torch.randn_like(images)
            perturbed_inputs = mean[:, None, None, None] * images + std[:, None, None, None] * z

            ## Forward
            with torch.no_grad():
                pretrained_feature = classifier_model(perturbed_inputs, timesteps=t, feature=True)
                label_prediction = discriminator_model(pretrained_feature, t, sigmoid=True).view(-1)

            ## Backward
            loss_val += bce_loss(label_prediction, labels).item()

            # compute accuracy
            accuracy_val += ((label_prediction > 0.5) == labels).float().mean().item()

        # per epoch
        loss_val_list.append(loss_val / len(val_dataloader))
        accuracy_val_list.append(accuracy_val / len(val_dataloader))

        # save model, loss, accuracy
        if epoch % 5 == 0 or epoch == params.nbr_epochs-1:
            torch.save(discriminator_model.state_dict(), os.path.join(params.outdir_discriminator, f'discriminator_{epoch}.pth'))
            np.save(os.path.join(params.outdir_eval, f'loss.npz'), loss_list)
            np.save(os.path.join(params.outdir_eval, f'accuracy.npz'), accuracy_list)
            np.save(os.path.join(params.outdir_eval, f'loss_val.npz'), loss_val_list)
            np.save(os.path.join(params.outdir_eval, f'accuracy_val.npz'), accuracy_val_list)
            print(f"Epoch {epoch}: val loss={loss_val_list[-1]}, val accuracy={accuracy_val_list[-1]}, loss={loss_list[-1]}, accuracy={accuracy_list[-1]}")

# train ensemble
if params.task_train_ensemble:
    print("\nTrain ensemble...")
    ensemble_dict = {}
    ensemble_dict['ensemble_0'] = {'nbr_epochs': params.nbr_epochs_e0, 'lr': params.lr_e0, 'weight_decay': params.weight_decay_e0, 'min_diff_time': params.min_diff_time_e0}
    ensemble_dict['ensemble_1'] = {'nbr_epochs': params.nbr_epochs_e1, 'lr': params.lr_e1, 'weight_decay': params.weight_decay_e1, 'min_diff_time': params.min_diff_time_e1}
    ensemble_dict['ensemble_2'] = {'nbr_epochs': params.nbr_epochs_e2, 'lr': params.lr_e2, 'weight_decay': params.weight_decay_e2, 'min_diff_time': params.min_diff_time_e2}
    ensemble_dict['ensemble_3'] = {'nbr_epochs': params.nbr_epochs_e3, 'lr': params.lr_e3, 'weight_decay': params.weight_decay_e3, 'min_diff_time': params.min_diff_time_e3}
    ensemble_dict['ensemble_4'] = {'nbr_epochs': params.nbr_epochs_e4, 'lr': params.lr_e4, 'weight_decay': params.weight_decay_e4, 'min_diff_time': params.min_diff_time_e4}
    ensemble_dict['ensemble_5'] = {'nbr_epochs': params.nbr_epochs_e5, 'lr': params.lr_e5, 'weight_decay': params.weight_decay_e5, 'min_diff_time': params.min_diff_time_e5}

    # load data
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(batch_size=params.batch_size)

    # evaluation list initialization for all ensembles
    loss_dict = {}
    accuracy_dict = {}
    loss_val_dict = {}
    accuracy_val_dict = {}

    # loop over all ensembles
    for e in ensemble_dict.keys():
        print(f"\nTrain {e}...")
        # train discriminator
        optimizer = optim.Adam(discriminator_model.parameters(), lr=ensemble_dict[e]['lr'], weight_decay=ensemble_dict[e]['weight_decay'])
        bce_loss = nn.BCELoss()
        scaler = lambda x: (x / 127.5) - 1

        loss_list =  []  
        accuracy_list = []
        loss_val_list = []
        accuracy_val_list = []       
        classifier_model.eval()
        for epoch in tqdm(range(ensemble_dict[e]['nbr_epochs'])):
            discriminator_model.train()
            loss_train = 0
            accuracy_train = 0
            for i, data in enumerate(train_dataloader):
                optimizer.zero_grad()
                # get data
                images, labels = data
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                # scale data [0, 255] to [-1,1]
                images = scaler(images)

                # sample time, diffuse data
                t, _ = diffusion.get_diffusion_time(images.shape[0], images.device, t_min=ensemble_dict[e]['min_diff_time'], importance_sampling=params.importance_sampling)
                mean, std = diffusion.marginal_prob(t)
                z = torch.randn_like(images)
                perturbed_inputs = mean[:, None, None, None] * images + std[:, None, None, None] * z

                ## Forward
                with torch.no_grad():
                    pretrained_feature = classifier_model(perturbed_inputs, timesteps=t, feature=True)
                label_prediction = discriminator_model(pretrained_feature, t, sigmoid=True).view(-1)

                ## Backward
                loss_net = bce_loss(label_prediction, labels)
                loss_net.backward()
                optimizer.step()

                # compute average loss, accuracy
                loss_train += loss_net.item()
                accuracy_train += ((label_prediction > 0.5) == labels).float().mean().item()
            
            # per epoch
            loss_list.append(loss_train / len(train_dataloader))
            accuracy_list.append(accuracy_train / len(train_dataloader))

            
            # validation loop
            discriminator_model.eval()
            loss_val = 0
            accuracy_val = 0
            for val in val_dataloader:
                # get data
                images, labels = val
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                # scale data [0, 255] to [-1,1]
                images = scaler(images)

                t, _ = diffusion.get_diffusion_time(images.shape[0], images.device, importance_sampling=params.importance_sampling)
                mean, std = diffusion.marginal_prob(t)
                z = torch.randn_like(images)
                perturbed_inputs = mean[:, None, None, None] * images + std[:, None, None, None] * z

                ## Forward
                with torch.no_grad():
                    pretrained_feature = classifier_model(perturbed_inputs, timesteps=t, feature=True)
                    label_prediction = discriminator_model(pretrained_feature, t, sigmoid=True).view(-1)

                ## Backward
                loss_val += bce_loss(label_prediction, labels).item()

                # compute accuracy
                accuracy_val += ((label_prediction > 0.5) == labels).float().mean().item()

            # per epoch
            loss_val_list.append(loss_val / len(val_dataloader))
            accuracy_val_list.append(accuracy_val / len(val_dataloader))

            # save model, loss, accuracy
            if epoch % 5 == 0 or epoch == params.nbr_epochs-1:
                torch.save(discriminator_model.state_dict(), os.path.join(params.outdir_discriminator, f'discriminator_{e}_{epoch}.pth'))
                print(f"Epoch {epoch}: val loss={loss_val_list[-1]}, val accuracy={accuracy_val_list[-1]}, loss={loss_list[-1]}, accuracy={accuracy_list[-1]}")
            if epoch % 10 == 0 or epoch == params.nbr_epochs-1:
                np.save(os.path.join(params.outdir_eval, f'loss_{e}.npz'), loss_list)
                np.save(os.path.join(params.outdir_eval, f'accuracy_{e}.npz'), accuracy_list)
                np.save(os.path.join(params.outdir_eval, f'loss_val_{e}.npz'), loss_val_list)
                np.save(os.path.join(params.outdir_eval, f'accuracy_val_{e}.npz'), accuracy_val_list)

        # save loss, accuracy for each ensemble
        loss_dict[e] = loss_list
        accuracy_dict[e] = accuracy_list
        loss_val_dict[e] = loss_val_list
        accuracy_val_dict[e] = accuracy_val_list

    # save loss, accuracy for all ensembles
    np.save(os.path.join(params.outdir_eval, f'loss_dict.npz'), loss_dict)
    np.save(os.path.join(params.outdir_eval, f'accuracy_dict.npz'), accuracy_dict)
    np.save(os.path.join(params.outdir_eval, f'loss_val_dict.npz'), loss_val_dict)
    np.save(os.path.join(params.outdir_eval, f'accuracy_val_dict.npz'), accuracy_val_dict)
    

# evaluation (FID, IS, etc.)
    # ${project_page}/DG/
    # ├── stats
    # │   ├── cifar10-32x32.npz
    # ├── ...



# questions: 
    # - what do we want to show with the MINST dataset? --> compare with state of the art GAN, VAE, etc.?
    # - How to achieve NFE? - I'm not sure about that
    # - Why did the paper uses Euler 1st order with step size 0.001 and not RK4 with step size 0.1 --> faster?
            
