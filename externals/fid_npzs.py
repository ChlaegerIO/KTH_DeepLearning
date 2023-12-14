# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Script for calculating Frechet Inception Distance (FID)."""

import os
import click
import pickle
import numpy as np
import scipy.linalg
import torch
import dnnlib
import random
from glob import glob
#----------------------------------------------------------------------------

def calculate_inception_stats_npz(image_path, num_samples=50000, device=torch.device('cuda'),
):
    print('Loading Inception-v3 model...')
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048
    with dnnlib.util.open_url(detector_url, verbose=(0 == 0)) as f:
        detector_net = pickle.load(f).to(device)

    print(f'Loading images from "{image_path}"...')
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)

    files = glob(os.path.join(image_path, '*.npz'))
    # in case we have paperWithDG (64, 32, 32, 3) --> take only every 64th image with 64 images (not 64 times the same image)
    image_batch = np.load(files[0])["samples"]
    img_len = len(image_batch.shape)
    if img_len == 4:                # take only every 64 file in list
        files_new = []
        for i in range(0, len(files), 64):
            files_new.append(files[i])
        files = files_new
        print(f'Cut number of files to {len(files)} for 64 batched images')
    random.shuffle(files)
    count = 0

    for file in files:

        images = np.load(file)["samples"]
        if img_len == 3:                  # if we have no batch --> expand dims
            images = np.expand_dims(images, axis=0)

        images = torch.tensor(images).permute(0, 3, 1, 2).to(device)
        features = detector_net(images, **detector_kwargs).to(torch.float64)
        if count + images.shape[0] > num_samples:
            remaining_num_samples = num_samples - count
        else:
            remaining_num_samples = images.shape[0]
        mu += features[:remaining_num_samples].sum(0)
        sigma += features[:remaining_num_samples].T @ features[:remaining_num_samples]
        count = count + remaining_num_samples
        if count % 1000 == 0 or img_len != 3:
            print(count)
        if count >= num_samples:
            break

    print(count)
    mu /= num_samples
    sigma -= mu.ger(mu) * num_samples
    sigma /= num_samples - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()


#----------------------------------------------------------------------------
def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))

