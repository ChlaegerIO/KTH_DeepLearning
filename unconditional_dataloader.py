import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import os
from tqdm import tqdm

import params

class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = torch.tensor(data).clone().detach()
        self.targets = torch.tensor(targets).clone().detach()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, target


# TODO: do we need sample size to match shapes of generated and real data?
def get_dataloader(batch_size=params.batch_size):
    print("Get dataloader...")
    real_data = np.load('data/true_data.npz')['arr_0']

    # get generated data
    images = []

    # Loop through all files in the directory
    for filename in tqdm(os.listdir(params.outdir_gen_path)):
        if filename.endswith(".png"):
            # Construct full file path
            filepath = os.path.join(params.outdir_gen_path, filename)
            # Open the image file
            with Image.open(filepath) as img:
                # Convert to NumPy array and append to list
                images.append(np.asarray(img))

    # Stack all images into a single NumPy array [batch_size, 3, 32, 32]
    all_images = np.stack(images)
    generated_data = np.transpose(np.array(all_images[0:50000]), (0, 3, 1, 2))
    real_data = np.transpose(real_data, (0, 3, 1, 2))
    # print("generated shape", generated_data.shape)
    # print("real shape", real_data.shape)

    # real_data = real_data[0:7]
    assert real_data.shape == generated_data.shape, f"real {real_data.shape} and generated {generated_data.shape} data shapes do not match!"

    all_data = np.concatenate((real_data, generated_data))
    # all_data = real_data
    # print("all_data.shape", all_data.shape)
    all_label = torch.zeros(all_data.shape[0])
    all_label[:real_data.shape[0]] = 1.

    train_data, test_val_data, train_label, test_val_label = train_test_split(all_data, all_label, test_size=0.2, random_state=42)
    test_data, val_data, test_label, val_label = train_test_split(test_val_data, test_val_label, test_size=0.5, random_state=42)

    #transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = CustomDataset(train_data, train_label)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    val_dataset = CustomDataset(val_data, val_label)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    test_dataset = CustomDataset(test_data, test_label)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader


# Function to check a few batches from a DataLoader
def check_dataloader(dataloader, name):
    i = 0
    for data, labels in dataloader:
        if i >= 1:  # Checking the first batch only, increase this number to check more batches
            break
        print(f"{name} - Batch {i+1}")
        print(f"Data shape: {data.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Labels type: {labels.dtype}")
        print("-" * 30)
        i += 1

def visualize_batch(dataloader, title):
    data, labels = next(iter(dataloader))
    plt.figure(figsize=(10, 10))
    for i in range(4):  # Visualize the first 4 images of the batch
        plt.subplot(2, 2, i+1)
        plt.imshow(data[i].permute(0, 1, 2))  # Adjust permute for your data format
        plt.title(f"Label: {labels[i]}")
        plt.axis("off")
    plt.suptitle(title)
    plt.show()



def dataloader_final_tests():
    train_dataloader, val_dataloader, test_dataloader = get_dataloader()
    print(f"Number of batches in train DataLoader: {len(train_dataloader)}")
    print(f"Number of batches in test DataLoader: {len(test_dataloader)}")
    print(f"Number of batches in validation DataLoader: {len(val_dataloader)}")
    check_dataloader(train_dataloader, "Train DataLoader")
    check_dataloader(test_dataloader, "Test DataLoader")
    check_dataloader(val_dataloader, "Validation DataLoader")
    visualize_batch(train_dataloader, "Train DataLoader Samples")
    visualize_batch(test_dataloader, "Test DataLoader Samples")
    visualize_batch(val_dataloader, "Validation DataLoader Samples")


# dataloader_final_tests()



