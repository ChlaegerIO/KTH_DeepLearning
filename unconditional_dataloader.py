import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = torch.tensor(data)
        self.targets = torch.tensor(targets)
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
def get_dataloader(generated_data=None):
    real_data = np.load('data_unconditional/true_data.npz')['arr_0']
    generated_data = np.random.rand(*real_data.shape)  # TODO: replace with actual data

    all_data = np.concatenate((real_data, generated_data))
    all_data = real_data
    all_label = torch.zeros(all_data.shape[0])
    all_label[:real_data.shape[0]] = 1.

    train_data, test_val_data, train_label, test_val_label = train_test_split(all_data, all_label,test_size=0.2, random_state=42)

    test_data, val_data, test_label, val_label = train_test_split(test_val_data, test_val_label, test_size=0.5, random_state=42)


    #transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = CustomDataset(train_data, train_label)
    train_dataloader = DataLoader(train_dataset, batch_size=32)  # TODO figure out batch_size

    val_dataset = CustomDataset(val_data, val_label)
    val_dataloader = DataLoader(val_dataset, batch_size=32)  # TODO figure out batch_size

    test_dataset = CustomDataset(test_data, test_label)
    test_dataloader = DataLoader(test_dataset, batch_size=32)  # TODO figure out batch_size

    return train_dataloader, val_dataloader, test_dataloader





# Function to check a few batches from a DataLoader
def check_dataloader(dataloader, name):
    for i, (data, labels) in enumerate(dataloader):
        if i >= 1:  # Checking the first batch only, increase this number to check more batches
            break
        print(f"{name} - Batch {i+1}")
        print(f"Data shape: {data.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Labels type: {labels.dtype}")
        print("-" * 30)

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


dataloader_final_tests()



