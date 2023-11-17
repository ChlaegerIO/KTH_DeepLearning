import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def _init_(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def _len_(self):
        return len(self.data)

    def _getitem_(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, target


# TODO: do we need sample size to match shapes of generated and real data?
def get_train_dataloader(generated_data):
    real_data = np.load('data_unconditional/true_data.npz')['arr_0']
    # generated_data = np.random.rand(*real_data.shape)  # TODO: replace with actual data

    train_data = np.concatenate((real_data, generated_data))
    train_label = torch.zeros(train_data.shape[0])
    train_label[:real_data.shape[0]] = 1.

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = CustomDataset(train_data, train_label, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32)  # TODO figure out batch_size

    return train_dataloader


train_loader = get_train_dataloader()
for batch in train_loader:
    batch_data, batch_labels = batch
    # You can add any checks or operations here to verify the data
