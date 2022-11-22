
# Learning from : https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

# Tutorial 1 — Quickstart

# lets import some libraries 
import torch 
from torch import nn 
from torch.utils.data import dataloader
from torchvision import datasets 
from torchvision.transforms import ToTensor 

# Tutorial 2 — Tensors 

import torch 
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#  Directly from data
data = [[11,22],[44,55]]
print("Original data : {}".format(data))
data_tensor = torch.tensor(data)
print("data in tensor : {}".format(data_tensor))

# From a NumPy array
np_array = np.array(data)
n_array_tensor = torch.from_numpy(np_array)
print("np to tensor : {}".format(n_array_tensor))

# From another tensor
data_ones = torch.ones_like(data_tensor)
print("Ones tensor : {}".format(data_ones))
data_rand = torch.rand_like(data_ones, dtype=torch.float)
print("Random tensor : {}".format(data_rand))

# Tutorial 3 -- Datasets and Dataloaders 
# 
# Learn in 5 steps:
# 1) Loading
# 2) Iterating & visualizing
# 3) Creating a custom dataset
# 4) Preparing your data for training with DataLoaders
# 5) Iterating through the DataLoader

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
# plt.show()
plt.savefig("fashion_dataset_01.png")

# Creating a Custom Dataset for your files
# A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.

import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# Preparing your data for training with DataLoaders

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Display image and label
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[1].squeeze()
label = train_labels[1]
plt.imshow(img, cmap="gray")
# plt.show()
print(f"Label: {label}")
plt.savefig("fashion_dataset_02.png")

