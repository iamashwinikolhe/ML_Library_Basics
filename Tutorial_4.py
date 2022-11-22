# Tutorial 04 - TRANSFORMS
# Data does not always come in its final processed form that is required for training machine learning algorithms.
# We use transforms to perform some manipulation of the data and make it suitable for training.
# All TorchVision datasets have two parameters -transform to modify the features and target_transform to modify the labels - that accept
# callables containing the transformation logic.
# The FashionMNIST features are in PIL Image format, and the labels are integers.
# For training, we need the features as normalized tensors, and the labels as one-hot encoded tensors.
# To make these transformations, we use ToTensor and Lambda.
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

# try it out 
y = [1,2,1,4,1,6,7,8,9]
torch.zeros(10).scatter_(0, torch.tensor(y), value=5)
target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

