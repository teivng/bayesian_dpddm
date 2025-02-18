import torch
import numpy as np
from torchvision.transforms import v2 
import torchvision
from dataclasses import dataclass
import argparse
import inspect
from bayesian_dpddm.configs import ConvModelConfig, TrainConfig

def filter_args(cls, args):
    """Given parsed arguments, filter arguments required by the dataclass

    Args:
        cls: class to get the constructor signature from
        args (argparse.Namespace): parsed arguments

    Returns:
        dict: dictionary with a kv pair for each constructor parameter and the matched argument parsed. 
    """
    
    sig = inspect.signature(cls.__init__)  # Get constructor signature
    return {k: v for k, v in vars(args).items() if k in sig.parameters}


def get_configs(args:argparse.Namespace):
    """From parsed arguments, generate ModelConfig and TrainConfig configs for the experiment.

    Args:
        args (argparse.Namespace): parsed argument

    Returns:
        tuple(ModelConfig, TrainConfig): the model and train configs respectively.
    """
    model_config = ConvModelConfig(**filter_args(ConvModelConfig, args))
    train_config = TrainConfig(**filter_args(TrainConfig, args))
    
    return model_config, train_config
    
    
class CIFAR101Dataset(torch.utils.data.Dataset):
    """Dataset 

    Attributes:
        X (torch.tensor): dataset features
        y (torch.tensor): dataset labels
    """
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        return len(self.y)


""" torchvision transforms """
train_transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomCrop(size=[32,32], padding=4),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
])
test_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
])


def get_cifar10_datasets():
    """Returns processed CIFAR10 and CIFAR10.1 Dataset objects. 

    Returns:
        tuple(Dataset, Dataset, CIFAR101Dataset): tuple containing CIFAR10 train, test, and CIFAR10.1. 
    """
    cifar10train = torchvision.datasets.CIFAR10(root='data/', 
                                                transform=train_transforms,
                                                download=True)
    cifar10test = torchvision.datasets.CIFAR10(root='data/', 
                                               train=False, 
                                               transform=test_transforms, 
                                               download=True)
    
        # Ensure CIFAR-10.1 data is in "data/" directory
    with open('data/cifar10.1_v6_data.npy', 'rb') as f:
        ood_data = np.load(f)
    with open('data/cifar10.1_v6_labels.npy', 'rb') as f:
        ood_labels = np.load(f)
    
    transformed101data = torch.zeros(size=(len(ood_data), 3, 32, 32))
    for idx in range(len(ood_data)):
        transformed101data[idx] = test_transforms(ood_data[idx])
    transformed101labels = torch.as_tensor(ood_labels, dtype=torch.long)
    cifar101 = CIFAR101Dataset(transformed101data, transformed101labels)
    
    return cifar10train, cifar10test, cifar101