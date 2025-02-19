import torch
import numpy as np
from torchvision.transforms import v2 
import torchvision
import argparse
import inspect
from bayesian_dpddm.configs import ConvModelConfig, TrainConfig
from omegaconf import DictConfig
import os

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

    def __str__(self):
        return """Dataset CIFAR10.1
    \tNumber of datapoints: 2000
    \tRoot location: data/
    \tSplit: OOD
    \tTestTransform
    \t{}
    """.format(test_transforms)
    

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


def get_configs(args:DictConfig):
    """From parsed arguments, generate ConvModelConfig and TrainConfig configs for the experiment.

    Args:
        args (argparse.Namespace): parsed argument

    Returns:
        tuple(ConvModelConfig, TrainConfig): 2-tuple containing:
        the model and train configs respectively.
    """
    model_config = ConvModelConfig(**filter_args(ConvModelConfig, args))
    train_config = TrainConfig(**filter_args(TrainConfig, args))
    
    return model_config, train_config


def get_datasets(args:DictConfig):
    """Generic class to get datasets. 
    
    Args:
        args (omegaconf.DictConfig): hydra config

    Returns:
        dict: dictionary with keys:
                - train         (used to train base model)
                - valid         (used ot validate base model)
                - dpddm_train   (used to train dpddm's Phi)
                - dpddm_id      (used to validate FPR)
                - dpddm_ood     (used to validate TPR)
    """
    dataset_dict = {}
    name = args.dataset.name
    if name == 'cifar10':
        train, test, test_no_aug, ood = get_cifar10_datasets(args, download=True)
        dataset_dict['train'] = train
        dataset_dict['valid'] = test
        dataset_dict['dpddm_train'] = test        # train dpddm using the CIFAR-10 validation set (10k)
        dataset_dict['dpddm_id'] = test_no_aug    # validate dpddm using the CIFAR-10 training set (50k)
        dataset_dict['dpddm_ood'] = ood
    
    elif name == 'uci':
        raise NotImplementedError
    return dataset_dict


def get_cifar10_datasets(args:DictConfig, download=True):
    """Returns processed CIFAR10 and CIFAR10.1 Dataset objects. 

    Returns:
        tuple(Dataset, Dataset, Dataset, CIFAR101Dataset): 4-tuple containing:
        CIFAR10 train, test, train with test transforms, and CIFAR10.1. 
    """
    cifar10train = torchvision.datasets.CIFAR10(root=args.dataset.data_dir, 
                                                transform=train_transforms,
                                                download=download)
    cifar10test = torchvision.datasets.CIFAR10(root=args.dataset.data_dir, 
                                               train=False, 
                                               transform=test_transforms, 
                                               download=download)
    
    cifar10train_with_test_transforms = torchvision.datasets.CIFAR10(root=args.dataset.data_dir, 
                                                transform=test_transforms,
                                                download=download)
    
    # Ensure CIFAR-10.1 data is in "data/" directory
    with open(os.path.join(args.dataset.data_dir, 'cifar10.1_v6_data.npy'), 'rb') as f:
        ood_data = np.load(f)
    with open(os.path.join(args.dataset.data_dir, 'cifar10.1_v6_labels.npy'), 'rb') as f:
        ood_labels = np.load(f)
    
    transformed101data = torch.zeros(size=(len(ood_data), 3, 32, 32))
    for idx in range(len(ood_data)):
        transformed101data[idx] = test_transforms(ood_data[idx])
    transformed101labels = torch.as_tensor(ood_labels, dtype=torch.long)
    cifar101 = CIFAR101Dataset(transformed101data, transformed101labels)
    return cifar10train, cifar10test, cifar10train_with_test_transforms, cifar101

def print_args_and_kwargs(*args, **kwargs):
    """Prints all args and kwargs"""
    # Print all positional arguments (*args)
    print("Positional arguments (*args):")
    for i, arg in enumerate(args, start=1):
        print(f"  Argument {i}: {arg}")

    # Print all keyword arguments (**kwargs)
    print("\nKeyword arguments (**kwargs):")
    for key, value in kwargs.items():
        print(f"  {key}: {value}")