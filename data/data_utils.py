import os 
import numpy as np
from abc import ABC, abstractmethod
import torch
import torchvision
from torchvision.transforms import v2
from omegaconf import DictConfig


# ================================================
# ===========General Data Utilities===============
# ================================================

class TensorDataset(torch.utils.data.Dataset, ABC):
    """Abstract torch tensor dataset class.
    Requires an implementation of TensorDataset.__str__

    Attributes:
        X (torch.tensor): dataset features
        y (torch.tensor): dataset labels
    """
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def __len__(self) -> int:
        return len(self.y)
    
    @abstractmethod
    def __str__(self) -> str:
        pass
    
# ================================================
# =============CIFAR-10 Utilities=================
# ================================================


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

    
class CIFAR101Dataset(TensorDataset):
    """CIFAR10.1 Dataset class

    Attributes:
        X (torch.tensor): dataset features
        y (torch.tensor): dataset labels
    """
    
    def __init__(self, X, y):
        super(CIFAR101Dataset, self).__init__(X=X, y=y)
        
    def __str__(self):
        return """Dataset CIFAR10.1
    \tNumber of datapoints: {}
    \tRoot location: data/cifar10_data/
    \tSplit: OOD
    \tTestTransform
    \t{}
    """.format(self.__len__(), test_transforms)


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


# ================================================
# =========UCI Heart Disease Utilities============
# ================================================


class UCIDataset(CIFAR101Dataset):
    """UCI Heart Disease Dataset class.
    Hack-y inheritance.

    Attributes:
        X (torch.tensor): dataset features
        y (torch.tensor): dataset labels
    """
    def __init__(self, X:torch.tensor, y:torch.tensor):
        super(UCIDataset, self).__init__(X=X, y=y)
        
    def __str__(self):
        return f"""Dataset UCI Heart Disease
    \tNumber of datapoints: {self.__len__()}
    \tRoot location: data/uci_data/
    """
        

def get_uci_datasets(args:DictConfig) -> dict:
    """Returns processed UCI Heart Disease Dataset objects

    Args:
        args (DictConfig): hydra arguments

    Returns:
        dict: dictionary containing all splits of the UCI Heart Disease processed dataset.
    """
    data_dict = torch.load(os.path.join(args.dataset.data_dir, 'uci_heart_torch.pt'))
    processed_dict = {}
    for k, data in data_dict.items():
        data = list(zip(*data))
        X, y = torch.stack(data[0]), torch.tensor(data[1], dtype=torch.int)
        if args.dataset.normalize:
            min_ = torch.min(X, dim=0).values
            max_ = torch.max(X, dim=0).values
            X = (X - min_) / (max_ - min_)
        processed_dict[k] = UCIDataset(X, y)
        
    return processed_dict