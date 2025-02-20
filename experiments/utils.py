import torch
import numpy as np
from torchvision.transforms import v2 
import torchvision
import inspect
from bayesian_dpddm.configs import TrainConfig
from bayesian_dpddm.configs.model_configs import ConvModelConfig, MLPModelConfig
from omegaconf import DictConfig, OmegaConf
import os
from abc import ABC, abstractmethod

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
    """From parsed arguments, generate ModelConfig and TrainConfig configs for the experiment.

    Args:
        args (argparse.Namespace): hydra argument

    Returns:
        tuple(ModelConfig, TrainConfig): 2-tuple containing:
        the model and train configs respectively.
    """
    Configs = {
        'cifar10': ConvModelConfig,
        'uci': MLPModelConfig,
    }
    model_args = OmegaConf.to_container(args.model)
    train_args = OmegaConf.to_container(args.train)
    model_args['out_features'] = args.dataset.num_classes
    model_config = Configs[args.dataset.name](**model_args)
    train_config = TrainConfig(**train_args)
    
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
        uci_dict = get_uci_datasets(args)
        dataset_dict['train'] = uci_dict['train']
        dataset_dict['valid'] = uci_dict['val']
        dataset_dict['dpddm_train'] = uci_dict['val']
        dataset_dict['dpddm_id'] = uci_dict['iid_test']
        dataset_dict['dpddm_ood'] = uci_dict['ood_test']
        
    else: 
        raise NameError('Not a dataset with a known data pipeline implementation.')
    return dataset_dict


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