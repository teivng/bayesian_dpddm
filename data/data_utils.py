import os 
import numpy as np
from abc import ABC, abstractmethod
import torch
import torchvision
from torchvision.transforms import v2
from omegaconf import DictConfig
from torch.utils.data import Dataset, Subset


# ================================================
# ===========General Data Utilities===============
# ================================================

class TensorDataset(Dataset, ABC):
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
    """Returns processed CIFAR10 and CIFAR10.1 Dataset objects
    We split the train dataset into a trainset and a validation set
    The validation set has two purposes: 
        - Validate the training of the base model
        - Train the distribution of in-distribution maximum disagreement rates (Phi)
    The CIFAR-10 test set will be used to validate the training of Phi, i.e. an iid_test sample
    The CIFAR-10.1 o.o.d. set will be used to study the TPR of bayesian D-PDDM.

    Returns:
        tuple(Dataset, Dataset, Dataset, CIFAR101Dataset): 4-tuple containing:
        CIFAR10 train, test, train with test transforms, and CIFAR10.1. 
    """
    os.makedirs(args.dataset.data_dir, exist_ok=True)
    # Loads the cifar-10 test set
    cifar10test = torchvision.datasets.CIFAR10(root=args.dataset.data_dir, 
                                               train=False, 
                                               transform=test_transforms, 
                                               download=download)
    
    # make the cifar-10 train and validation sets
    cifar10train = torchvision.datasets.CIFAR10(root=args.dataset.data_dir,
                                                train=True, 
                                                transform=None,
                                                download=download)
    
    cifar10train, cifar10val = torch.utils.data.random_split(cifar10train, [40000, 10000])
    cifar10train = torch.utils.data.Subset(
        dataset=torchvision.datasets.CIFAR10(
            root=args.dataset.data_dir,
            train=True,
            transform=train_transforms,
            download=True
        ),
        indices=cifar10train.indices
    )
    cifar10val = torch.utils.data.Subset(
        dataset=torchvision.datasets.CIFAR10(
            root=args.dataset.data_dir,
            train=True,
            transform=test_transforms,
            download=True
        ),
        indices=cifar10val.indices
    )
    
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
    return cifar10train, cifar10val, cifar10test, cifar101


# ================================================
# =========UCI Heart Disease Utilities============
# ================================================


class UCIDataset(TensorDataset):
    """UCI Heart Disease Dataset class.

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
    data_path = os.path.join(args.dataset.data_dir, 'uci_heart_torch.pt')
    assert os.path.exists(data_path)        
    data_dict = torch.load(data_path)
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

# ================================================
# ===========Synthetic Data Utilities=============
# ================================================

class SyntheticDataset(TensorDataset):
    """Synthetic Dataset class

    Attributes:
        X (torch.tensor): dataset features
        y (torch.tensor): dataset labels
    """
    def __init__(self, X:torch.tensor, y:torch.tensor):
        super(SyntheticDataset, self).__init__(X=X, y=y)    

    def __str__(self):
        return f"""Dataset Synthetic
    \tNumber of datapoints: {self.__len__()}
    \tRoot location: data/synthetic_data/
    """
    
def get_synthetic_datasets(args:DictConfig) -> dict:
    """Returns synthetic data according to generation parameters

    Args:
        args (DictConfig): hydra arguments

    Returns:
        dict: dictionary containing all splits of generated synthetic data.
    """
    def _generate_data(n, f, mean, var, var2=1, d=2, gap=0):
        # Generate independent component of data
        # Assume isotropic Gaussian
        means = mean * np.ones(shape=(n, d-1))
        vrs = var * np.ones(shape=(n, d-1))
        x1 = np.random.normal(means, vrs, size=(n, d-1))
        eps = np.random.normal(0, var2, size=n)
        labels = np.sign(eps)
            
        #convert to 0,1
        labels = [int(p) if p==1 else 0 for p in labels]
        
        # Generate dependent component of data
        x2 = np.array(list(map(f, x1))) + eps + np.sign(eps)*gap
        x2 = np.expand_dims(x2, axis=1)
        # Merge x1 and x2
        features = np.concatenate([x1, x2], axis=1)
        return features, labels

    n = args.dataset.n
    m = args.dataset.m
    #f = lambda x : sum([np.sin(c) for c in x])
    f = lambda x : np.sin(sum([c for c in x]))
    id_mean = args.dataset.id_mean
    var = args.dataset.var
    d = args.model.in_features
    ood_mean = args.dataset.ood_mean
    gap = args.dataset.gap

    data_dict = {}
    
    data_dict['train'] = SyntheticDataset(*list(map(lambda x, dtype: torch.as_tensor(x, dtype=dtype), _generate_data(n, f, id_mean, var, d=d, gap=gap), [torch.float, torch.long])))
    data_dict['valid'] = SyntheticDataset(*list(map(lambda x, dtype: torch.as_tensor(x, dtype=dtype), _generate_data(m, f, id_mean, var, d=d, gap=gap), [torch.float, torch.long])))
    data_dict['dpddm_train'] = data_dict['valid']
    data_dict['dpddm_id'] = SyntheticDataset(*list(map(lambda x, dtype: torch.as_tensor(x, dtype=dtype), _generate_data(m, f, id_mean, var, d=d, gap=gap), [torch.float, torch.long])))
    data_dict['dpddm_ood'] = SyntheticDataset(*list(map(lambda x, dtype: torch.as_tensor(x, dtype=dtype), _generate_data(m, f, ood_mean, var, d=d, gap=gap), [torch.float, torch.long])))

    return data_dict

    