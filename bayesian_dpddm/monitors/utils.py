from torch.utils.data import Dataset
import numpy as np
import torch


def temperature_scaling(logits, temperature):
    return logits / temperature


def sample_from_dataset(n_samples:int, dataset:Dataset, replace=True):
    """Given a dataset, sample n_samples.

    Args:
        n_samples (int): number of samples to sample
        dataset (Dataset): torch Dataset object
        replace (bool, optional): Whether to sample with replacement. Defaults to True.
clear
s
    Returns:
        torch.tensor: sample from dataset.
    """
    data_size = dataset[0][0].size()
    indices = np.random.choice(np.arange(len(dataset)), n_samples, replace=replace)
    tmp = torch.zeros(size=(n_samples, *data_size))
    true_labels = torch.zeros(size=(n_samples,))
    for i in range(len(indices)):
        tmp[i] = dataset[indices[i]][0]
        true_labels[i] = dataset[indices[i]][1]
    return tmp, true_labels


def get_class_from_string(class_string):
    """Given a class name, return the module to be instantiated.
    To be used to retrieve objects like "torch.optim.AdamW". 

    Args:
        class_string (str): string name of the package to be retrieved

    Returns:
        <class 'module'>: the class to be retrieved.
    """
    # Split the string into module and class name
    module_name, class_name = class_string.rsplit('.', 1)
    
    # Import the module dynamically
    module = __import__(module_name, fromlist=[class_name])
    
    # Get the class from the module
    cls = getattr(module, class_name)
    
    return cls