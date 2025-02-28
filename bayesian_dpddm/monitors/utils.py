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

def joint_sample_from_datasets(n_samples: int, datasetA:Dataset, datasetB: Dataset, balance_ratio:float = 1, replace=True):
    """
    Given two datasets, sample (n * balance_ratio) samples from A, and (n * (1-balance_ratio)) samples from B
    Args:
        n_samples (int): number of samples to sample
        datasetA (Dataset): torch Dataset object
        datasetB (Dataset): torch Dataset object
        balance_ratio (float): ratio of samples to sample from datasetA
        replace (bool, optional): Whether to sample with replacement. Defaults to True.
clear
s
    Returns:
        torch.tensor: sample from dataset.
        torch.tensor: mask of the samples, 1 if sample is from datasetA, 0 if sample is from datasetB
    """
    
    assert 0 <= balance_ratio <= 1, "balance_ratio must be between 0 and 1"
    n_samples_A = int(n_samples * balance_ratio)
    n_samples_B = n_samples - n_samples_A
    data_size = datasetA[0][0].size()
    indices_A = np.random.choice(np.arange(len(datasetA)), n_samples_A, replace=replace)
    indices_B = np.random.choice(np.arange(len(datasetB)), n_samples_B, replace=replace)
    tmp = torch.zeros(size=(n_samples, *data_size))
    true_labels = torch.zeros(size=(n_samples,))
    mask = torch.zeros(size=(n_samples,))
    for i in range(len(indices_A)):
        tmp[i] = datasetA[indices_A[i]][0]
        true_labels[i] = datasetA[indices_A[i]][1]
        mask[i] = 1
    for i in range(len(indices_B)):
        tmp[i + n_samples_A] = datasetB[indices_B[i]][0]
        true_labels[i + n_samples_A] = datasetB[indices_B[i]][1]
    return tmp, true_labels, mask


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