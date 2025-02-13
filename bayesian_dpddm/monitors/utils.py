from torch.utils.data import Dataset
import numpy as np
import torch


def temperature_scaling(logits, temperature):
    return logits / temperature

def sample_from_dataset(n_samples:int, dataset:Dataset, device:torch.device, replace=True):
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
    for i in range(len(indices)):
        tmp[i] = dataset[indices[i]][0]
    tmp = tmp.to(device)
    return tmp
