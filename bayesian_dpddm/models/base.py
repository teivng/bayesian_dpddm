from abc import ABC, abstractmethod
import torch
from torch import nn as nn 

class DPDDM_ABSTRACTMODEL(ABC, nn.Module):
    """Defines the interface for a DPDDM model"""
    
    def __init__(self):
        super(DPDDM_ABSTRACTMODEL, self).__init__()
        
    @abstractmethod
    def get_features(self, x):
        """Returns the activations right before the last VBLL layer

        Args:
            x (torch.tensor): input tensor
        """
        pass

    @abstractmethod
    def forward(self, x):
        """Full forward pass. Should call self.get_features.

        Args:
            x (torch.tensor): input tensor
        """
        pass
    
    @abstractmethod
    def get_last_layer(self):
        """Returns the last layer object of the model"""
        pass
        