from abc import ABC, abstractmethod
from torch import nn as nn 

class DPDDM_ABSTRACTCLASS(ABC, nn.Module):
    
    def __init__(self):
        super(DPDDM_ABSTRACTCLASS, self).__init__()
        
    @abstractmethod
    def get_features(self, x):
        pass

    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def get_last_layer(self):
        pass
    
    @abstractmethod
    def compute_max_dis_rate(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def get_pseudolabels(self, X, *args, **kwargs):
        pass
    
        