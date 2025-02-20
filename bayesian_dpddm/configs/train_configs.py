from dataclasses import dataclass
import torch
from pprint import pprint

@dataclass 
class TrainConfig:
    """Training configuration"""
    
    num_epochs: int
    batch_size: int
    lr: float
    wd: float
    optimizer: str # string should resolve to a torch.optim.Optimizer object
    clip_val: float = 1
    val_freq: int = 1
    
    def __str__(self):
        self.print_config()
        return ''
    
    def print_config(self):
        pprint(vars(self))