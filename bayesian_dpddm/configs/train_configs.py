from dataclasses import dataclass
import torch
from pprint import pprint

@dataclass 
class TrainConfig:
    """Training configuration"""
    disagreement_epochs: int
    disagreement_optimizer: str
    disagreement_wd: float
    disagreement_lr: float
    disagreement_batch_size: int
    disagreement_alpha: float

    num_epochs: int
    batch_size: int
    lr: float
    wd: float
    optimizer: str # string should resolve to a torch.optim.Optimizer object
    num_workers: int
    pin_memory: bool
    clip_val: float = 1
    val_freq: int = 1
    
    
    def __str__(self):
        self.print_config()
        return ''
    
    def print_config(self):
        pprint(vars(self))