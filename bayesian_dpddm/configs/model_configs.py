from dataclasses import dataclass

    
@dataclass
class ConvModelConfig:
    """Configuration for the base CNN model"""
    
    in_channels: int
    mid_channels: int
    out_features: int
    kernel_size: int
    mid_layers: int
    pool_dims: int
    hidden_dim: int
    dropout: float
    reg_weight: float
    param: str
    prior_scale: float
    wishart_scale: float
    
    return_ood: bool = False
