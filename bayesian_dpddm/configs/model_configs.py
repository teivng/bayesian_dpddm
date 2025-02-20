from dataclasses import dataclass


@dataclass
class ModelConfig:
    # standard stuff
    name: str
    out_features: int
    
    # vbll-specific configs
    reg_weight_factor: float
    param: str
    prior_scale: float
    wishart_scale: float
    
    
@dataclass
class ConvModelConfig(ModelConfig):
    """Configuration for the base CNN model"""
    in_channels: int
    mid_channels: int
    kernel_size: int
    mid_layers: int
    pool_dims: int
    hidden_dim: int
    dropout: float
    
    return_ood: bool = False