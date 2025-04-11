import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34
import vbll
from .base import DPDDMAbstractModel
from ..configs import ModelConfig


class ResNetModel(DPDDMAbstractModel):
    """ DPDDM implementation with ResNet features. """
    def __init__(self, cfg:ModelConfig, train_size:int):
        assert train_size is not None
        super(ResNetModel, self).__init__()
        self.init_conv = nn.Conv2d(cfg.in_channels, cfg.mid_channels, 
                                  kernel_size=7, stride=2, padding=3)
        self.init_pool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Mid convolutions (no downsampling)
        self.mid_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cfg.mid_channels, cfg.mid_channels, 
                         kernel_size=cfg.kernel_size, padding=(cfg.kernel_size-1)//2),
                nn.BatchNorm2d(cfg.mid_channels),
                nn.ReLU()
            ) for _ in range(cfg.mid_layers)
        ])
        
        # Final spatial size: 24x24
        self.flatten_dim = cfg.mid_channels * 24 * 24
        self.fc = nn.Linear(self.flatten_dim, cfg.hidden_dim)
        self.out_layer = vbll.DiscClassification(cfg.hidden_dim, 
                                                 cfg.out_features, 
                                                 cfg.reg_weight_factor * 1/train_size, 
                                                 parameterization = cfg.param, 
                                                 return_ood=cfg.return_ood,
                                                 prior_scale=cfg.prior_scale, 
                                                 wishart_scale=cfg.wishart_scale
                                                 )
                                            
        self.pool = nn.MaxPool2d(cfg.pool_dims, cfg.pool_dims)
        self.dropout = nn.Dropout(p=cfg.dropout)
        self.cfg = cfg
    
    def get_features(self, x):
        # initial convolution
        x = F.elu(self.init_conv(x))
        x = self.dropout(self.init_pool(x))
        #x = self.dropout(self.pool(x))
        # mid convolutions with skip connections
        for conv in self.mid_convs:
            x = conv(x)        # 24x24 (unchanged)
        x = x.flatten(1)       # [B, C, 24, 24] -> [B, C*24*24]
        x = self.fc(x)
    
        x = self.dropout(F.elu(x))
        return x
    
    def forward(self, x):
        x = self.get_features(x)
        return self.out_layer(x)
  
    def get_last_layer(self):
        return self.out_layer
    