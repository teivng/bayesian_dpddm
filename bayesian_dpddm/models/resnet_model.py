import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 
import vbll
from .base import DPDDMAbstractModel
from ..configs import ModelConfig


class ResNetModel(DPDDMAbstractModel):
    """ DPDDM implementation with ResNet features. """
    def __init__(self, cfg:ModelConfig, train_size:int):
        super(ResNetModel, self).__init__()
        self.features = models.resnet50(models.ResNet50_Weights.DEFAULT)
        for param in self.features.parameters():
            param.requires_grad = False
        self.features.fc = nn.Linear(2048, 1000)
        self.features.fc.requires_grad = True
        
        self.out_layer = vbll.DiscClassification(cfg.hidden_dim, 
                                                 cfg.out_features, 
                                                 cfg.reg_weight_factor * 1/train_size, 
                                                 parameterization = cfg.param, 
                                                 return_ood=cfg.return_ood,
                                                 prior_scale=cfg.prior_scale, 
                                                 wishart_scale=cfg.wishart_scale
                                                 )
        self.cfg = cfg
    
    def get_features(self, x):
        x = self.features(x)
        return x
    
    def forward(self, x):
        x = self.get_features(x)
        return self.out_layer(x)
  
    def get_last_layer(self):
        return self.out_layer
    