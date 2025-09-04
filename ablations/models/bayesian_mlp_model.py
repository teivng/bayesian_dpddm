import torch.nn as nn
import torch.nn.functional as F
from bayesian_torch.layers import LinearReparameterization
from bayesian_dpddm.models.base import DPDDMAbstractModel

class BayesianMLP(DPDDMAbstractModel):
    def __init__(self, cfg):
        super().__init__()
        self.init_fc = LinearReparameterization(cfg.in_features, cfg.mid_features)

        self.mid_fc = nn.ModuleList([
            LinearReparameterization(cfg.mid_features, cfg.mid_features)
            for _ in range(cfg.mid_layers)
        ])

        self.out_layer = LinearReparameterization(cfg.mid_features, cfg.out_features)
    
    def get_features(self, x):
        kl_sum = 0
        x, kl = self.init_fc(x)
        kl_sum += kl
        x = F.elu(x)
        for layer in self.mid_fc:
            identity = x
            out, kl = layer(x)
            kl_sum += kl
            x = F.elu(out + identity)
        return x, kl_sum

    def forward(self, x):
        kl_sum = 0
        x, kl = self.get_features(x)
        kl_sum += kl
        
        return self.out_layer(x)
    
    def get_last_layer(self):
        return self.out_layer
