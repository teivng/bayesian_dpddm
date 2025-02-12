import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import vbll
from utils import temperature_scaling
from base import DPDDM_ABSTRACTMODEL

class CNN_DPDDM(DPDDM_ABSTRACTMODEL):
    """DPDDM implementation with CNN features."""
    
    def __init__(self, cfg):
        
        super(CNN_DPDDM, self).__init__()
        
        self.init_conv = nn.Conv2d(cfg.IN_CHANNELS, cfg.MID_CHANNELS, kernel_size=cfg.KERNEL_SIZE)
        
        self.mid_convs = nn.ModuleList(
            [nn.Conv2d(cfg.MID_CHANNELS, cfg.MID_CHANNELS, kernel_size=cfg.KERNEL_SIZE) for _ in range(cfg.MID_LAYERS)]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(cfg.MID_CHANNELS) for _ in range(cfg.MID_LAYERS)]
        )
        
        self.fc = nn.Linear(57, cfg.HIDDEN_DIM)
        self.out_layer = vbll.DiscClassification(cfg.HIDDEN_DIM, 
                                            cfg.OUT_FEATURES, 
                                            cfg.REG_WEIGHT, 
                                            parameterization = cfg.PARAM, 
                                            return_ood=cfg.RETURN_OOD,
                                            prior_scale=cfg.PRIOR_SCALE, 
                                            wishart_scale=cfg.WISHART_SCALE)
                                            
        self.pool = nn.MaxPool2d(cfg.POOL_DIMS, cfg.POOL_DIMS)
        self.dropout = nn.Dropout(p=cfg.DROPOUT)
        self.cfg = cfg
    
    def get_features(self, x):
        # initial convolution
        x = self.dropout(self.pool(F.elu(self.init_conv(x))))
        
        # mid convolutions with skip connections
        for idx in range(len(self.mid_conv)):
            identity = x
            out = self.mid_convs[idx](x)
            out = self.bns[idx](out)
            out += identity
            x = self.dropout(F.elu(out))
            

        x = self.pool(x).view(x.size()[0], -1)
        x = self.dropout(F.elu(self.fc(x)))
        return x
    
    def forward(self, x):
        x = self.get_features(x)
        return self.out_layer(x)
  
    def get_last_layer(self):
        return self.out_layer
  
    def compute_max_dis_rate(self, X, y, n_post_samples=5000, temperature=1):
        X, y = X.to(self.device), y.to(self.device)
        with torch.no_grad():
            features = self.get_features(X)
            ll_dist = self.out_layer.logit_predictive(features)
            logits_samples = ll_dist.rsample(sample_shape=torch.Size([n_post_samples]))
            
            # scale down with temperature
            logits_samples = temperature_scaling(logits_samples, temperature)
            y_hat = torch.argmax(logits_samples, -1)
            y_tile = torch.tile(y, (n_post_samples, 1)).cuda()
            dis_mat = (y_hat != y_tile)
            dis_rate = dis_mat.sum(dim=-1)/len(y)
        return torch.max(dis_rate).item()
      
    def get_pseudolabels(self, X, n_post_samples=5000):
        X = X.to(self.device)
        with torch.no_grad():
            features = self.get_features(X)
            #ll_dist = self.params['out_layer'].logit_predictive(features)
            # (n_post_samples, len, categories) (57, 1000, 2)
            ll_dist = self.out_layer.logit_predictive(features)
            '''
            logits_samples = ll_dist.rsample(sample_shape=torch.Size([n_post_samples]))
            logits_samples_mean = logits_samples.mean(0) # (1000, 2)
            y_hat = torch.argmax(logits_samples_mean,  -1) # (1000,)
            '''
            y_hat = torch.argmax(ll_dist.loc, 1)
        return y_hat