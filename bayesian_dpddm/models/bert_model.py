import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 
import vbll
from .base import DPDDMAbstractModel
from ..configs import ModelConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel

bert_dict = {
    'distilbert': "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
}


class BERTModel(DPDDMAbstractModel):
    """ DPDDM implementation with BERT features. """
    def __init__(self, cfg:ModelConfig, train_size:int):
        super(BERTModel, self).__init__()
        
        self.features = AutoModel.from_pretrained(bert_dict[cfg.bert_type])
        #self.tokenizer = AutoTokenizer.from_pretrained(bert_dict[cfg.bert_type])
        
        if cfg.freeze_features: 
            for param in self.features.parameters():
                param.requires_grad = False
    
        self.out_layer = vbll.DiscClassification(self.features.config.hidden_size, 
                                                 cfg.out_features, 
                                                 cfg.reg_weight_factor * 1/train_size, 
                                                 parameterization = cfg.param, 
                                                 return_ood=cfg.return_ood,
                                                 prior_scale=cfg.prior_scale, 
                                                 wishart_scale=cfg.wishart_scale
                                                 )
        self.cfg = cfg
        
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)  # call base method to move model
        self.device = next(self.features.parameters()).device  # update tracked device
        return self  # to allow chaining .to(...)
    
    def get_features(self, x):
        output = self.features(**x).last_hidden_state[:,0,:]
        return output
    
    def forward(self, x):
        x = self.get_features(x)
        return self.out_layer(x)
  
    def get_last_layer(self):
        return self.out_layer
    