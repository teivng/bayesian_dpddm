import os 
import sys
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
from models import DPDDM_ABSTRACTCLASS
from torch.utils.data import DataLoader, Dataset

from utils import eval_acc


class DPDDM_Monitor:
    def __init__(self, model:DPDDM_ABSTRACTCLASS, 
                 trainset:Dataset, 
                 valset:Dataset, 
                 train_cfg, 
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                 ):
        
        self.model = model
        self.trainset = trainset
        self.valset = valset
        
        self.optimizer = train_cfg.OPTIMIZER(
            self.model.parameters(),
            lr=train_cfg.LR,
            weight_decay=train_cfg.WD,
        )
        self.trainloader = DataLoader(self.trainset, batch_size=train_cfg.BATCH_SIZE, shuffle=True)
        self.valloader = DataLoader(self.valset, batch_size=train_cfg.BATCH_SIZE, shuffle=True)

        self.output_metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'ood_auroc': []
        }
        self.train_cfg = train_cfg 
        self.device = device
        
    def train_model(self, tqdm_enabled=False):
        f = tqdm if tqdm_enabled else lambda x: x
        for epoch in f(range(self.train_cfg.NUM_EPOCHS)):
            self.model.train()
            running_loss = []
            running_acc = []
            for train_step, (features, labels) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                features, labels = features.to(self.device), labels.to(self.device)
                out = self.model(features)
                loss = out.train_loss_fn(labels)
                probs = out.predictive.probs
                acc = eval_acc(probs, labels).item()
                running_loss.append(loss.item())
                running_acc.append(acc)
                loss.backward()
                self.optimizer.step()
            self.output_metrics['train_loss'].append(np.mean(running_loss))
            self.output_metrics['train_acc'].append(np.mean(running_acc))
            if epoch % self.train_cfg.VAL_FREQ == 0:
                running_val_loss = []
                running_val_acc = []
                with torch.no_grad():
                    self.model.eval()
                    for test_step, (features, labels) in enumerate(self.valloader):
                        features, labels = features.to(self.device), labels.to(self.device)

                        out = self.model(features)
                        loss = out.val_loss_fn(labels)
                        probs = out.predictive.probs
                        acc = eval_acc(probs, labels).item()

                        running_val_loss.append(loss.item())
                        running_val_acc.append(acc)

                    self.output_metrics['val_loss'].append(np.mean(running_val_loss))
                    self.output_metrics['val_acc'].append(np.mean(running_val_acc))
            if epoch % 10 == 0:
                print('Epoch: {:2d}, train loss: {:4.4f}'.format(epoch, np.mean(running_loss)))
                print('Epoch: {:2d}, valid loss: {:4.4f}'.format(epoch, np.mean(np.mean(running_val_loss))))

        return self.output_metrics
    
    