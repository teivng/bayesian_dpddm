import os 
import sys
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
from models import DPDDM_ABSTRACTMODEL
from torch.utils.data import DataLoader, Dataset

from utils import temperature_scaling, sample_from_dataset

class DPDDM_Bayesian_Monitor:
    """Defines the Bayesian DPDDM Monitor (Algorithms 3 and 4)
    
    Attributes:
        model (DPDDM_ABSTRACTMODEL): the model class, i.e. hypothesis class for the base classifier
        trainset (Dataset): torch training dataset
        valset (Dataset): torch validation dataset
        optimizer (torch.optim.Optimizer): optimizer for the base classifier
        trainloader (DataLoader): train set loader
        valloader (DataLoader): validation set loader
        output_metrics (dict): dictionary containing the training metrics for the base classifier
        train_cfg (train_cfg): data object for the training configuration
        device (torch.device): device, cuda or cpu
        Phi (list): distribution of disagreement rates, to be populated
    """

    def __init__(self, model:DPDDM_ABSTRACTMODEL, 
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
        
        # For DPDDM pretraining
        self.Phi = []
    
    def eval_acc(self, preds:torch.tensor, y:torch.tensor):
        """Evaluates the accuracy of the Bayesian model

        Args:
            preds (torch.tensor): predictions
            y (torch.tensor): labels

        Returns:
            float: accuracy score of prediction
        """
        map_preds = torch.argmax(preds, dim=1)
        return (map_preds == y).float().mean()


    def train_model(self, tqdm_enabled=False):
        """Initial training of the model

        Args:
            tqdm_enabled (bool, optional): Enables tqdm during training. Defaults to False.

        Returns:
            dict: dictionary containing training metrics
        """
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
                acc = self.eval_acc(probs, labels).item()
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
                        acc = self.eval_acc(probs, labels).item()

                        running_val_loss.append(loss.item())
                        running_val_acc.append(acc)

                    self.output_metrics['val_loss'].append(np.mean(running_val_loss))
                    self.output_metrics['val_acc'].append(np.mean(running_val_acc))
            if epoch % 10 == 0:
                print('Epoch: {:2d}, train loss: {:4.4f}'.format(epoch, np.mean(running_loss)))
                print('Epoch: {:2d}, valid loss: {:4.4f}'.format(epoch, np.mean(np.mean(running_val_loss))))

        return self.output_metrics
    
    def get_pseudolabels(self, X):
        """Given samples X, return pseudolabels assigned by self.model

        Args:
            X (torch.tensor): input tensor

        Returns:
            torch.tensor: pseudolabels labeled by self.model
        """
        self.model.eval()
        X = X.to(self.device)
        with torch.no_grad():
            features = self.model.get_features(X)
            ll_dist = self.model.out_layer.logit_predictive(features)
            y_hat = torch.argmax(ll_dist.loc, 1)
        return y_hat

    def compute_max_dis_rate(self, X, y, n_post_samples=5000, temperature=1):
        """Approximates the maximum disagreement rate among sampled weights.

        Args:
            X (torch.tensor): input tensor
            y (torch.tensor): pseudolabels
            n_post_samples (int, optional): number of posterior weights. Defaults to 5000.
            temperature (int, optional): softening of the logits. Defaults to 1.

        Returns:
            float: approximate maximum disagreement rate
        """
        self.model.eval()
        with torch.no_grad():
            features = self.model.get_features(X)
            ll_dist = self.model.out_layer.logit_predictive(features)
            logits_samples = ll_dist.rsample(sample_shape=torch.Size([n_post_samples]))
            
            # scale down with temperature
            logits_samples = temperature_scaling(logits_samples, temperature)
            y_hat = torch.argmax(logits_samples, -1)
            y_tile = torch.tile(y, (n_post_samples, 1)).cuda()
            dis_mat = (y_hat != y_tile)
            dis_rate = dis_mat.sum(dim=-1)/len(y)
    
        return torch.max(dis_rate).item()


    def pretrain_disagreement_distribution(self, dataset, n_post_samples=5000, data_sample_size=1000, Phi_size=500, temperature=1, tqdm_enabled=True):
        """Given a dataset, generates the Phi distribution of maximum disagreement rates.

        Args:
            dataset (Dataset): dataset object
            n_post_samples (int, optional): number of posterior weights. Defaults to 5000.
            data_sample_size (int, optional): size of bootstraped dataset. Defaults to 1000.
            Phi_size (int, optional): size of phi. Defaults to 500.
            temperature (int, optional): softening of the logits. Defaults to 1.
        """
        f = tqdm if tqdm_enabled else lambda x: x 
        self.model.eval()
        with torch.no_grad():
            for i in f(range(Phi_size)):
                X = sample_from_dataset(n_samples=data_sample_size, dataset=dataset, device=self.device)
                y_pseudo = self.get_pseudolabels(X)
                X, y_pseudo = X.to(self.device), y_pseudo.to(self.device)
                max_dis_rate = self.compute_max_dis_rate(X, y_pseudo, n_post_samples=n_post_samples, temperature=temperature)
                self.Phi.append(max_dis_rate)