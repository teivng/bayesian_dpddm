from tqdm import tqdm
import wandb 
import numpy as np
import torch
import multiprocessing
from torch.utils.data import DataLoader, Dataset, TensorDataset
import copy

from ..models.base import DPDDMAbstractModel
from ..configs import TrainConfig
from .utils import temperature_scaling, sample_from_dataset, get_class_from_string, FILoss, joint_sample_from_datasets, MaskedDataset
from .monitor import DPDDMMonitor


class DPDDMFullInformationMonitor(DPDDMMonitor):
    """ Defines the Full information version of the DPDDM Moniter.
    
    Attributes:
        model (DPDDMAbstractModel): the model class, i.e. hypothesis class for the base classifier
        trainset (Dataset): torch training dataset
        valset (Dataset): torch validation dataset
        train_cfg (TrainConfig): TrainConfig object configuring all aspects of training
        device (torch.device): torch.device, cuda or cpu
    """

    def __init__(self, full_network_ft=False, *args, **kwargs, ):
        """
        full_netowrk_ft (bool): whether to fine-tune the full network (if false, only the last layer is fine-tuned)
        """
        super().__init__(*args, **kwargs)

        self.full_network_ft = full_network_ft
        
        # Over-write the models final layer
        self.model.out_layer = torch.nn.Linear(self.model.cfg.mid_features, self.model.cfg.out_features)
        self.model.to(self.device)
        self.rejection_loss_fn = FILoss()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.probs = torch.nn.Softmax()

        
    
    def train_model(self, tqdm_enabled=False):
        """Initial training of the model.

        Args:
            tqdm_enabled (bool, optional): Enables tqdm during training. Defaults to False.

        Returns:
            dict: dictionary containing training metrics
        """
        f = tqdm if tqdm_enabled else lambda x: x 
        for epoch in f(range(self.train_cfg.num_epochs)):
            self.model.train()
            running_loss = []
            running_acc = []
            for train_step, (features, labels) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                features, labels = features.to(self.device), labels.to(self.device)
                out = self.model(features)
                loss = self.loss_fn(out, labels.to(torch.long))
                with torch.no_grad():
                    probs = self.probs(out)
                    acc = self.eval_acc(probs, labels).item()
                running_loss.append(loss.item())
                running_acc.append(acc)
                loss.backward()
                self.optimizer.step()
            self.output_metrics['train_loss'].append(np.mean(running_loss))
            self.output_metrics['train_acc'].append(np.mean(running_acc))
            if epoch % self.train_cfg.val_freq == 0:
                running_val_loss = []
                running_val_acc = []
                with torch.no_grad():
                    self.model.eval()
                    for test_step, (features, labels) in enumerate(self.valloader):
                        features, labels = features.to(self.device), labels.to(self.device)

                        out = self.model(features)
                        loss = self.loss_fn(out, labels.to(torch.long))
                        with torch.no_grad():
                            probs = self.probs(out)
                            acc = self.eval_acc(probs, labels).item()

                        running_val_loss.append(loss.item())
                        running_val_acc.append(acc)

                    self.output_metrics['val_loss'].append(np.mean(running_val_loss))
                    self.output_metrics['val_acc'].append(np.mean(running_val_acc))
            if epoch % 10 == 0:
                print('Epoch: {:2d}, train loss: {:4.4f}'.format(epoch, np.mean(running_loss)))
                print('Epoch: {:2d}, valid loss: {:4.4f}'.format(epoch, np.mean(np.mean(running_val_loss))))
            
            # wandb logging
            if wandb.run is not None:
                wandb.log({
                    'train_loss': self.output_metrics['train_loss'][-1],
                    'train_acc': self.output_metrics['train_acc'][-1],
                    'val_loss': self.output_metrics['val_loss'][-1],
                    'val_acc': self.output_metrics['val_acc'][-1],
                })

        return self.output_metrics

    def get_pseudolabels(self, X:torch.tensor):
        """Given samples X, return pseudolabels assigned by self.model

        Args:
            X (torch.tensor): input tensor

        Returns:
            torch.tensor: pseudolabels labeled by self.model
        """
        self.model.eval()
        
        X = X.to(self.device)
        with torch.no_grad():
            logits = self.model(X)
            y_hat = torch.argmax(logits, dim =1)
        return y_hat
    
    def compute_max_dis_rate(self, X, y, *args, **kwargs):
        """Approximates the maximum disagreement rate among sampled weights.

        Args:
            X (torch.tensor): input tensor
            y (torch.tensor): pseudolabels
            n_post_samples (int, optional): number of posterior weights. Defaults to 5000.
            temperature (int, optional): softening of the logits. Defaults to 1.

        Returns:
            float: approximate maximum disagreement rate
        """
        disagreement_model = self.model
        disagreement_model.train()
        disagreement_model.to(self.device)

        opt_cls = get_class_from_string(self.train_cfg.optimizer)
        disagreement_optimizer =  opt_cls(
            disagreement_model.parameters(),
            lr=self.train_cfg.lr,
            weight_decay=self.train_cfg.wd,
        )

        # Create a datalaoder with the 50 ood samples and the train dataset in order to learn to agree with train and disagree with ood.
        joint_dataset = (
            MaskedDataset(self.trainset, mask=True) + 
            MaskedDataset(TensorDataset(X, y.cpu()), mask=False)
        )

        rejection_loader = DataLoader(joint_dataset, batch_size=32, shuffle=True)

        # TODO: Implement the fine tuning of disagreement_mode

        # if not self.full_network_ft:
        #     for param in disagreement_model.parameters():
        #         param.requires_grad = Falses

        X, y = X.to(self.device), y.to(self.device)

        disagreement_model.to(self.device)
        with torch.set_grad_enabled(True):
            for epoch in (range(10)):
                disagreement_model.train()
                for train_step, (features, labels, mask) in enumerate(rejection_loader):
                    disagreement_optimizer.zero_grad()
                    features, labels, mask = features.to(self.device), labels.to(torch.long).to(self.device), mask.to(self.device)
                    out = disagreement_model(features)
                    loss = self.rejection_loss_fn(out, labels, mask)
                    loss.backward()
                    disagreement_optimizer.step()
                                
            disagreement_model.eval()
            y_hat_ft = torch.argmax(disagreement_model(X), dim=1)
            dis_rate = (y != y_hat_ft).sum() / len(y)
        
        return dis_rate.item()
