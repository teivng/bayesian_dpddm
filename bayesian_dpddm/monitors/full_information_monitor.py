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


class DPDDMFullInformationMoniter(DPDDMMonitor):
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
        super(DPDDMBayesianMonitor, self).__init__(*args, **kwargs)

        self.full_network_ft = full_network_ft
        
        # Over-write the models final layer
        self.model.out_layer = torch.nn.Linear(self.model.cfg.mid_features, self.model.cfg.out_features)
        self.rejection_loss_fn = FILoss()
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
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
                loss = self.loss_fn(out, labels)
                probs = out.predictive.probs
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
        X, y, mask = X.to(self.device), y.to(self.device), mask.to(self.device)

        disagreement_model = copy.deepcopy(self.model)
        disagreement_model.train()
        disagreement_model.to(self.device)

        # Create a datalaoder with the 50 ood samples and the train dataset in order to learn to agree with train and disagree with ood.
        joint_dataset = (
            MaskedDataset(self.trainset, mask=True) + 
            MaskedDataset(TensorDataset(X, y), mask=False)
        )

        rejection_loader = DataLoader(joint_dataset, batch_size=32, shuffle=True)

        # TODO: Implement the fine tuning of disagreement_mode
        if not self.full_network_ft:
            for param in self.disagreement_model.parameters():
                param.requires_grad = False
        
        for epoch in (range(10)):
            for train_step, (features, labels, mask) in enumerate(rejection_loader):
                self.optimizer.zero_grad()
                features, labels, mask = features.to(self.device), labels.to(self.device), mask.to(self.device)
                out = self.model(features)
                loss = self.rejection_loss_fn(out, labels, mask)
                loss.backward()
                self.optimizer.step()
        
        # Post training, compute the disagreement rate + accuracy on original dataset
                
        disagreement_model.eval()
        y_hat_ft = disagreement_model(X)
        dis_rate = (y != y_hat_ft).sum() / len(y)
        
        return dis_rate
    

    # def dpddm_test(self, dataset:Dataset, data_sample_size:int=1000, alpha=0.95, balance_ratio=0.5, replace=True, *args, **kwargs) -> Tuple[float, bool]:
    #     """Given a dataset, computes the maximum disagreement rate as well as the OOD verdict.
    #     Used to both generate Phi and Algorithms 2 and 4.

    #     Args:
    #         dataset (Dataset): dataset object
    #         data_sample_size (int, optional): size of bootstraped dataset. Defaults to 1000
    #         alpha (float): statistical power of the test. Defaults to 0.95
    #         replace (bool): sample with replacement. Defaults to True

    #     Returns:
    #         tuple(float, bool): 2-tuple containing:
    #         - maximum disagreement rate achievable by models from the same hypothesis class
    #             while (approximately) maintaining correctness on training set
    #         - OOD verdict w.r.t. self.Phi
    #     """
    #     with torch.no_grad():
    #         X, _, mask = joint_sample_from_datasets(n_samples=data_sample_size, datasetA=self.trainset, datasetB=dataset, balance_ratio=balance_ratio, replace=replace)
    #         y_pseudo = self.get_pseudolabels(X)
    #         max_dis_rate = self.compute_max_dis_rate(X, y_pseudo, mask=mask *args, **kwargs)
    #     return max_dis_rate, max_dis_rate >= np.quantile(self.Phi, alpha) if self.Phi != [] else 0 