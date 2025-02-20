from tqdm import tqdm
import wandb 
import numpy as np
import torch
import multiprocessing
from torch.utils.data import DataLoader, Dataset

from ..models.base import DPDDMAbstractModel
from ..configs import TrainConfig
from .utils import temperature_scaling, sample_from_dataset, get_class_from_string


class DPDDMBayesianMonitor:
    """Defines the Bayesian DPDDM Monitor (Algorithms 3 and 4)
    
    Attributes:
        model (DPDDMAbstractModel): the model class, i.e. hypothesis class for the base classifier
        trainset (Dataset): torch training dataset
        valset (Dataset): torch validation dataset
        optimizer (torch.optim.Optimizer): optimizer for the base classifier
        trainloader (DataLoader): train set loader
        valloader (DataLoader): validation set loader
        output_metrics (dict): dictionary containing the training metrics for the base classifier
        train_cfg (TrainConfig): data object for the training configuration
        device (torch.device): device, cuda or cpu
        Phi (list): distribution of disagreement rates, to be populated
    """

    def __init__(self, model:DPDDMAbstractModel, 
                 trainset:Dataset, 
                 valset:Dataset, 
                 train_cfg:TrainConfig, 
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                 ):

        self.model = model
        self.trainset = trainset
        self.valset = valset
        
        # Get optimizer from string
        opt_cls = get_class_from_string(train_cfg.optimizer)
        self.optimizer = opt_cls(
            self.model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.wd,
        )
        
        num_cpus = multiprocessing.cpu_count()
        print(f'Setting the number of DataLoader workers to the number of CPUs available: {num_cpus}')
        self.trainloader = DataLoader(self.trainset, batch_size=train_cfg.batch_size, shuffle=True, num_workers=num_cpus)
        self.valloader = DataLoader(self.valset, batch_size=train_cfg.batch_size, shuffle=True, num_workers=num_cpus)

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

        # To device
        self.model = self.model.to(self.device)
        
        
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
        for epoch in f(range(self.train_cfg.num_epochs)):
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
            features = self.model.get_features(X)
            ll_dist = self.model.out_layer.logit_predictive(features)
            y_hat = torch.argmax(ll_dist.loc, 1)
        return y_hat


    def compute_max_dis_rate(self, X:torch.tensor, y:torch.tensor, n_post_samples=5000, temperature=1):
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


    def pretrain_disagreement_distribution(self, dataset:Dataset, n_post_samples=5000, data_sample_size=1000, Phi_size=500, temperature=1, tqdm_enabled=True):
        """Given a dataset, generates the Phi distribution of maximum disagreement rates.
        Uses self.dpddm_test to compute Phi. 

        Args:
            dataset (Dataset): dataset object
            n_post_samples (int, optional): number of posterior weights. Defaults to 5000.
            data_sample_size (int, optional): size of bootstraped dataset. Defaults to 1000.
            Phi_size (int, optional): size of phi. Defaults to 500.
            temperature (int, optional): softening of the logits. Defaults to 1.
        """
        f = tqdm if tqdm_enabled else lambda x: x 
        # save temperature
        self.temperature = temperature
        self.model.eval()
        with torch.no_grad():
            for i in f(range(Phi_size)):
                max_dis_rate, _ = self.dpddm_test(dataset, n_post_samples, data_sample_size, self.temperature)
                self.Phi.append(max_dis_rate)
    
    
    def dpddm_test(self, dataset:Dataset, n_post_samples=5000, data_sample_size=1000, temperature=1, alpha=0.95):
        """Given a dataset, computes the maximum disagreement rate as well as the OOD verdict.
        Used to both generate Phi and Algorithm 4

        Args:
            dataset (Dataset): dataset object
            n_post_samples (int, optional): number of posterior weights. Defaults to 5000.
            data_sample_size (int, optional): size of bootstraped dataset. Defaults to 1000.
            temperature (int, optional): softening of the logits. Defaults to 1.

        Returns:
            _type_: _description_
        """
        with torch.no_grad():
            X = sample_from_dataset(n_samples=data_sample_size, dataset=dataset, device=self.device)
            y_pseudo = self.get_pseudolabels(X)
            X, y_pseudo = X.to(self.device), y_pseudo.to(self.device)
            max_dis_rate = self.compute_max_dis_rate(X, y_pseudo, n_post_samples=n_post_samples, temperature=temperature)
        return max_dis_rate, max_dis_rate >= np.quantile(self.Phi, alpha) if self.Phi != [] else 0 

    
    def repeat_tests(self, n_repeats=1, *args, **kwargs):
        assert self.Phi != []
        """After training Phi, monitors D-PDD using the DPDDM test (Algorithm 4) on dataset
    
        Args:
            dataset (Dataset): dataset to run DPDDM test
            n_post_samples (int, optional): number of posterior weights. Defaults to 5000.
            data_sample_size (int, optional): size of bootstraped dataset (few-shot evaluations). Defaults to 1000.
            n_repeats (int, optional): number of times to repeat independent realizations of DPDDM test (TPR/FPR calculations). Defaults to 1.
        """

        self.model.eval()
        with torch.no_grad():
            tprs = []
            max_dis_rates = []
            for i in tqdm(range(n_repeats)):
                max_dis_rate, result = self.dpddm_test(*args, **kwargs)
                tprs.append(result)
                max_dis_rates.append(max_dis_rate)
            return np.mean(tprs), max_dis_rates
            