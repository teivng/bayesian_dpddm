import os
os.environ['HYDRA_FULL_ERROR'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import wandb
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

import sys
parentdir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentdir)
import fcntl
from tqdm.auto import tqdm
from bayesian_dpddm.monitors import DPDDMBayesianMonitor, DPDDMFullInformationMonitor, DPDDMBERTMonitor
from bayesian_dpddm.models import ConvModel, MLPModel, ResNetModel, BERTModel

import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.multiprocessing as mp
import numpy as np
from experiments.utils import get_datasets, get_configs
from ablations.utils import * 

torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, accuracy_score

# base models
base_models = {
    'cifar10': ConvModel,
    'uci': MLPModel,
    'synthetic': MLPModel,
    'camelyon17': ResNetModel,
    'civilcomments': BERTModel,
}


monitors = {
    'bayesian': DPDDMBayesianMonitor,
    'fi': DPDDMFullInformationMonitor,
    'bert': DPDDMBERTMonitor,
}

@hydra.main(config_path='configs/', config_name='uci_best', version_base='1.2')
def main(args:DictConfig):
    
    # =========================================================
    # ========================Seeding==========================
    # =========================================================
    
    RANDOM_SEED = args.seed
    np.random.seed(RANDOM_SEED)
    torch.random.manual_seed(RANDOM_SEED)
    
    # =========================================================
    # =============Print All Configurations====================
    # =========================================================
    
    print("Hydra Configuration:")
    print(OmegaConf.to_yaml(args))
    RANDOM_SEED = args.seed
    np.random.seed(RANDOM_SEED)
    torch.random.manual_seed(RANDOM_SEED)
    
    ''' Get datasets '''
    dataset = get_datasets(args) 
    
    ''' Get config objects '''
    model_config, train_config = get_configs(args)
    
    ''' Instantiate loaders ''' 
    trainloader = torch.utils.data.DataLoader(
        dataset['train'],
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
        persistent_workers=True,
    )
    
    valloader = torch.utils.data.DataLoader(
        dataset['valid'],
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
        persistent_workers=True,
    )
    
    oodloader = torch.utils.data.DataLoader(
        dataset['dpddm_ood'],
        batch_size = train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=True, 
        persistent_workers=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # =========================================================
    # =======================Train Loop========================
    # =========================================================
    
    for i_runs in tqdm(range(10), leave=True):
        # Instantiate models and optimizers
        d3m_model = base_models[args.dataset.name](model_config, train_size=len(dataset['train'])).to(device)
        reg_model = base_models[args.dataset.name](model_config, train_size=len(dataset['train'])).to(device)
        reg_model.out_layer = nn.Linear(model_config.mid_features, model_config.out_features).to(device)
        reg_criterion = nn.CrossEntropyLoss(reduction='mean')
        # Instantiate optimizers
        d3m_opt = torch.optim.AdamW(d3m_model.parameters(), lr=train_config.lr)
        reg_opt = torch.optim.AdamW(reg_model.parameters(), lr=train_config.lr)
        
        # Train
        for epoch in tqdm(range(train_config.num_epochs), leave=False):
            d3m_model.train()
            reg_model.train()
            
            for train_step, batch in enumerate(tqdm(trainloader, leave=False)):
                features, labels, *_ = batch
                features, labels = features.to(device), labels.to(device)
                
                d3m_out = d3m_model(features)
                reg_out = reg_model(features)
                
                d3m_loss = d3m_out.train_loss_fn(labels)
                labels = labels.long()
                reg_loss = reg_criterion(reg_out, labels)

                d3m_opt.zero_grad()
                reg_opt.zero_grad()

                d3m_loss.backward()
                reg_loss.backward()

                d3m_opt.step()
                reg_opt.step()
                
        # Training complete, compute evaluation metrics.
        d3m_model.eval()
        reg_model.eval()
        all_labels = []
        all_d3m_preds = []
        all_reg_preds = []
        all_d3m_probs = []
        all_reg_probs = []

        with torch.no_grad():
            for test_step, batch in enumerate(tqdm(valloader, leave=False)):
                features, labels, *_ = batch
                features, labels = features.to(device), labels.to(device)
                
                d3m_out = d3m_model(features)
                reg_out = reg_model(features)
                
                d3m_probs = d3m_out.predictive.probs     # shape: [B, C]
                reg_probs = F.softmax(reg_out, dim=-1)   # if reg_out is logits

                d3m_preds = torch.argmax(d3m_probs, dim=-1)
                reg_preds = torch.argmax(reg_probs, dim=-1)

                all_labels.append(labels.cpu())
                all_d3m_preds.append(d3m_preds.cpu())
                all_reg_preds.append(reg_preds.cpu())
                all_d3m_probs.append(d3m_probs.cpu())
                all_reg_probs.append(reg_probs.cpu())

        # Flatten all tensors
        y_true = torch.cat(all_labels).numpy().squeeze()
        y_pred_d3m = torch.cat(all_d3m_preds).numpy()
        y_pred_reg = torch.cat(all_reg_preds).numpy()
        probs_d3m = torch.cat(all_d3m_probs).numpy()
        probs_reg = torch.cat(all_reg_probs).numpy()

        # Determine number of classes
        unique_classes = np.unique(y_true)
        num_classes = len(unique_classes)
        is_binary = num_classes == 2

        # Helper: compute metrics
        def compute_metrics(y_true, y_pred, probs):
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            mcc = matthews_corrcoef(y_true, y_pred)

            if is_binary:
                auroc = roc_auc_score(y_true, probs[:, 1])
            else:
                auroc = roc_auc_score(y_true, probs, average='macro', multi_class='ovr')

            return acc, f1, auroc, mcc

        # Compute once per model
        acc_d3m, f1_d3m, auroc_d3m, mcc_d3m = compute_metrics(y_true, y_pred_d3m, probs_d3m)
        acc_reg, f1_reg, auroc_reg, mcc_reg = compute_metrics(y_true, y_pred_reg, probs_reg)

        # Print
        print("=== D3M Model ===")
        print(f"Accuracy: {acc_d3m:.4f}")
        print(f"F1 Score: {f1_d3m:.4f}")
        print(f"AUROC: {auroc_d3m:.4f}")
        print(f"MCC: {mcc_d3m:.4f}")

        print("\n=== Reg Model ===")
        print(f"Accuracy: {acc_reg:.4f}")
        print(f"F1 Score: {f1_reg:.4f}")
        print(f"AUROC: {auroc_reg:.4f}")
        print(f"MCC: {mcc_reg:.4f}")
        

        # Save results
        results = [
            {
                "run_id": i_runs,
                "model": "D3M",
                'dataset': args.dataset.name,
                "accuracy": acc_d3m,
                "f1_score": f1_d3m,
                "auroc": auroc_d3m,
                "mcc": mcc_d3m
            },
            {
                "run_id": i_runs,
                "model": "Reg",
                'dataset': args.dataset.name,
                "accuracy": acc_reg,
                "f1_score": f1_reg,
                "auroc": auroc_reg,
                "mcc": mcc_reg
            }
        ]
        # Convert to DataFrame
        df = pd.DataFrame(results)
        '''
        # Save (append if file exists)
        csv_path = "ablations/ablation_results/reghead/reghead.csv"
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, mode='w', header=True, index=False)
        '''
if __name__ == '__main__':
    #mp.set_start_method('spawn', force=True)
    main()