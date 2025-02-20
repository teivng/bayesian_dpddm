import os
os.environ['HYDRA_FULL_ERROR'] = "1"
import wandb
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

import sys
parentdir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentdir)

import argparse
from bayesian_dpddm import DPDDMConvModel, DPDDMBayesianMonitor
import torch
import numpy as np

torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from experiments.utils import get_datasets, get_configs


# Seeding
RANDOM_SEED = 9927
np.random.seed(RANDOM_SEED)
torch.random.manual_seed(RANDOM_SEED)

@hydra.main(config_path='configs/', config_name='defaults', version_base='1.2')
def main(args:DictConfig):

    # =========================================================
    # =============Print All Configurations====================
    # =========================================================
    
    print("Hydra Configuration:")
    print(OmegaConf.to_yaml(args))
    
    # ==================Initialization=========================
    # =========================================================
    
    ''' wandb initialization '''
    run = wandb.init(project=args.wandb_cfg.project,
                     config={
                         'mid_channels': args.model.mid_channels,
                         'kernel_size': args.model.kernel_size,
                         'mid_layers': args.model.mid_layers,
                         'hidden_dim': args.model.hidden_dim,
                         'reg_weight_factor': args.model.reg_weight_factor,
                         'temp': args.dpddm.temp
                         },
                     settings=wandb.Settings(start_method='thread') # for hydra
                     )
    
    ''' Get datasets '''
    dataset = get_datasets(args) 
    
    ''' Get config objects '''
    model_config, train_config = get_configs(args)
    
    ''' Build model and monitor '''
    base_model = DPDDMConvModel(model_config,train_size=len(dataset['train']))
    monitor = DPDDMBayesianMonitor(
        model=base_model,
        trainset=dataset['train'],
        valset=dataset['valid'],
        train_cfg=train_config,
        device=device,
    )
    
    # =========================================================
    # ==============Base Classifier Training===================
    # =========================================================
    
    ''' Load base model if pretrained, else train '''
    if args.from_pretrained:
        base_model.load_state_dict(torch.load(os.path.join('saved_weights', f'{args.dataset.name}.pth')))
    else:
        monitor.train_model(tqdm_enabled=True)
        os.makedirs('saved_weights', exist_ok=True)
        torch.save(monitor.model.state_dict(), os.path.join('saved_weights', f'{args.dataset.name}.pth'))
    
    # =========================================================
    # ===================D-PDDM Training=======================
    # =========================================================
    
    ''' Pretrain the disagreement distribution Phi '''
    monitor.pretrain_disagreement_distribution(dataset=dataset['dpddm_train'],
                                               n_post_samples=args.dpddm.n_post_samples,
                                               data_sample_size=args.dpddm.data_sample_size,
                                               Phi_size=args.dpddm.Phi_size, 
                                               temperature=args.dpddm.temp,
                                               )
    
    ''' wandb log Phi statistics '''
    wandb.log({
        'Phi-mean': np.mean(monitor.Phi),
        'Phi-std': np.std(monitor.Phi),
        'Phi-med': np.median(monitor.Phi)
    })
    
    # =========================================================
    # ===================D-PDDM Testing========================
    # =========================================================
    
    ''' Test TPR/FPR on all datasets '''
    stats = {}
    dis_rates = {}
    for k,dataset in {
        'dpddm_train': dataset['dpddm_train'],
        'dpddm_id': dataset['dpddm_id'],
        'dpddm_ood': dataset['dpddm_ood']
    }.items():
        rate, max_dis_rates = monitor.repeat_tests(n_repeats=100,
                                      dataset=dataset, 
                                      n_post_samples=args.dpddm.n_post_samples,
                                      data_sample_size=args.dpddm.data_sample_size,
                                      temperature=args.dpddm.temp
                                      )
        print(f"{k}: {rate}")
        stats[k] = rate
        dis_rates[k] = (np.mean(max_dis_rates), np.std(max_dis_rates))

    ''' wandb log statistics '''
    wandb.log({
        'fpr_train': stats['dpddm_train'],
        'fpr_id': stats['dpddm_id'],
        'tpr': stats ['dpddm_ood']
    })
    wandb.log({
        'dr_train': dis_rates['dpddm_train'],
        'dr_test': dis_rates['dpddm_id'],
        'dr_ood': dis_rates['dpddm_ood']
    })
    
    return 0


if __name__ == '__main__':
    main()