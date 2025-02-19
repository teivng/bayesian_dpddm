import os
import wandb
import hydra
from omegaconf import DictConfig

import sys
parentdir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentdir)

import argparse
from bayesian_dpddm import DPDDMConvModel, DPDDMBayesianMonitor
import torch
import numpy as np

torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from experiments.utils_cifar10 import get_cifar10_datasets, get_configs


# Seeding
RANDOM_SEED = 9927
np.random.seed(RANDOM_SEED)
torch.random.manual_seed(RANDOM_SEED)

@hydra.main(config_path='configs/', config_name='defaults')
def main(args:DictConfig):
    # =========================================================
    # ==================Initialization=========================
    # =========================================================
    
    run = wandb.init(project=args.wandb_args.project,
                     config={
                         'mid_channels': args.model.mid_channels,
                         'kernel_size': args.model.kernel_size,
                         'mid_layers': args.model.mid_layers,
                         'hidden_dim': args.model.hidden_dim,
                         'reg_weight_factor': args.model.reg_weight_factor,
                         'temp': args.dpddm.temp
                         })
    
    ''' Get datasets '''
    cifar10train, cifar10test, cifar10train_with_test_transforms, cifar101 = get_cifar10_datasets()
    
    
    
    ''' wandb initialization '''
    wandb.init(
        project="bayesian_dpddm",

        # track hyperparameters and run metadata
        config={
            'mid_channels': args.mid_channels,
            'kernel_size': args.kernel_size,
            'mid_layers': args.mid_layers,
            'hidden_dim': args.hidden_dim,
            'reg_weight_factor': args.reg_weight_factor,
            'temp': args.temp
        }
    )
    
    ''' Build model and monitor '''
    base_model = DPDDMConvModel(model_config,train_size=len(cifar10train))
    monitor = DPDDMBayesianMonitor(
        model=base_model,
        trainset=cifar10train,
        valset=cifar10test,
        train_args=train_config,
        device=device,
    )
    
    # =========================================================
    # ===================Base Training=========================
    # =========================================================
    
    ''' Load base model if pretrained, else train '''
    if args.from_pretrained:
        base_model.load_state_dict(torch.load(os.path.join('saved_weights', f'{args.save_name}.pth')))
    else:
        monitor.train_model(tqdm_enabled=True)
        os.makedirs('saved_weights', exist_ok=True)
        torch.save(monitor.model.state_dict(), os.path.join('saved_weights', f'{args.save_name}.pth'))
    
    # =========================================================
    # =============D-PDDM Training and Testing=================
    # =========================================================
    
    ''' Pretrain the disagreement distribution Phi '''
    monitor.pretrain_disagreement_distribution(dataset=cifar10test,
                                               n_post_samples=args.n_post_samples,
                                               data_sample_size=args.data_sample_size,
                                               Phi_size=args.Phi_size, 
                                               temperature=args.temp,
                                               )
    
    ''' wandb log Phi statistics '''
    wandb.log({
        'Phi-mean': np.mean(monitor.Phi),
        'Phi-std': np.std(monitor.Phi),
        'Phi-med': np.median(monitor.Phi)
         
    })
    
    ''' Test TPR/FPR on all datasets '''
    stats = {}
    dis_rates = {}
    for k,dataset in {
        'cifar10-train': cifar10train_with_test_transforms,
        'cifar10-test': cifar10test,
        'cifar10.1': cifar101
    }.items():
        rate, max_dis_rates = monitor.repeat_tests(n_repeats=100,
                                      dataset=dataset, 
                                      n_post_samples=args.n_post_samples,
                                      data_sample_size=args.data_sample_size,
                                      temperature=args.temp
                                      )
        print(f"{k}: {rate}")
        stats[k] = rate
        dis_rates[k] = (np.mean(max_dis_rates), np.std(max_dis_rates))

    ''' wandb log statistics '''
    wandb.log({
        'fpr_train': stats['cifar10-train'],
        'fpr_test': stats['cifar10-test'],
        'tpr': stats ['cifar10.1']
    })
    wandb.log({
        'dis_rates_train': dis_rates['cifar10-train'],
        'dis_rates_test': dis_rates['cifar10-test'],
        'dis_rates_ood': dis_rates['cifar10.1']
    })
    
    return 0


if __name__ == '__main__':
    main()