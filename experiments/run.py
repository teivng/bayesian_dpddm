import os
os.environ['HYDRA_FULL_ERROR'] = "1"
import wandb
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

import sys
parentdir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentdir)
import fcntl

from bayesian_dpddm import ConvModel, DPDDMBayesianMonitor, MLPModel
import torch
import numpy as np

torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from experiments.utils import get_datasets, get_configs


# Seeding
#RANDOM_SEED = 9927
#np.random.seed(RANDOM_SEED)
#torch.random.manual_seed(RANDOM_SEED)


# base models
base_models = {
    'cifar10': ConvModel,
    'uci': MLPModel,
}

@hydra.main(config_path='configs/', config_name='defaults', version_base='1.2')
def main(args:DictConfig):
    # =========================================================
    # ========================Seeding==========================
    # =========================================================
    
    RANDOM_SEED = args.seed
    np.random.seed(RANDOM_SEED)
    torch.random.manual_seed(RANDOM_SEED)
    
    print(args.dpddm.data_sample_size)
    print(args.model.mid_channels)
    
    print("Final Config:\n", OmegaConf.to_yaml(args))
    
    exit()
    
    # =========================================================
    # =============Print All Configurations====================
    # =========================================================
    
    print("Hydra Configuration:")
    print(OmegaConf.to_yaml(args))
    
    # ==================Initialization=========================
    # =========================================================
    
    ''' wandb initialization '''
    wandb.config = OmegaConf.to_container(
        args, resolve=True, throw_on_missing=True
    )
    
    run = wandb.init(project=args.wandb_cfg.project,
                     settings=wandb.Settings(start_method='thread') # for hydra
                     )
    
    ''' Get datasets '''
    dataset = get_datasets(args) 
    
    ''' Get config objects '''
    model_config, train_config = get_configs(args)
    
    ''' Build model and monitor '''
    base_model = base_models[args.dataset.name](model_config, train_size=len(dataset['train']))
    monitor = DPDDMBayesianMonitor(
        model=base_model,
        trainset=dataset['train'],
        valset=dataset['valid'],
        train_cfg=train_config,
        device=device,
    )
    
    ''' Log random seed '''
    wandb.log({'seed': args.seed})
    
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
        rate, max_dis_rates = monitor.repeat_tests(n_repeats=args.dpddm.n_repeats,
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
        'dr_id': dis_rates['dpddm_id'],
        'dr_ood': dis_rates['dpddm_ood']
    })
    
    ''' Self-logging initialization '''
    logger = {}
    log_dir = 'results/'
    os.makedirs(log_dir, exist_ok=True)
    
    logger['seed'] = args.seed
    logger['data_sample_size'] = args.dpddm.data_sample_size
    
    ''' self-log statistics '''
    logger['fpr_train'] = stats['dpddm_train']
    logger['fpr_id'] = stats['dpddm_id']
    logger['tpr'] = stats['dpddm_ood']
    
    ''' write self-log to file '''
    csv_path = os.path.join(log_dir, f'results_cifar10_{args.dpddm.data_sample_size}.csv')
    csv_exists = os.path.isfile(csv_path)
    
    with open(csv_path, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        df = pd.DataFrame([logger])
        if not csv_exists:
            df.to_csv(f, index=False)
        else:
            df.to_csv(f, mode='a', header=False, index=False)
        fcntl.flock(f, fcntl.LOCK_UN)
    return 0


if __name__ == '__main__':
    main()