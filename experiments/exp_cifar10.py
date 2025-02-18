import os

import argparse
from bayesian_dpddm import DPDDMConvModel, DPDDMBayesianMonitor
import torch
import numpy as np

torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from .utils_cifar10 import get_cifar10_datasets, get_configs


# Seeding
RANDOM_SEED = 9927
np.random.seed(RANDOM_SEED)
torch.random.manual_seed(RANDOM_SEED)

    
def main():
    ''' Get datasets '''
    cifar10train, cifar10test, cifar101 = get_cifar10_datasets()
    
    ''' Parse model configuration '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--mid_channels', type=int, default=64)
    parser.add_argument('--out_features', type=int, default=10)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--mid_layers', type=int, default=3)
    parser.add_argument('--pool_dims', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0) # per Resnet paper
    parser.add_argument('--reg_weight', type=float, default=1/len(cifar10train)*10)
    parser.add_argument('--param', type=str, default='diagonal')
    parser.add_argument('--prior_scale', type=float, default=1.0)
    parser.add_argument('--wishart_scale', type=float, default=1.0)
    
    ''' Parse training configuration '''
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=0.0001)


    ''' Parse DPDDM configuration '''
    parser.add_argument('--Phi_size', type=int, default=500)
    parser.add_argument('--n_post_samples', type=int, default=5000)
    parser.add_argument('--temp', type=float, default=1)
    parser.add_argument('--data_sample_size', type=int, default=1000)
    
    ''' Misc. configuration '''
    parser.add_argument('--from_pretrained', action=argparse.BooleanOptionalAction)
    parser.add_argument('--save_name', type=str, default='cnn_model')

    args = parser.parse_args()
    
    ''' Get configs '''
    model_config, train_config = get_configs(args)
    
    ''' Build model and monitor '''
    base_model = DPDDMConvModel(model_config)
    monitor = DPDDMBayesianMonitor(
        model=base_model,
        trainset=cifar10train,
        valset=cifar10test,
        train_cfg=train_config,
        device=device,
    )
    
    ''' Load base model if pretrained, else train '''
    if args.from_pretrained:
        base_model.load_state_dict(torch.load(os.path.join('saved_weights', f'{args.save_name}.pth')))
    else:
        monitor.train_model(tqdm_enabled=True)
        os.makedirs('saved_weights', exist_ok=True)
        torch.save(monitor.model.state_dict(), os.path.join('saved_weights', f'{args.save_name}.pth'))
    
    ''' Pretrain the disagreement distribution Phi '''
    monitor.pretrain_disagreement_distribution(dataset=cifar10test,
                                               n_post_samples=args.n_post_samples,
                                               data_sample_size=args.data_sample_size, 
                                               temperature=args.temp,
                                               )
    
    ''' Test TPR/FPR on all datasets '''
    for k,dataset in {
        'cifar10-train': cifar10train,
        'cifar10-test': cifar10test,
        'cifar10.1': cifar101
    }.items():
        tpr, _ = monitor.repeat_tests(n_repeats=100,
                                      dataset=dataset, 
                                      n_post_samples=args.n_post_samples,
                                      data_sample_size=args.data_sample_size,
                                      temperature=args.temp)
        print(f"{k}: {tpr}")


    return 0


if __name__ == '__main__':
    main()