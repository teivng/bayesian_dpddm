import os
import torch
from wilds.datasets.civilcomments_dataset import CivilCommentsDataset
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from bayesian_dpddm.models.bert_model import bert_dict
from tqdm.auto import tqdm
from .common import split_dataset


# ================================================
# ===========CivilComments Utilities==============
# ================================================


class TokenizedCivilCommentsDataset(Dataset):
    def __init__(self, raw_dataset, name, tokenizer, max_length=1024):
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"Tokenizing {name} set...")
        self.tokenized_data = self._tokenize_all()
        

    def _tokenize_all(self):
        tokenized = []
        for i in tqdm(range(len(self.raw_dataset))):
            text, label, metadata = self.raw_dataset[i]

            encoded = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            encoded['input_ids'] = encoded['input_ids'].squeeze(0)
            encoded['attention_mask'] = encoded['attention_mask'].squeeze(0)
            # Flatten single-batch tensors and include label & any extras            
            item = (encoded,
                    label,
                    metadata
                    )
            tokenized.append(item)
        return tokenized

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]



def get_civilcomments_datasets(args:DictConfig):
    """Returns WILDS CivilComments Dataset objects
    We split the training set into TWO smaller validation sets set to train Phi and perform id-validation.
    
    Returns:
        dict: contains the WILDS dataset splits already configured for Bayesian D-PDDM.
    """
    pretokenized_dataset_path = f'data/civilcomments_v1.0/{args.model.bert_type}_tokenized_data_{args.dataset.frac}.pt'
    if os.path.exists(pretokenized_dataset_path):
        print('Pre-tokenized dataset exists! Loading...')
        dataset_dict = torch.load(pretokenized_dataset_path, weights_only=False)
        print('Loading succesful!')
    else:
        print('Pre-tokenized dataset not found...Tokenizing...')
        os.makedirs(args.dataset.data_dir, exist_ok=True)
        dataset = CivilCommentsDataset(root_dir=args.dataset.data_dir, download=args.dataset.download)
        
        
        trainset = dataset.get_subset('train', frac=args.dataset.frac)
        #valset = dataset.get_subset('val')
        testset = dataset.get_subset('test', frac=args.dataset.frac)
        
        # Split training 80-10-10
        new_trainset, id_val1, id_val2 = split_dataset(
            trainset, 
            [0.8, 0.1, 0.1],
            random_seed=args.seed
        )
        tokenizer = AutoTokenizer.from_pretrained(bert_dict[args.model.bert_type])
        
        new_trainset = TokenizedCivilCommentsDataset(new_trainset, 'train', tokenizer, args.dataset.max_length)
        id_val1 = TokenizedCivilCommentsDataset(id_val1, 'valid1', tokenizer, args.dataset.max_length)
        id_val2 = TokenizedCivilCommentsDataset(id_val2, 'valid2', tokenizer, args.dataset.max_length)
        testset = TokenizedCivilCommentsDataset(testset, 'test', tokenizer, args.dataset.max_length)
        
        dataset_dict = {
            'train': new_trainset,
            'valid': id_val1,
            'dpddm_train': id_val1,
            'dpddm_id': id_val2,
            'dpddm_ood': testset
        }
        torch.save(dataset_dict, pretokenized_dataset_path)
        print('Saved pre-tokenized dataset!')
    return dataset_dict
    
    