
import os
import torch
import numpy as np
import pandas as pd
import pickle 

from torch import nn
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import KFold


class Microbe_Dataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
 

class Drug_Asso_Dataset(Dataset):
    def __init__(self, args, drug_dataset, train_idx):
        super().__init__()
        if args.path_drug_emb.split('/')[-1] == "dataset_MDAD_drug_sim.pk":
            self.drug_emb = drug_dataset[:, train_idx]
        else:
            self.drug_emb = drug_dataset

        adj = np.loadtxt(args.path_adj).astype(int) # [N_association, 3]
        self.association = self.adj_to_wider(adj)

    def __len__(self):
        return len(self.drug_emb)

    def __getitem__(self, idx):
        dataset = {'drug_emb': self.drug_emb[idx],
                   'association': torch.from_numpy(self.association)[idx]}
        return dataset
    
    def adj_to_wider(self, adj):
        adj_df = pd.DataFrame(adj).rename(columns = {0: 'Drug_ID', 1: 'Microbe_ID', 2: 'association'})
        association = adj_df.pivot(index='Drug_ID', columns='Microbe_ID').fillna(0).astype(int)
        return np.array(association)


def get_folds(args, drug_emb, n_splits):
    indices = list(range(len(drug_emb)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = []

    test_indices = pd.DataFrame()
    for i, (train_idx, test_idx) in enumerate(kf.split(indices)):
        dataset = Drug_Asso_Dataset(args, drug_emb, train_idx)

        np.random.seed(42)
        np.random.shuffle(train_idx)
        num_train = (int)(len(train_idx) * 0.8)
        tmp_idx = train_idx
        train_idx = tmp_idx[:num_train]
        valid_idx = tmp_idx[num_train:]

        test_idx.sort()

        train = Subset(dataset, train_idx)
        valid = Subset(dataset, valid_idx)
        test = Subset(dataset, test_idx)
        folds.append((train, valid, test))

        test_idx_df = pd.DataFrame({"Fold": i+1, "Test_idx": test_idx})
        test_indices = pd.concat([test_indices, test_idx_df])

    # Output Test indices
    test_indices.to_csv(os.path.join(args.logpath, f"test_indices.csv"), index=False)

    return folds


def get_data(args):
    # Get drug embeddings
    path = args.path_drug_emb
    if os.path.isfile(path) == False:
        print(f"File not found: {path}")
    else:  # load datasetfile
        print(f"Importing drug molecular embeddings from {path}...")
        with open(path, 'rb') as f:
            drug_emb = pickle.load(f)
 
    drug_folds = get_folds(args, drug_emb=drug_emb, n_splits=args.n_splits)

    # Get microbe embeddings
    path = args.path_microbe_emb
    if os.path.isfile(path) == False:
        print(f"File not found: {path}")
    else:  # load datasetfile
        print(f"Importing microbial genome embeddings from {path}...")
        with open(path, 'rb') as f:
            micro_emb = pickle.load(f)

    micro_dataset = Microbe_Dataset(micro_emb)

    return drug_folds, micro_dataset


def my_collate_fn(batch, microbe_data, r):
    drug_collate, microbe_collate, association_collate = [],[],[]
    for item in batch:
        for j, microbe in enumerate(microbe_data):
            drug_collate.append(item['drug_emb'])
            microbe_collate.append(microbe)
            association_collate.append(item['association'][j])

    drug_collate = torch.from_numpy(np.array(drug_collate))
    microbe_collate = torch.from_numpy(np.array(microbe_collate))
    association_collate = torch.from_numpy(np.array(association_collate))

    if r is not None:
        pos_indices = (association_collate == 1).nonzero(as_tuple=True)[0]
        neg_indices = (association_collate == 0).nonzero(as_tuple=True)[0]

        np.random.seed(42)
        torch.manual_seed(42)

        sampled_pos_indices = pos_indices
        sampled_neg_indices = neg_indices[torch.randperm(len(neg_indices))[:int(len(pos_indices) * r)]]
    
        sampled_indices = torch.cat([sampled_pos_indices, sampled_neg_indices])
        shuffled_indices = sampled_indices[torch.randperm(len(sampled_indices))]
    
        drug_collate = drug_collate[shuffled_indices]
        microbe_collate = microbe_collate[shuffled_indices]
        association_collate = association_collate[shuffled_indices]

    return {'drug_emb': drug_collate, 
            'microbe_emb': microbe_collate, 
            'association': association_collate}

