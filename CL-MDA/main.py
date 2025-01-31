
import argparse, time, random, os, json
import torch
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from functools import partial

from dataset import get_data, my_collate_fn
from model import CL4MDA
from train import Trainer

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def set_seed(s):                                                                                              
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

    torch.cuda.manual_seed_all(s)
    #add additional seed
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms = True


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0, pos_weight=1.0, neg_weight=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, x0, x1, y):
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = self.pos_weight * y * dist_sq + self.neg_weight * (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]

        return loss


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)


def add_fold_scores(fold_scores, metrics, auc, auprc):
    fold_scores["accuracy"].append(metrics["accuracy"])
    fold_scores["precision"].append(metrics["precision"])
    fold_scores["recall"].append(metrics["recall"])
    fold_scores["f1"].append(metrics["f1"])
    fold_scores["auc"].append(auc)
    fold_scores["auprc"].append(auprc)

    return fold_scores


def calculate_print_metrics(fold_scores):
    metrics = {}
    for metric_name, scores in fold_scores.items():
        metrics[metric_name] = {
            "mean": np.mean(scores),
            "std": np.std(scores)
        }
        
    print(f"Average Accuracy: {metrics['accuracy']['mean']:.2f}% ({metrics['accuracy']['std']:.2f}), "
          f"Average Precision: {metrics['precision']['mean']:.2f}% ({metrics['precision']['std']:.2f})")
    print(f"Average Recall: {metrics['recall']['mean']:.2f}% ({metrics['recall']['std']:.2f}), "
          f"Average F1: {metrics['f1']['mean']:.2f}% ({metrics['f1']['std']:.2f})")
    print(f"Average AUC: {metrics['auc']['mean']:.4f} ({metrics['auc']['std']:.4f}), "
          f"Average AUPRC: {metrics['auprc']['mean']:.4f} ({metrics['auprc']['std']:.4f})")
    print()


def main(args):
    time_start = time.time()
    folds, microbe_data = get_data(args)
    n_microbe = len(microbe_data)
    microbe_dim = len(microbe_data[0])

    fold_scores1 = {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": [], "auprc": []}
    fold_scores2 = {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": [], "auprc": []}

    for fold_idx, (train_dataset, valid_dataset, test_dataset) in enumerate(folds):
        print(f"Starting fold {fold_idx + 1}/{args.n_splits}...")
        save_model_path = os.path.join(args.logpath, f"model_fold{fold_idx + 1}.pth")
        print("Model path: " + save_model_path)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                  collate_fn=partial(my_collate_fn, microbe_data=microbe_data, r=args.sampling_ratio))
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, 
                                  collate_fn=partial(my_collate_fn, microbe_data=microbe_data, r=args.sampling_ratio))
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                                 collate_fn=partial(my_collate_fn, microbe_data=microbe_data, r=args.sampling_ratio))

        drug_dim = len(folds[fold_idx][0][0]['drug_emb'])
        model = CL4MDA(microbe_dim=microbe_dim, drug_dim=drug_dim, 
                       hidden_dim1=args.hidden_dim1, hidden_dim2=args.hidden_dim2, emb_dim=args.emb_dim).to(args.device)
        
        if fold_idx == 0:
            print(model)
            print()

        initialize_weights(model)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = ContrastiveLoss(margin=args.margin, pos_weight=args.pos_weight, neg_weight=args.neg_weight)
    
        # Training
        trainer = Trainer(args, model, optimizer, criterion, n_microbe)
        trainer.train(train_loader, valid_loader, fold_idx, save_model_path)

        # Test
        metrics_method1, metrics_method2, auc, auprc = trainer.test(test_loader, fold_idx, save_model_path)
        fold_scores1 = add_fold_scores(fold_scores1, metrics_method1, auc, auprc)
        fold_scores2 = add_fold_scores(fold_scores2, metrics_method2, auc, auprc)

        print("Elapsed time: {0}".format(time.time() - time_start) + " [sec]")
        print()

        #break ### for debug

    print("Method 1:")
    calculate_print_metrics(fold_scores1)
    print("Method 2:")
    calculate_print_metrics(fold_scores2)


def save_experiment_config(args):
    config_path = os.path.join(args.logpath, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)


if __name__ == "__main__":
    set_seed(123)

    parser = argparse.ArgumentParser(description='<Contrastive learning for predictiong microbe-drug assocition>')
    
    parser.add_argument('--model',     default="CNN",   type=str,   help='contrastive learning encoding model')
    parser.add_argument('--device',    default="cuda:0",type=str,   help='GPU Device(s) used for training')
    parser.add_argument('--margin',    default=1,       type=float, help='Margins used in the contrastive training')
    parser.add_argument('--lr',        default=1e-3,    type=float, help='Learning rate')
    parser.add_argument('--epoch',     default=150,     type=int,   help='Training epcohs')
    parser.add_argument('--batch_size',default=32,      type=int,   help="Batch_size of training.")
    parser.add_argument('--n_splits',  default=5,       type=int,   help='Number of folds for cross validation')
    parser.add_argument('--hidden_dim1',   default=512,     type=int,   help='Hidden dimension 1')
    parser.add_argument('--hidden_dim2',   default=256,     type=int,   help='Hidden dimension 1')
    parser.add_argument('--emb_dim',   default=128,     type=int,   help='Embedding dimension')
    parser.add_argument('--sampling_ratio', default=None, type=float, help='Ratio of negative to positive samples in each batch (default: no sampling)')
    parser.add_argument('--pos_weight', default=1, type=float, help='Weight for positive data in loss function')
    parser.add_argument('--neg_weight', default=1, type=float, help='Weight for negative data in loss function')

    parser.add_argument('--dataset_name', default="", type=str)
    parser.add_argument('--path_drug_emb', default="data/dataset_MDAD_drug_molformer.pk", type=str)
    parser.add_argument('--path_microbe_emb', default="data/dataset_MDAD_microbe_evo.pk", type=str)
    parser.add_argument('--path_adj', default="../indata/MDAD/adj_new.txt", type=str)
    parser.add_argument('--logpath', default="../logs", type=str)
    
    args = parser.parse_args()
    
    save_experiment_config(args)
    print(args)
    print()
  
    main(args)


