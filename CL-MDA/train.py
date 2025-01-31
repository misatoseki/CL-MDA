# train the constrastive learning 

import time, os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class Trainer:
    def __init__(self, args, model, optimizer, criterion, n_microbe):
        self.args = args
        self.device = args.device
        self.margin = args.margin
        self.epoch = args.epoch
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.logpath = args.logpath
        self.batch_size = args.batch_size
        self.n_microbe = n_microbe
        self.r = args.sampling_ratio

    def train(self, train_loader, valid_loader, fold_idx, save_model_path):
        start_train = time.time()
        self.model.train()

        current_best_valid_loss = 100
        patience = 300
        count = 0
        for ep in range(self.epoch):
            if count == patience:
                break

            epoch_loss = 0
            for batch in train_loader:
                input_drug = batch['drug_emb'].to(self.device)
                input_microbe = batch['microbe_emb'].to(self.device)
                labels = batch['association'].to(self.device)
    
                output_drug, output_microbe = self.model(input_drug, input_microbe)
                loss = self.criterion(output_drug, output_microbe, labels)
                epoch_loss += loss.item()
    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            valid_loss = self.evaluate(valid_loader)[0]
            if valid_loss < current_best_valid_loss:
                current_best_valid_loss = valid_loss
                print(f"- Epoch {ep}, Best validation loss: {current_best_valid_loss:.4f}")
                torch.save(self.model.state_dict(), save_model_path)
                count = 0
            else:
                count += 1

            print(f"Epoch-{ep}, Train loss={epoch_loss/len(train_loader):.4f}, Valid loss={valid_loss:.4f}")

        used_train = time.time() - start_train
        print(" @ used training time:", round(used_train,2), "[sec]")

    def test(self, test_loader, fold_idx, save_model_path):
        self.model.load_state_dict(torch.load(save_model_path))
        self.model.eval()
        loss, metrics1, metrics2, auc, auprc, dist_label = self.evaluate(test_loader)
        print('Test results -- Method 1')
        print(f'Accuracy: {metrics1["accuracy"]:.2f}%, Precision: {metrics1["precision"]:.2f}%, '
              f'Recall: {metrics1["recall"]:.2f}%, F1: {metrics1["f1"]:.2f}%, '
              f'AUC: {auc:.4f}, AUPRC: {auprc:.4f}')

        print('Test results -- Method 2')
        print(f'Accuracy: {metrics2["accuracy"]:.2f}%, Precision: {metrics2["precision"]:.2f}%, '
              f'Recall: {metrics2["recall"]:.2f}%, F1: {metrics2["f1"]:.2f}%, '
              f'AUC: {auc:.4f}, AUPRC: {auprc:.4f}')

        dist_file = os.path.join(self.logpath, f"dist_label_fold{fold_idx + 1}.txt")
        dist_label.to_csv(dist_file, index=False)

        return metrics1, metrics2, auc, auprc

    def confusion_matrix(self, predicted, labels, conf_matrix):
        tp = ((predicted == 1) & (labels == 1)).sum().item()
        fp = ((predicted == 1) & (labels == 0)).sum().item()
        tn = ((predicted == 0) & (labels == 0)).sum().item()
        fn = ((predicted == 0) & (labels == 1)).sum().item()

        conf_matrix['total_tp'] += tp
        conf_matrix['total_fp'] += fp
        conf_matrix['total_tn'] += tn
        conf_matrix['total_fn'] += fn
       
        return conf_matrix

    def calculate_metrics(self, conf_matrix):
        tp = conf_matrix['total_tp']
        fp = conf_matrix['total_fp']
        tn = conf_matrix['total_tn']
        fn = conf_matrix['total_fn']
 
        accuracy  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {'accuracy': 100 * accuracy, 
                'precision': 100 * precision, 
                'recall': 100 * recall, 
                'f1': 100 * f1}

    def evaluate(self, loader):
        self.model.eval()
        conf_matrix_method1 = {'total_tp': 0, 'total_fp':0, 'total_tn':0, 'total_fn':0}
        conf_matrix_method2 = {'total_tp': 0, 'total_fp':0, 'total_tn':0, 'total_fn':0}

        total_loss = 0
        all_labels, all_outputs, pred_dist, label_pred_method1, label_pred_method2 = [], [], [], [], []

        with torch.no_grad():
            for batch in loader:
                input_drug = batch['drug_emb'].to(self.device)
                input_microbe = batch['microbe_emb'].to(self.device)
                labels = batch['association'].to(self.device)
 
                output_drug, output_microbe = self.model(input_drug, input_microbe)
                loss = self.criterion(output_drug, output_microbe, labels)
                total_loss += loss.item()

                diff = output_drug - output_microbe
                dist_sq = torch.sum(torch.pow(diff, 2), 1)
                dist = torch.sqrt(dist_sq)

                # Method 1: Predict positive if distance is min AND distance < margin
                reshaped_dist = dist.reshape(-1, self.n_microbe)
                min_idx = torch.argmin(reshaped_dist, dim=1)
                reshaped_predicted = torch.zeros_like(reshaped_dist)
                reshaped_predicted[torch.arange(reshaped_dist.size(0)), min_idx] = 1
                reshaped_predicted[reshaped_dist >= self.margin] = 0
                predicted1 = reshaped_predicted.reshape(-1)

                # Method 2: Predict positive if distance < margin
                predicted2 = (dist < self.margin).float()

                conf_matrix_method1 = self.confusion_matrix(predicted1, labels, conf_matrix_method1)
                conf_matrix_method2 = self.confusion_matrix(predicted2, labels, conf_matrix_method2)

                output = 1 - dist
                all_labels.append(labels.cpu())
                all_outputs.append(output.cpu())

                pred_dist.append(dist.to("cpu").detach().numpy())
                label_pred_method1.append(predicted1.to("cpu").detach().numpy())
                label_pred_method2.append(predicted2.to("cpu").detach().numpy())

        eval_loss = total_loss/len(loader)
        metrics_method1 = self.calculate_metrics(conf_matrix_method1)
        metrics_method2 = self.calculate_metrics(conf_matrix_method2)

        all_labels = torch.cat(all_labels).numpy()
        all_outputs = torch.cat(all_outputs).detach().numpy()
        auc = roc_auc_score(all_labels, all_outputs)
        auprc = average_precision_score(all_labels, all_outputs)

        dist_label = pd.DataFrame({'predicted distance': np.concatenate(pred_dist, axis=0),
                                   'predicted label method 1': np.concatenate(label_pred_method1, axis=0),
                                   'predicted label method 2': np.concatenate(label_pred_method2, axis=0),
                                   'true label': all_labels})

        return eval_loss, metrics_method1, metrics_method2, auc, auprc, dist_label


