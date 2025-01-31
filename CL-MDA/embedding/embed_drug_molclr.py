import os, sys, yaml, time, pickle
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False


class Drug_Dataset(Dataset):
    def __init__(self, drug_list_path, device):
        super().__init__()
        self.device = device

        sys.path.append(os.path.abspath(os.path.join('..', '..', 'MolCLR')))
        from dataset.dataset_test import MolTestDataset
        from models.ginet_molclr import GINet

        self.config = yaml.load(open("../../MolCLR/config.yaml", "r"), Loader=yaml.FullLoader)
        self.config['load_model'] = 'pretrained_gin'

        dataset = MolTestDataset(drug_list_path)
        #data_loader = DataLoader(dataset, drop_last=False)

        model = GINet(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)

        drug_emb = []
        with torch.no_grad():
            model.eval()
            for i in range(len(dataset)):
                tmp, _ = model(dataset[i].to(self.device))
                drug_emb.append(tmp.squeeze(0).cpu())

        self.drug_emb = torch.from_numpy(np.array(drug_emb))

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('../../MolCLR/ckpt', self.config['load_model'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=self.device)
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def __len__(self):
        return len(self.drug_emb)

    def __getitem__(self, idx):
        return self.drug_emb[idx]


def embed_drug_MolCLR(drug_list_path, output_path, output_name, device='cuda'):
    time_start = time.time()

    path = os.path.join(output_path, output_name)
    if os.path.isfile(path) == True:
        print(f"File is already available: {path}")
        print("Execusion stopped.")
    else:
        print("Start drug molecular embedding...")
        drug_dataset = Drug_Dataset(drug_list_path, device)
        with open(path, 'wb') as f:
            pickle.dump(drug_dataset[:], f)
        print(f"File was created: {path}")

    print("Elapsed time: {0}".format(time.time() - time_start) + " [sec]")


if __name__ == "__main__":
    embed_drug_MolCLR(drug_list_path = os.path.join('..', '..', 'indata2', 'MDAD', 'drugs_new.xlsx'),
                      output_path = os.path.join('..', 'data'),
                      output_name = 'dataset_MDAD_drug_molclr.pk')


