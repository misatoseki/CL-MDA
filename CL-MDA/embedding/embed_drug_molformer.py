import os, pickle, time
import torch
import pandas as pd
import numpy as np

from torch import nn
from torch.utils.data import Dataset


class Drug_Dataset(Dataset):
    def __init__(self, drug_list_path):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

        drug_list = pd.read_excel(drug_list_path)
        smiles = list(drug_list['SMILES']) # Length N_drug
        inputs = tokenizer(smiles, padding=True, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
        self.drug_emb = outputs.pooler_output

    def __len__(self):
        return len(self.drug_emb)

    def __getitem__(self, idx):
        return self.drug_emb[idx]


def embed_drug_MolFormer(drug_list_path, output_path, output_name, device='cuda'):
    time_start = time.time()

    path = os.path.join(output_path, output_name)
    if os.path.isfile(path) == True:
        print(f"File is already available: {path}")
        print("Execusion stopped.")
    else:
        print("Start drug molecular embedding...")
        drug_dataset = Drug_Dataset(drug_list_path)
        with open(path, 'wb') as f:
            pickle.dump(drug_dataset[:], f)
        print(f"File was created: {path}")

    print("Elapsed time: {0}".format(time.time() - time_start) + " [sec]")


if __name__ == "__main__": 
    #embed_drug_MolFormer(drug_list_path = os.path.join('..', '..', 'indata2', 'MDAD', 'drugs_new.xlsx'),
    #                     output_path = os.path.join('..', 'data'),
    #                     output_name = 'dataset_MDAD_drug_molformer.pk')

    #embed_drug_MolFormer(drug_list_path = os.path.join('..', '..', 'indata', 'aBiofilm', 'drugs_new.xlsx'),
    #                     output_path = os.path.join('..', 'data'),
    #                     output_name = 'dataset_aBiofilm_drug_molformer.pk')

    embed_drug_MolFormer(drug_list_path = os.path.join('..', '..', 'indata', 'DrugVirus', 'drugs_new.xlsx'),
                         output_path = os.path.join('..', 'data'),
                         output_name = 'dataset_DrugVirus_drug_molformer.pk')


