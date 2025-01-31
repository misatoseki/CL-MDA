import os, pickle, time, sys
import torch
import pandas as pd
import numpy as np

from torch import nn
from torch.utils.data import Dataset


class Drug_Dataset(Dataset):
    def __init__(self, selfies_dataset, model_file):
        super().__init__()
        from pandarallel import pandarallel
        from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
        
        df = pd.read_csv(selfies_dataset) 
        
        model_name = model_file # path of the pre-trained model
        config = RobertaConfig.from_pretrained(model_name)
        config.output_hidden_states = True
        self.tokenizer = RobertaTokenizer.from_pretrained("../../SELFormer/data/RobertaFastTokenizer")
        self.model = RobertaModel.from_pretrained(model_name, config=config)
       
        pandarallel.initialize(nb_workers=1,progress_bar=True) # number of threads
        tmp = df.selfies.parallel_apply(self.get_sequence_embeddings)

        out = []
        for i in range(len(tmp)):
            out.append(np.array(tmp[i]))
            
        self.drug_emb = torch.from_numpy(np.array(out))

    def get_sequence_embeddings(self, selfies):
        token = torch.tensor([self.tokenizer.encode(selfies, add_special_tokens=True, max_length=512, padding=True, truncation=True)])
        output = self.model(token)
    
        sequence_out = output[0]
        return torch.mean(sequence_out[0], dim=0).tolist()
 
    def __len__(self):
        return len(self.drug_emb)

    def __getitem__(self, idx):
        return self.drug_emb[idx]


def embed_drug_SELFormer(selfies_dataset, model_file, output_path, output_name, device='cuda'):
    time_start = time.time()

    path = os.path.join(output_path, output_name)
    if os.path.isfile(path) == True:
        print(f"File is already available: {path}")
        print("Execusion stopped.")
    else:
        print("Start drug molecular embedding...")
        drug_dataset = Drug_Dataset(selfies_dataset, model_file)
        with open(path, 'wb') as f:
            pickle.dump(drug_dataset[:], f)
        print(f"File was created: {path}")

    print("Elapsed time: {0}".format(time.time() - time_start) + " [sec]")


if __name__ == "__main__": 
    # Prepare SMILES csv
    drug_list_path = os.path.join('..', '..', 'indata2', 'MDAD', 'drugs_new.xlsx')
    smiles_dataset = os.path.join('..', '..', 'indata2', 'MDAD', 'drugs_SMILES.csv')
    #if os.path.isfile(smiles_dataset) == False:
    drug_list = pd.read_excel(drug_list_path)
    drug_list.rename(columns={'SMILES': 'canonical_smiles'}).to_csv(smiles_dataset, sep="\t", index=False)

    # Prepare SELFIES csv
    selfies_dataset = os.path.join('..', '..', 'indata2', 'MDAD', 'drugs_SELFIES.csv')
    #if os.path.isfile(selfies_dataset) == False:
    sys.path.append(os.path.abspath(os.path.join('..', '..', 'SELFormer')))
    from prepare_pretraining_data import prepare_data

    prepare_data(path=smiles_dataset, save_to=selfies_dataset)
    print("SELFIES representation file is ready.")

    # Execute SELFormer
    embed_drug_SELFormer(selfies_dataset = selfies_dataset,
                         model_file = os.path.join('..', '..', 'SELFormer', 'pretrained_models', 'SELFormer'),
                         output_path = os.path.join('..', 'data'),
                         output_name = 'dataset_MDAD_drug_selformer.pk')


