import os, pickle, time, gzip
import torch
import pandas as pd
import numpy as np

from Bio import SeqIO
from torch import nn
from torch.utils.data import Dataset


class CustomEmbedding(nn.Module):
  def unembed(self, u):
    return u


class Bacteria_Dataset(Dataset):
    def __init__(self, microbe_list_path, input_genome_path, output_path, device):
        super().__init__()
        self.device = device
        self.input_genome_path = input_genome_path
        self.output_path = output_path
        self.batch_size = 1

        microbe_list = pd.read_excel(microbe_list_path)
        genomeid = list(microbe_list[microbe_list.Genome_ID.notna()]['Genome_ID'])
        self.evo_model = None
        self.microbe_emb = self.evo_embedding(microbe_list, genomeid) # [N_microbe, x, x]

    def __len__(self):
        return len(self.microbe_emb)

    def __getitem__(self, idx):
        return torch.from_numpy(self.microbe_emb)[idx]

    def get_evo_model(self, model_name='evo-1-8k-base'):
        if self.evo_model is None:
            from evo import Evo
            print("Loading Evo model...")
            self.evo_model = Evo(model_name) 
            print("Completed loading Evo model...")
            self.model, self.tokenizer = self.evo_model.model, self.evo_model.tokenizer
            self.model.unembed = CustomEmbedding()
            self.model.to(self.device)
            self.model.eval()
        else:
            print("Evo model already loaded.")
        return self.evo_model, self.model, self.tokenizer

    def prepare_batch(self, seqs, prepend_bos):
        seq_lengths = [ len(seq) for seq in seqs ]
        max_seq_length = max(seq_lengths)
    
        input_ids = []
        for seq in seqs:
            padding = [self.tokenizer.pad_id] * (max_seq_length - len(seq))
            input_ids.append(
                torch.tensor(
                    ([self.tokenizer.eod_id] * int(prepend_bos)) + self.tokenizer.tokenize(seq) + padding,
                    dtype=torch.long,
                ).to(self.device).unsqueeze(0)
            )   
        input_ids = torch.cat(input_ids, dim=0)
 
        return input_ids, seq_lengths

    def embed_sequences(self, seqs):
        input_ids, seq_lengths = self.prepare_batch(seqs, prepend_bos=False)
        assert(len(seq_lengths) == input_ids.shape[0])
    
        with torch.inference_mode():
            embeds, _ = self.model(input_ids) # (batch, length, vocab)
    
        embeds_np = embeds.float().cpu().numpy()
    
        return [np.mean(embeds_np[idx][:seq_lengths[idx]], axis=0) for idx in range(len(seq_lengths))]

    def evo_embedding(self, microbe_list, genomeid):
        microbe_emb = []
        max_seq_length = 8000

        for i, genid in enumerate(genomeid):
            print(f"Bacterial genome {i+1}/{len(genomeid)}: {genid}...")
            path_emb = os.path.join(self.output_path, 'tmp', f"emb_{genid}.csv")
            if os.path.isfile(path_emb) == False: 
                self.evo_model, self.model, self.tokenizer = self.get_evo_model(model_name='evo-1-8k-base')

                base_path = os.path.join(self.input_genome_path, genid)
                file_name = [x for x in os.listdir(base_path) if x.endswith('fna.gz')][0]
                file_path = os.path.join(base_path, file_name)
     
                with gzip.open(file_path, 'rt') as handle:
                    full_seqs = [ str(record.seq) for record in SeqIO.parse(handle, 'fasta') ]
                    seqs = []
                    for seq in full_seqs:
                        for start in range(0, len(seq), max_seq_length):
                            seqs.append(seq[start:start + max_seq_length])
     
                embeddings = []
                for j in range(0, len(seqs), self.batch_size):
                    print(f"    Processing sequence {j+1}/{len(seqs)}...")
                    batch_seqs = seqs[j:j+self.batch_size]
                    batch_embeds = self.embed_sequences(batch_seqs)
                    embeddings.extend(batch_embeds)
    
                    del batch_embeds
                    torch.cuda.empty_cache()
    
                embeddings_mean = np.mean(np.array(embeddings), axis=0)
                embed_df = pd.DataFrame({genid: embeddings_mean.tolist()})
                embed_df.to_csv(path_emb, index=False)

            else:
                print(f"    {genid} is already available.")
                embed_df = pd.read_csv(path_emb)
   
            microbe_emb.append(embed_df.iloc[:, -1].to_list())
            #break ### For debug
 
        return np.array(microbe_emb) # [N_microbe, embed_dim]


def embed_microbe_evo(microbe_list_path, input_genome_path, output_path, output_name, device='cuda'):
    time_start = time.time()

    path = os.path.join(output_path, output_name)
    if os.path.isfile(path) == True:
        print(f"File is already available: {path}")
        print("Execusion stopped.")
    else:
        print("Start microbial genome embedding...")
        micro_dataset = Bacteria_Dataset(microbe_list_path, input_genome_path, output_path, device)
        with open(path, 'wb') as f:
            pickle.dump(micro_dataset[:], f)
        print(f"File was created: {path}")

    print("Elapsed time: {0}".format(time.time() - time_start) + " [sec]")


if __name__ == "__main__": 
    #embed_microbe_evo(microbe_list_path = os.path.join('..', '..', 'indata2', 'MDAD', 'microbes_new.xlsx'),
    #                  input_genome_path = os.path.join('..', '..', 'indata2', 'genome'),
    #                  output_path = os.path.join('..', 'data'),
    #                  output_name = 'dataset_MDAD_microbe_evo.pk')

    #embed_microbe_evo(microbe_list_path = os.path.join('..', '..', 'indata', 'aBiofilm', 'microbes_new.xlsx'),
    #                  input_genome_path = os.path.join('..', '..', 'indata', 'genome'),
    #                  output_path = os.path.join('..', 'data'),
    #                  output_name = 'dataset_aBiofilm_microbe_evo.pk')

    embed_microbe_evo(microbe_list_path = os.path.join('..', '..', 'indata', 'DrugVirus', 'microbes_new.xlsx'),
                      input_genome_path = os.path.join('..', '..', 'indata', 'genome'),
                      output_path = os.path.join('..', 'data'),
                      output_name = 'dataset_DrugVirus_microbe_evo.pk')


