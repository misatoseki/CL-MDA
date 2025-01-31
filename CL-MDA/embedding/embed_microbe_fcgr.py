import os, pickle, time, gzip
import io, collections, math
import torch
import pandas as pd
import numpy as np

from Bio import SeqIO
from torch import nn
from torch.utils.data import Dataset


class Bacteria_Dataset(Dataset):
    def __init__(self, microbe_list_path, input_genome_path, kmer):
        super().__init__()
        self.kmer = kmer
        self.input_genome_path = input_genome_path
 
        microbe_list = pd.read_excel(microbe_list_path)
        genomeid = list(microbe_list[microbe_list.Genome_ID.notna()]['Genome_ID'])
        self.microbe_emb = self.fcgr(microbe_list, genomeid) # [N_microbe, x, x] (if 6-mer, x=64)

    def __len__(self):
        return len(self.microbe_emb)

    def __getitem__(self, idx):
        return torch.from_numpy(self.microbe_emb)[idx]

    def fcgr(self, microbe_list, genomeid):
        microbe_emb = []
        for i, genid in enumerate(genomeid):
            base_path = os.path.join(self.input_genome_path, genid)
            file_name = [x for x in os.listdir(base_path) if x.endswith('fna.gz')][0]
            file_path = os.path.join(base_path, file_name)
            with gzip.open(file_path, 'rt') as handle:
                fc = collections.defaultdict(int)
                N_base = 0   # Count total length of sequences
                N_seq = 0    # Count number of sequences
                for record in SeqIO.parse(handle, 'fasta'):
                    fc_tmp = self.count_kmers(record.seq, self.kmer)
                    for key in list(fc_tmp.keys()):
                        fc[key] += fc_tmp[key]
                    N_base += len(record.seq)
                    N_seq += 1

                f_prob = self.probabilities(N_base, N_seq, fc, self.kmer)
                chaos_k = self.chaos_game_representation(f_prob, self.kmer)
                microbe_emb.append(chaos_k)

        return np.array(microbe_emb)

    def count_kmers(self, sequence, k):
        d = collections.defaultdict(int)
        for i in range(len(sequence)-(k-1)):
            d[sequence[i:i+k]] += 1
    
        for key in list(d.keys()):
            if 'N' in key:
                del d[key]
        return d
 
    def probabilities(self, N, N_seq, kmer_count, k):
        probabilities = collections.defaultdict(float)
        for key, value in kmer_count.items():
            probabilities[key] = float(value) / (N - N_seq * (k + 1))
        return probabilities
 
    def chaos_game_representation(self, probabilities, k):
        array_size = int(math.sqrt(4**k))
        chaos = []
        for i in range(array_size):
            chaos.append([0]*array_size)
    
        maxx, maxy = array_size, array_size
        posx, posy = 1, 1
    
        for key, value in probabilities.items():
            for char in key:
                if char == 'T' or char == 't':
                    posx += maxx / 2
                elif char == 'C' or char == 'c':
                    posy += maxy / 2
                elif char == 'G' or char == 'g':
                    posx += maxx / 2
                    posy += maxy / 2
                maxx = maxx / 2
                maxy /= 2
    
            chaos[int(posy)-1][int(posx)-1] = value
            maxx = array_size
            maxy = array_size
            posx = 1
            posy = 1
    
        return chaos


def embed_microbe_evo(microbe_list_path, input_genome_path, output_path, output_name):
    time_start = time.time()

    path = os.path.join(output_path, output_name)
    if os.path.isfile(path) == True:
        print(f"File is already available: {path}")
        print("Execusion stopped.")
    else:
        print("Start microbial genome embedding...")
        micro_dataset = Bacteria_Dataset(microbe_list_path, input_genome_path, kmer=6)
        with open(path, 'wb') as f:
            pickle.dump(micro_dataset[:], f)
        print(f"File was created: {path}")

    print("Elapsed time: {0}".format(time.time() - time_start) + " [sec]")


if __name__ == "__main__": 
    embed_microbe_evo(microbe_list_path = os.path.join('..', '..', 'indata2', 'MDAD', 'microbes_new.xlsx'),
                      input_genome_path = os.path.join('..', '..', 'indata', 'genome'),
                      output_path = os.path.join('..', 'data'),
                      output_name = 'dataset_MDAD_microbe_fcgr.pk')


