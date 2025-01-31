
import torch
from torch import nn


class CL4MDA(nn.Module):
    def __init__(self, microbe_dim=4096, drug_dim=256, hidden_dim1=512, hidden_dim2=256, emb_dim=256):
        super().__init__()
        
        # Drug - MLP
        self.process_drug = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim1),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim2),
            nn.Linear(hidden_dim2, emb_dim),
        )

        # Microbe - MLP
        self.process_microbe = nn.Sequential(
            nn.Linear(microbe_dim, hidden_dim1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim1),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim2),
            nn.Linear(hidden_dim2, emb_dim),
        )

    def forward(self, x_drug, x_microbe):
        x_microbe = x_microbe.float()
        x_drug = x_drug.float()

        output_microbe = self.process_microbe(x_microbe)
        output_drug = self.process_drug(x_drug)

        return output_drug, output_microbe


