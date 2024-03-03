#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralpredictors.layers.readouts import FullGaussian2d
from nnvision.models import se_core_full_gauss_readout
from nnfabrik.builder import get_data



class SingleCellModel(nn.Module):
    def __init__(self, model, idx):
        super().__init__()
        self.model = model
        self.idx = idx
    
    def forward(self, x):
        return self.model(x)[:, self.idx].squeeze()

