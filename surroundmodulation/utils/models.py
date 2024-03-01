#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralpredictors.layers.readouts import FullGaussian2d
from nnvision.models import se_core_full_gauss_readout
from nnfabrik.builder import get_data

dataset_fn, dataset_config = ('nnvision.datasets.monkey_loaders.monkey_static_loader_combined',
        {'dataset': 'CSRF19_V1',
        'neuronal_data_files': ['/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3631896544452.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3632669014376.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3632932714885.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3633364677437.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3634055946316.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3634142311627.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3634658447291.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3634744023164.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3635178040531.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3635949043110.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3636034866307.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3636552742293.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3637161140869.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3637248451650.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3637333931598.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3637760318484.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3637851724731.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638367026975.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638456653849.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638885582960.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638373332053.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638541006102.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638802601378.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638973674012.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3639060843972.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3639406161189.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3640011636703.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3639664527524.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3639492658943.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3639749909659.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3640095265572.pickle',
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3631807112901.pickle'],
        'image_cache_path': '/data/monkey/toliaslab/CSRF19_V1/images/',
        'crop': 70,
        'subsample': 1,
        'seed': 1000,
        'time_bins_sum': 12,
        'batch_size': 128})

dataloaders = get_data(dataset_fn, dataset_config)


# def get_full_gauss_monkey_model(model_config, dataloaders, seed=None):
#     seed = seed if seed is not None else np.random.randint(0, 100)
#     model = se_core_full_gauss_readout(
#         dataloaders=dataloaders, seed=seed, **model_config
#     )
#     return model

# class keyless_full_gaussian_monkey_model(nn.Module):
#     def __init__(self, model_initial):
#         super().__init__()
#         model_initial.eval()
#         keys = list(model_initial.readout.keys())
#         att = dict(**model_initial.readout[keys[0]].__dict__)
#         # print(att)
#         att['bias'] = True
#         att['outdims'] = 458

#         start = 0
#         readout = FullGaussian2d(**att)
#         for k in keys:
#             r = model_initial.readout[k]
#             add = r.outdims
#             end = start + add
#             readout.features.data[:,:,:, start:end] = r.features.data
#             readout.grid.data[:,start:end,:, :] = r.grid.data
#             readout.mu.data[:,start:end,:, :] = r.mu.data
#             readout.bias.data[start:end] = r.bias.data 
#             start = start + add
#         self.core = model_initial.core   
#         self.readout = readout
#         self.eval()
        
#     def forward(self, x):
#         x = self.readout(self.core(x))
#         return torch.nn.functional.elu(x) + 1 

# class keyless_ensamble_full_gaussian_monkey_model(nn.Module):
#     def __init__(self, models_paths, model_config, seed=None, dataloaders=dataloaders):
#         super().__init__()
#         models =[]
#         for path in models_paths:
#             model = get_full_gauss_monkey_model(model_config, dataloaders, seed=None)
#             model.load_state_dict(torch.load(path))
#             models.append(model)
#         self.models = nn.ModuleList(keyless_full_gaussian_monkey_model(model) for model in models)
#         self.eval()
        
#     def forward(self, x):
#         x = torch.stack([model(x) for model in self.models]).mean(dim=0)
#         return x


class SingleCellModel(nn.Module):
    def __init__(self, model, idx):
        super().__init__()
        self.model = model
        self.idx = idx
    
    def forward(self, x):
        return self.model(x)[:, self.idx].squeeze()


class EnsambleModel(nn.Module):
    def __init__(self, models_paths, model_config, dataloaders=dataloaders,  seed=None):
        seed = seed if seed is not None else np.random.randint(0, 100)
        super().__init__()
        models =[]
        for path in models_paths:
            model = se_core_full_gauss_readout(dataloaders, seed, **model_config,)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.load_state_dict(torch.load(path, map_location=device))
            models.append(model)
        self.models = nn.ModuleList([model for model in models])
        self.eval()
        
    def forward(self, x, data_key=None):
        x = torch.stack([model(x, data_key=None) for model in self.models]).mean(dim=0)
        return x


# %%
