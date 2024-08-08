##################################################################################################
#####    THIS FILE AIMS TO EXECUTE THE EXPERIMENTS AND ANALYSES CHOSEN IN THE CONFIG FILE    #####
#####              -----------------------------------------------------------               #####
##################################################################################################


#%%
############################
##### PART 0 : Imports #####
############################


## Import the config objects
from config import experiments_config, analyses_config, execute_function


########################
##### PART I : Run #####
########################

## Perform the experiments
for exp in experiments_config :
    name_function = exp[0]
    params = exp[1]
    result = execute_function(name_function, params)

## Perform the analyses
for res in analyses_config :
    name_function = res[0]
    params = res[1]
    result = execute_function(name_function, params)

# # %%
# import h5py

# import h5py
# import numpy as np

# def stack_neuron_datasets(file_name):
#     with h5py.File(file_name, 'r') as f:
#         neuron_datasets = []
        
#         def collect_neuron_datasets(name, obj):
#             if isinstance(obj, h5py.Dataset) and name.startswith('full_field_params/neuron_'):
#                 neuron_datasets.append(f[name][:])
        
#         f.visititems(collect_neuron_datasets)
        
#         # Stack all collected datasets
#         if neuron_datasets:
#             stacked_data = np.vstack(neuron_datasets)
#             return stacked_data
#         else:
#             return None

# # Replace 'your_file.h5' with the path to your HDF5 file
# stacked_data = stack_neuron_datasets('/project/classical_exps/run_pipeline/results_convnext_model.h5')

# if stacked_data is not None:
#     print("Stacked data shape:", stacked_data.shape)
# else:
#     print("No datasets found.")
# # %%
# stacked_data
# # %%
#%%
import h5py

h5 = '/project/run_pipeline/results_convnext_model.h5'

import h5py
import numpy as np
import h5py

def list_datasets(file_name):
    with h5py.File(file_name, 'r') as f:
        def print_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(name)
        f.visititems(print_datasets)

# Replace 'your_file.h5' with the path to your HDF5 file
list_datasets(h5)

# %%
data = {i: {} for i in range(458)}
with h5py.File(h5, 'r') as f:
    for i in range(458):
        dataset_name = f'full_field_params/neuron_{i}'
        if dataset_name in f:
            data[i]['preferred_ori'] = f[dataset_name][:][0]
        dataset_name = f'preferred_pos/neuron_{i}'
        if dataset_name in f:
            data[i]['preferred_pos'] = f[dataset_name][:]