#%%
import numpy as np
## Models
from nnvision.models.trained_models.v1_task_fine_tuned import v1_convnext_ensemble
## Functions
from functions   import *
from functions_2 import *
from functions_3 import *
from functions_4 import *
## To select the well predicted neurons 
from surroundmodulation.utils.misc import pickleread
## Import the config file
from config import experiments_config, results_config, execute_function
from surroundmodulation.utils.plot_utils import plot_img


#%%


## TODO : Why None ????
# for exp in experiments_confi:
#     name_function = exp[0]
#     params = exp[1]
#     result = execute_function(name_function, params)
#     print(result) 


for res in results_config :
    name_function = res[0]
    params = res[1]
    result = execute_function(name_function, params)

# %%
