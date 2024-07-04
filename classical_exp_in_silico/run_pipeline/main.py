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
