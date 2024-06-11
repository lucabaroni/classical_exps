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
from surroundmodulation.utils.plot_utils import plot_img




'''
This is the file that will serve to configure the experiments you want to perform.


'''
''''''

# The experiments are explained with more details in the TODO rename_the_file.py file, every arguments are explained there
#
# List of available experiments :
#
# PRE-ANALYSES
# - get_all_grating_parameters : Find the preferred orientation, spatial frequency and phase for full fields gratings (this phase value will rarely be used since it is specific to the full field image)
#
# - get_preferred_position     : Find the receptive field center by fitting the neurons response to dots at different positions in the receptive field to a Gaussian model
#
# EXPERIMENTS 1 
# - size_tuning_experiment_all_phases           : Compute the size tuning curve of circular and annular grating stimuli and computes the Grating Summation Field (GSF), the surround extent, the Annular Minimum Response Field (AMRF) and the Suppression Index (SI)
#
# - contrast_response_experiment                : Compute the response to grating stimuli with different center and surround contrast
#
# - contrast_size_tuning_experiment_all_phases  : Compute the size tuning curves for different contrasts and computes the ratio of GSF at low contrast over GSF at high contrast
#
# EXPERIMENTS 2
# - orientation_tuning_experiment_all_phases        : Compute the orientation tuning curves of different stimuli : 1) Orientation of the center 2) Orientation of the surround with a fixed center orientation (fixed at -45°, 0°, +45° relative to the preferred position)
#
# - center_contrast_surround_suppression_experiment : Compute the response of different stimuli for different fixed center contrasts : center + iso-oriented surround, center + ortho-oriented surround, center with no surround
#
# EXPERIMENTS 3
# - black_white_preference_experiment   : Compute the Signal Noise Ratio (SNR) to black and white stimuli
#
# EXPERIMENTS 4
# - texture_noise_response_experiment   : Compute the response to texture images and matched noise images



#####    ####   #####    ####   ##   ##   ####
#    #  #    #  #    #  #    #  # # # #  #
#####   ######  #####   ######  #  #  #   ### 
#       #    #  #   #   #    #  #     #      #
#       #    #  #    #  #    #  #     #  #### 


## Working directory
workdir = "/project/mathys_code"
## Name of the HDF5 file 
h5_file = workdir + "/article1.h5"

## Select the model with all neurons
all_neurons_model = v1_convnext_ensemble

## Chose the indices of the neurons to work with
n = 458
neuron_ids = np.arange(n) ## Because 458 outputs in our model
corrs = pickleread(workdir + "/avg_corr.pkl") ## The correlation score of the neurons
neuron_ids = neuron_ids[corrs>0.75]

## Grating parameters to test
orientations = np.linspace(0, np.pi, 37)[:-1] 
spatial_frequencies = np.linspace(1, 7, 25) 
phases = np.linspace(0, 2*np.pi, 37)[:-1]

## Parameters for the dot stimulation TODO change
dot_size_in_pixels_gauss = 4
num_dots=200000
bs = 40
seed = 0

## Fixed image parameters
contrast = 1         
img_res = [93,93] 
pixel_min = -1.7876 # (the lowest value encountered in the data that served to train the model, serves as the black reference)
pixel_max =  2.1919 # (the highest [...] serves at white reference )
size = 2.67 

## Allow negative responses (because the response we show is the difference between the actual response and the response to a gray screen)
neg_val = True ## TODO not useful ???

## EXPERIMENTS 1 arguments
radii = np.logspace(-2,np.log10(2),40)
## For the contrast response experiment
center_contrasts = np.logspace(np.log10(0.06),np.log10(1),18) 
surround_contrasts = np.logspace(np.log10(0.06),np.log10(1),6)

## EXPERIMENTS 2 arguments
## For the orientation tuning experiment
ori_shifts = np.linspace(-np.pi,np.pi,9)
## For the ccss experiment
center_contrasts_ccss = np.array([0.06,0.12,0.25,0.5,1.0,1.5])
surround_contrast = contrast

## EXPERIMENTS 3 arguments
contrasts = np.logspace(np.log10(0.06),np.log10(1),5)  ## Same as in the article
dot_size_in_pixels = 5  #TODO fix the name in the function so not twice

## EXPERIMENTS 4 arguments
directory_imgs = workdir + '/shareStim_NN13'
target_res = img_res
num_samples = 15


#####  ##   ##  #####
#        # #    #    #
###       #     #####
#        # #    #
#####  ##   ##  #

## If overwrite is set to True, this will clean the results of the performed experiments before reperforming them
overwrite = False 

## (optional) Device to perform the experiments on (default will be gpu if available, cpu else)
device=None

# TODO DELETE
device = 'cpu'
neuron_ids = np.arange(458)

## Example Pipeline
experiments_config = [
    # ['get_all_grating_parameters', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'orientations':orientations, 'spatial_frequencies':spatial_frequencies, 'phases':phases, 'contrast':contrast, 'img_res':img_res, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'size':size}], 
    # ['get_preferred_position', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'dot_size_in_pixels':dot_size_in_pixels_gauss, 'contrast':contrast, 'num_dots':num_dots, 'img_res':img_res, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'bs':bs, 'seed':seed}],
    # ['size_tuning_experiment_all_phases', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'radii':radii, 'phases':phases, 'contrast':contrast, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'size':size, 'img_res':img_res, 'neg_val':neg_val}],
    # ['contrast_response_experiment', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'center_contrasts':center_contrasts, 'surround_contrasts':surround_contrasts, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'size':size, 'img_res':img_res, 'neg_val':neg_val}],
    # ['contrast_size_tuning_experiment_all_phases', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'phases':phases, 'contrasts':contrasts, 'radii':radii, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'size':size, 'img_res':img_res, 'neg_val':neg_val}],
    # ['orientation_tuning_experiment_all_phases', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'phases':phases, 'ori_shifts':ori_shifts, 'contrast':contrast, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'size':size, 'img_res':img_res}],
    # ['center_contrast_surround_suppression_experiment', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'center_contrasts_ccss':center_contrasts_ccss, 'surround_contrast':surround_contrast, 'phases':phases, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'size':size, 'img_res':img_res}],
    # ['black_white_preference_experiment', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'dot_size_in_pixels':dot_size_in_pixels, 'contrast':contrast, 'img_res':img_res, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device}],
    # ['texture_noise_response_experiment', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'directory_imgs':directory_imgs, 'overwrite':overwrite, 'contrast':contrast, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'num_samples':num_samples, 'img_res':img_res, 'device':device}]
    ]


#####   #####   ####  #   #  #   #######  ####
#    #  #      #      #   #  #      #    #
#####   ###     ###   #   #  #      #     ###     
#   #   #          #  #   #  #      #        #
#    #  #####  ####    ###   #####  #    ####

# The results are explained with more details in the TODO rename_the_file.py file, every arguments are explained there
#
# List of available analyses :
#
# FILTERING
#
# - filter_fitting_error       (1)     : This function shows the filtered neurons
# 
# - filter_no_supp_neurons     (2)
#
# - filter_low_supp_neurons    (3)
#
# - filter_SNR                 (4)
#
# EXPERIMENTS 1 

# - size_tuning_results_1            (1), (2), (3)
#
# - size_tuning_results_2            (1), (2)
#
# - contrast_response_results_1      (1), (2)
#
# - contrast_size_tuning_results_1   (1), (2)        
#
# EXPERIMENTS 2
#      
# - orientation_tuning_results_1        (1), (2)  
# 
# - orientation_tuning_results_2        (1), (2)  
#
# - ccss_results_1                      (1), (2) 
#
# - ccss_results_2                      (1), (2)  
#
# EXPERIMENTS 3
#
# - black_white_results_1               (4)
#
# EXPERIMENTS 4
#
# - texture_noise_response_results_1   
#
# - texture_noise_response_results_2
#
# - texture_noise_response_results_3


## Filtering parameters
fit_err_thresh = 0.2
supp_thresh = 0.1
SNR_thresh = 2

##size_tuning_results_2

sort_by_std = False
spread_to_plot = [0,15,50,85,100]

## contrast_size_tuning_results_1
shift_to_plot = spread_to_plot
low_contrast_id  =  0
high_contrast_id = -1

## Center Contrast Surround Suppression
high_center_contrast_id      = -3 ## The contrast of the center corresponding to the 'high' contrast
high_norm_center_contrast_id = high_center_contrast_id
low_center_contrast_id       = 1  ## The contrast of the center corresponding to the 'low' contrast
low_norm_center_contrast_id  = low_center_contrast_id  

## black_white_results_1
neuron_depths = pickleread(workdir + "/depth_info.pickle") 

## texture_noise_response_results
wanted_fam_order = ['60', '56', '13', '48', '71', '18', '327', '336', '402', '38', '23', '52', '99', '393', '30'] ## The order of the textures

## Example Pipeline
neuron_id = 0

results_config = [
    # ['plot_size_tuning_curve', {'h5_file':h5_file, 'neuron_id':neuron_id}],
    # ['size_tuning_results_1', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'fit_err_thresh':fit_err_thresh, 'supp_thresh':supp_thresh}],
    # ['size_tuning_results_2', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'fit_err_thresh':fit_err_thresh}],
    # ['contrast_response_results_1', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'fit_err_thresh':fit_err_thresh, 'sort_by_std':sort_by_std, 'spread_to_plot':spread_to_plot}],
    # ['contrast_size_tuning_results_1', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'fit_err_thresh':fit_err_thresh, 'shift_to_plot':shift_to_plot, 'low_contrast_id':low_contrast_id, 'high_contrast_id':high_contrast_id}],
    # ['orientation_tuning_results_1', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'fit_err_thresh':fit_err_thresh}],
    # ['orientation_tuning_results_2', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'fit_err_thresh':fit_err_thresh}],
    # ['ccss_results_1', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'fit_err_thresh':fit_err_thresh}],
    # ['ccss_results_2', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'contrast_id':high_center_contrast_id, 'norm_center_contrast_id':high_norm_center_contrast_id, 'fit_err_thresh':fit_err_thresh}],
    # ['ccss_results_2', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'contrast_id':low_center_contrast_id, 'norm_center_contrast_id':low_norm_center_contrast_id, 'fit_err_thresh':fit_err_thresh}],
    ['black_white_results_1', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'neuron_depths':neuron_depths, 'SNR_thresh':SNR_thresh}],
    # ['texture_noise_response_results_1',  {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'wanted_fam_order':wanted_fam_order}],
    # ['texture_noise_response_results_2',  {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'wanted_fam_order':wanted_fam_order}],
    # ['texture_noise_response_results_3',  {'h5_file':h5_file, 'neuron_ids':neuron_ids}]
]


texture_noise_response_results_3
def execute_function(func_name, params):
    # Get the function object by name
    func = globals().get(func_name)
    
    # Check if the function exists
    if func is None or not callable(func):
        raise ValueError(f"Function '{func_name}' not found or not callable.")
    
    # Call the function with the parameters
    func(**params)


# %%
## TODO DELETE BELOW


import numpy as np
## Models
from nnvision.models.trained_models.v1_task_fine_tuned import v1_convnext_ensemble
## Functions
from functions   import *
from functions_2 import *
from functions_3 import *
## To select the well predicted neurons 
from surroundmodulation.utils.misc import pickleread
## Import the config file

h5_file= '/project/mathys_code/article1.h5'
overwrite = False



for res in results_config :
    name_function = res[0]
    params = res[1]
    result = execute_function(name_function, params)


# %%
# print(len(neuron_ids))


# for neuron_id in neuron_ids[:10] :
#     plot_size_tuning_curve(h5_file=h5_file, neuron_id=neuron_id)


# %%

for exp in experiments_config :
    name_function = exp[0]
    params = exp[1]
    result = execute_function(name_function, params)



# %%

weights = np.ones(corrs.shape) / len(corrs)
plt.hist(corrs, bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], edgecolor='k', weights=weights)
plt.plot([0.75, 0.75], [0, 0.4], label = "Threshold")
plt.legend()
plt.ylabel("Proportion of Cells")
plt.xlabel("Correlation")