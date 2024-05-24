#%%
import numpy as np
## Models
from nnvision.models.trained_models.v1_task_fine_tuned import v1_convnext_ensemble
## Functions
from functions import *


#%%

############################################
############## FIRST ANALYSIS ##############
############################################
# COMPARE MODEL TO THE FOLLOWING ARTICLE : #
# https://doi.org/10.1152/jn.00692.2001    #
############################################

#%%

##########################
## CHOSE THE PARAMETERS ##
##########################


#TODO change path so it is available for everybody

# 1) Chose the arguments 

## Name of the HDF5 file 
h5_file = "/project/mathys_code/test2.h5"

## Select the model of all neurons
all_neurons_model = v1_convnext_ensemble

## Chose the number of neurons to work with
n = 458
neuron_ids = np.arange(n) ## Because 458 outputs in our model

## Parameters to test
orientations = np.linspace(0, np.pi, 37)[:-1] 
spatial_frequencies = np.linspace(1, 7, 25) 
phases = np.linspace(0, 2*np.pi, 37)[:-1]
radii = np.logspace(-2,np.log10(2),40)

## Parameters for the dot stimulation
dot_size_in_pixels = 4
num_dots=200000
bs = 40
seed = 0

## Fixed image parameters
contrast = 1         
img_res = [100,100] 
pixel_min = -1.7876 # (the lowest value encountered in the data that served to train the model, serves as the black reference)
pixel_max =  2.1919
size = 2.67 #2.35

## Allow negative responses (because the response we show is the difference between the actual response and the response to a gray screen)
neg_val = True

## Analysis1 arguments
radii = radii

## Analysis2 arguments
center_contrasts = np.logspace(np.log10(0.06),np.log10(1),18) #np.logspace(-2,np.log10(1),18) 
surround_contrasts = np.logspace(np.log10(0.06),np.log10(1),6) #np.logspace(-2,np.log10(1),6) 

## Analysis3 arguments
contrasts = np.logspace(np.log10(0.06),np.log10(1),5)  ## Same as in the article

#%%
################################
## GET THE NEURONS PREFERENCE ##
################################

## Find the neuron's preferred parameters

## Setting overwrite to true will empty any chosen experiment results and reperform it
overwrite = True

## (Optional) select the device
device = None

## Get the preferred full field gratings parameters for the neuron (orientation, spatial frequency, full field phase)
'''
get_all_grating_parameters_fast(h5_file=h5_file, all_neurons_model=all_neurons_model, neuron_ids=neuron_ids, overwrite=overwrite, orientations = orientations, spatial_frequencies = spatial_frequencies, phases = phases, contrast = contrast, img_res = img_res, pixel_min = pixel_min, pixel_max = pixel_max, device = device, size = size )
'''

## Find the preferred stimulus position by finding the center of the gaussian model fitted to the excitatory pixels for every neuron
'''
get_preferred_position(h5_file=h5_file, all_neurons_model=all_neurons_model, neuron_ids=neuron_ids, overwrite=overwrite, dot_size_in_pixels = dot_size_in_pixels, num_dots = num_dots, contrast = contrast, img_res = img_res, pixel_min = pixel_min, pixel_max =  pixel_max, device = device, bs = bs, seed = seed)
'''

#%%
#############################
## PERFORM THE EXPERIMENTS ##
#############################

## Setting overwrite to true will empty any chosen experiment results and reperform it
overwrite = False

## (Optional) select the device
device = None

## Perform size tuning experiment
'''
size_tuning_experiment_all_phases(h5_file=h5_file, all_neurons_model=all_neurons_model, neuron_ids=neuron_ids, overwrite=overwrite, radii=radii, phases=phases, contrast=contrast, pixel_min=pixel_min, pixel_max=pixel_max, device=device, size=size, img_res=img_res, neg_val=neg_val)
'''

## Perform contrast response experiment (15min)
'''
contrast_response_experiment(h5_file=h5_file, all_neurons_model=all_neurons_model, neuron_ids=neuron_ids, overwrite=overwrite, center_contrasts=center_contrasts, surround_contrasts=surround_contrasts, pixel_min=pixel_min, pixel_max=pixel_max, device=device, size=size, img_res=img_res, neg_val=neg_val)
'''

## Perform contrast size tuning experiment 
'''
contrast_size_tuning_experiment_all_phase(h5_file=h5_file, all_neurons_model=all_neurons_model, neuron_ids=neuron_ids, overwrite=overwrite, phases=phases, contrasts=contrasts, radii=radii, pixel_min=pixel_min, pixel_max=pixel_max, device=device, size=size, img_res=img_res, neg_val=neg_val)
'''

#%%
###########################
## VISUALISE THE RESULTS ##
###########################

# 1) Select the filtering parameters

## This is the threshold for the fitting error to the Gaussian Model for neurons' receptive field
fit_err_thresh = 0.2

## Visualise how many neurons are being filtered
## Exclude neurons with high error. That are the neurons which receptive field could not fit to a gaussian function
filtered_neuron_ids = filter_fitting_error(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh, print_results=True)

## Exclude neurons that show neither surround suppression, nor response saturation
filtered_neuron_ids = filter_no_supp_neurons(h5_file=h5_file, neuron_ids=filtered_neuron_ids, print_results=True)

# 2) Visualise the results

## Visualise the size tuning experiment results
## For this function, select only the neurons with a decent suppression index
supp_thresh = 0.1
## Exclude the neurons with a low suppression index
filtered_neuron_ids = filter_low_supp_neurons(h5_file=h5_file, neuron_ids=filtered_neuron_ids, print_results=True)
size_tuning_results_1(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh, supp_thresh=supp_thresh)

## Visualise the suppression index distribution
size_tuning_results_2(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh)

## Visualise the contrast response results 
sort_by_std = False
spread_to_plot = [0,15,50,85,100]
contrast_response_results_1(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh, sort_by_std=sort_by_std, spread_to_plot=spread_to_plot)

## Visualise the contrast size tuning results
shift_to_plot = spread_to_plot
contrast_size_tuning_results_1(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh, shift_to_plot=shift_to_plot)


# #"MATHYS APRES VA VOIR LES FONCTIONS AU DESSUS 'exclude low supp neurons' etc
# # TODO aussi rename "phase" pour l'expérience full field ??? est ce qu'on veut qu'il y ait une erreur si sur ls annalyses de la fin on n'utilise pas la meme phase ?
# # TODO mon code n'est pas robuste aux tensors, listes, arrays etc, il se peut que si qqun utilise mes fonctions avec le mauvais truc ca marche pas
# # TODO in a readme file, make a full documentation, quote the source articles, define GSF, AMRF, SI ...

# %%
