#%%
import numpy as np
## Models
from nnvision.models.trained_models.v1_task_fine_tuned import v1_convnext_ensemble
## Functions
from functions import *
from functions_2 import *
from functions_3 import *
from functions_4 import *
## To select the well predicted neurons 
from surroundmodulation.utils.misc import pickleread


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
h5_file = "/project/mathys_code/article1.h5"

## Select the model of all neurons
all_neurons_model = v1_convnext_ensemble

## Chose the number of neurons to work with
n = 458
neuron_ids = np.arange(n) ## Because 458 outputs in our model
corrs = pickleread("/project/mathys_code/avg_corr.pkl")
neuron_ids = neuron_ids[corrs>0.75]

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
img_res = [93,93] 
pixel_min = -1.7876 # (the lowest value encountered in the data that served to train the model, serves as the black reference)
pixel_max =  2.1919
size = 2.67 #2.35

## Allow negative responses (because the response we show is the difference between the actual response and the response to a gray screen)
neg_val = True

## Analysis1 arguments
radii = radii
center_contrasts = np.logspace(np.log10(0.06),np.log10(1),18) #np.logspace(-2,np.log10(1),18) 
surround_contrasts = np.logspace(np.log10(0.06),np.log10(1),6) #np.logspace(-2,np.log10(1),6) 

## Analysis2 arguments
## For the orientation tuning experiment
ori_shifts = np.linspace(-np.pi,np.pi,9)
## For the ccss experiment
center_contrasts_ccss = np.array([0.06,0.12,0.25,0.5,1.0,1.5])
surround_contrast = contrast


## Analysis3 arguments
contrasts = np.logspace(np.log10(0.06),np.log10(1),5)  ## Same as in the article

#%%
################################
## GET THE NEURONS PREFERENCE ##
################################

## Find the neuron's preferred parameters

## Setting overwrite to true will empty any chosen experiment results and reperform it
overwrite = False

## (Optional) select the device
device = None

## Get the preferred full field gratings parameters for the neuron (orientation, spatial frequency, full field phase)

'''
get_all_grating_parameters(h5_file=h5_file, all_neurons_model=all_neurons_model, neuron_ids=neuron_ids, overwrite=overwrite, orientations = orientations, spatial_frequencies = spatial_frequencies, phases = phases, contrast = contrast, img_res = img_res, pixel_min = pixel_min, pixel_max = pixel_max, device = device, size = size )
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
contrast_size_tuning_experiment_all_phases(h5_file=h5_file, all_neurons_model=all_neurons_model, neuron_ids=neuron_ids, overwrite=overwrite, phases=phases, contrasts=contrasts, radii=radii, pixel_min=pixel_min, pixel_max=pixel_max, device=device, size=size, img_res=img_res, neg_val=neg_val)
'''


#%%
###########################
## VISUALISE THE RESULTS ##
###########################

# 1) Select the filtering parameters

#TODO DELETE
# neuron_ids = np.arange(458)
# neuron_id = 144
# filtered_neuron_ids = filter_no_supp_neurons(h5_file=h5_file, neuron_ids=neuron_ids, print_results=True)
# plot_size_tuning_curve(h5_file=h5_file, neuron_id=neuron_id)

# with h5py.File(h5_file, 'r') as f:
#     group = f['size_tuning/results']
#     circular_tuning_curve = f['size_tuning/curves'][f"neuron_{neuron_id}"][:][0]
#     print(group[f"neuron_{neuron_id}"][:])

# #%%

# print(circular_tuning_curve)
# print(radii)


#%%
import matplotlib.pyplot as plt

circular_tuning_curve = [0.01,0.07, 0.45, 0.96, 1, 0.45, 1.1]
r = np.arange(len(circular_tuning_curve))
plt.plot(circular_tuning_curve)
get_GSF_surround_AMRF(
    r,
    circular_tuning_curve,
    annular_tuning_curve = None
    )
#%%

## This is the threshold for the fitting error to the Gaussian Model for neurons' receptive field
fit_err_thresh = 0.2

## Visualise how many neurons are being filtered
## Exclude neurons with high error. That are the neurons which receptive field could not fit to a gaussian function
filtered_neuron_ids = filter_fitting_error(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh, print_results=True)

## Exclude neurons that show neither surround suppression, nor response saturation
filtered_neuron_ids = filter_no_supp_neurons(h5_file=h5_file, neuron_ids=filtered_neuron_ids, print_results=True)

# 2) Visualise the results
sort_by_std = False
spread_to_plot = [0,15,50,85,100]


## Visualise the size tuning experiment results
## For this function, select only the neurons with a decent suppression index
supp_thresh = 0.1
## Exclude the neurons with a low suppression index
filtered_neuron_ids = filter_low_supp_neurons(h5_file=h5_file, neuron_ids=filtered_neuron_ids, print_results=True)
size_tuning_results_1(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh, supp_thresh=supp_thresh)

## Visualise the suppression index distribution
size_tuning_results_2(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh)

## Visualise the contrast response results 

contrast_response_results_1(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh, sort_by_std=sort_by_std, spread_to_plot=spread_to_plot)

#%%
## Visualise the contrast size tuning results
shift_to_plot = spread_to_plot
low_contrast_id  = 1
high_contrast_id = -1
contrast_size_tuning_results_1(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh, shift_to_plot=shift_to_plot, low_contrast_id=low_contrast_id, high_contrast_id=high_contrast_id)

#%%
filtered_neuron_ids = filter_fitting_error(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh, print_results=False)
filtered_neuron_ids = filter_no_supp_neurons(h5_file=h5_file, neuron_ids=filtered_neuron_ids, print_results=False)
r = np.random.randint(0, len(filtered_neuron_ids))
neuron_id = filtered_neuron_ids[r]

with h5py.File(h5_file, 'r') as f :
    ## Get the parameters used in the size tuning experiment
    group = f['contrast_size_tuning']
    group_args_str = group.attrs["arguments"]
    group_dico_args = get_arguments_from_str(group_args_str)
    radii = group_dico_args["radii"]
    ## Convert str to a list
    radii = get_list_from_str(radii)
    
    curves = f['contrast_size_tuning/curves'][f"neuron_{neuron_id}"][:]
    low_contrast_curve = curves[low_contrast_id]
    high_contrast_curve = curves[high_contrast_id]
    GSF_low,_,_,_,_ = get_GSF_surround_AMRF(radii = radii,circular_tuning_curve=low_contrast_curve, annular_tuning_curve = None)
    GSF_high,_,_,_,_ = get_GSF_surround_AMRF(radii = radii,circular_tuning_curve=high_contrast_curve, annular_tuning_curve = None)
    print(GSF_low/GSF_high)
plot_contrast_size_tuning_curve(h5_file,neuron_id,title = None)
#%%

#############################################
############## SECOND ANALYSES ##############
#############################################
# COMPARE MODEL TO THE FOLLOWING ARTICLE :  #
#  https://doi.org/10.1152/jn.00693.2001    #
#############################################

#%%

##########################
## CHOSE THE PARAMETERS ##
##########################



#%%
#############################
## PERFORM THE EXPERIMENTS ##
#############################
overwrite = False

## Perform the orientation tuning experiment 

'''
orientation_tuning_experiment_all_phases(h5_file=h5_file, all_neurons_model=all_neurons_model, neuron_ids=neuron_ids, overwrite=overwrite, phases=phases, ori_shifts=ori_shifts, contrast=contrast, pixel_min=pixel_min, pixel_max=pixel_max, device=device, size=size, img_res=img_res)
'''

## Perform the center contrast surround suppression experiment (ccss)
## 1h15

'''
center_contrast_surround_suppression_experiment(h5_file=h5_file, all_neurons_model=all_neurons_model, neuron_ids=neuron_ids, overwrite=overwrite, center_contrasts_ccss=center_contrasts_ccss, surround_contrast=surround_contrast, phases=phases, pixel_min=pixel_min, pixel_max=pixel_max, device=device, size=size, img_res=img_res)
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

sort_by_std = False
spread_to_plot = [0,15,50,85,100]

## Orientation tuning results :
orientation_tuning_results_1(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh)

orientation_tuning_results_2(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh)

#%%
## CCSS results :

ccss_results_1(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh)

ccss_results_2(h5_file=h5_file, neuron_ids=neuron_ids,
    contrast_id = -4,       
    fit_err_thresh=fit_err_thresh)
    
ccss_results_2(h5_file=h5_file, neuron_ids=neuron_ids,
    contrast_id = 1,
    fit_err_thresh=fit_err_thresh)

#%%
####################################################
################# THIRD ANALYSES ###################
####################################################
#     COMPARE MODEL TO THE FOLLOWING ARTICLE :     #
#  https://doi.org/10.1523/JNEUROSCI.1991-09.2009  #
####################################################

#%%
#############################
## PERFORM THE EXPERIMENTS ##
#############################

## TODO name of the experiment not good
## 1) Chose the parameters for the black or white preference experiment
#TODO NAME ALREADY TAKEN, CAN LEAD TO SOME UNWANTED ERRORS 
dot_size_in_pixels = 5 ## Because ~ 0.2 DVF

overwrite = False

## Perform the black or white preference experiment

'''
black_white_preference_experiment(h5_file=h5_file, all_neurons_model=all_neurons_model, neuron_ids=neuron_ids, overwrite=overwrite, dot_size_in_pixels=dot_size_in_pixels, contrast=contrast, img_res=img_res, pixel_min=pixel_min, pixel_max=pixel_max, device=device)
'''


#%%
###########################
## VISUALISE THE RESULTS ##
###########################

# 1) Select the filtering parameters


## This is the threshold for the Signal Noise Ratio 
SNR_thresh = 2

## Visualise how many neurons are being filtered
## Exclude the neurons with neither SNRw nor SNRb that are > SNR_thresh. function 'filter_SNR'
filter_SNR(h5_file=h5_file, neuron_ids=neuron_ids, SNR_thresh=SNR_thresh, print_results=True)

# 2) Visualise the results

## Black or white preference results :

'''
black_white_results_1(h5_file=h5_file, neuron_ids=neuron_ids, neuron_depths=neuron_depths, SNR_thresh=SNR_thresh)
'''



#%%
####################################################
################# Fourth ANALYSES ##################
####################################################
#     COMPARE MODEL TO THE FOLLOWING ARTICLE :     #
#          https://doi.org/10.1038/nn.3402         #
####################################################

#%%
#############################
## PERFORM THE EXPERIMENTS ##
#############################

## 1) Chose the parameters 



overwrite = False

## Perform the texture noise response experiment

'''
texture_noise_response_experiment(h5_file=h5_file, all_neurons_model=all_neurons_model, neuron_ids=neuron_ids, directory_imgs=directory_imgs, overwrite=overwrite, num_samples=num_samples, img_res=target_res, device=device)
'''

#%%
###########################
## VISUALISE THE RESULTS ##
###########################

texture_noise_response_results_1(h5_file=h5_file, neuron_ids=neuron_ids)

# %%



# "MATHYS APRES VA VOIR LES FONCTIONS AU DESSUS 'exclude low supp neurons' etc
# TODO aussi rename "phase" pour l'expérience full field ??? est ce qu'on veut qu'il y ait une erreur si sur ls annalyses de la fin on n'utilise pas la meme phase ?
# TODO mon code n'est pas robuste aux tensors, listes, arrays etc, il se peut que si qqun utilise mes fonctions avec le mauvais truc ca marche pas
# TODO in a readme file, make a full documentation, quote the source articles, define GSF, AMRF, SI ...
# TODO maybe interesting to add a description in the subgroups that explains the shape of the results
# TODO clean functions_2.py
# TODO in every function, make a print in the terminal like print("Size tuning experiment :")
# TODO remove all TODO



############
## RANDOM ##
############


h5_file= '/project/mathys_code/article1 copy.h5'

plot_size_tuning_curve(h5_file=h5_file, neuron_id=0)



# %%
