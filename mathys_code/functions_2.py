#%%
import numpy as np
import torch
from tqdm import tqdm
from itertools import combinations
## Plots
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
## Image generation
import imagen
from imagen.image import BoundingBox
## Important functions
from surroundmodulation.utils.misc import rescale
from scipy.optimize import curve_fit
## Import models 
from nnvision.models.trained_models.v1_task_fine_tuned import v1_convnext_ensemble #TODO remove this line since useful only in main
from surroundmodulation.models import SingleCellModel
## Data storage
import h5py
## Import functions
from functions import *

#%%

##############################################################
## Functions for the ARTICLE II :                           ##
## Selectivity and Spatial Distribution of Signals From the ##
## Receptive Field Surround in Macaque V1 Neurons           ##
## JAMES R. CAVANAUGH, WYETH BAIR, AND J. ANTHONY MOVSHON   ##
## DOI : https://doi.org/10.1152/jn.00693.2001              ##
##############################################################

def get_orientation_tuning_curves_all_phase(
    single_model,
    preferred_ori,
    preferred_sf,
    x_pix,
    y_pix,
    GSF,
    AMRF,
    phases = np.linspace(0, 2*np.pi, 37)[:-1],
    ori_shifts = np.linspace(-180,180,8),
    contrast = 1,
    pixel_min = -1.7876, 
    pixel_max =  2.1919, 
    size = 2.67,
    img_res = [93,93],
    device = None,
    do_surround_fixed_center = True,
    center_shift = 0
):
    ''' For a given neuron, this function does either :

            If do_surround_fixed_center == True :

                Computes an orientation tuning curve for the surround orientation with the center's orientation fixed

            Else :
                
                Computes the orientation tuning curve for the center with no surround

        - Returns a curve of response for a center stimulus alone at different orientations
        - Returns a curve of response for a center stimulus with a fixed orientation and a surround stimulus at different orientations

    Arguments :

        - single_model              : A single cell model of the class 'surroundmodulation.models.SingleCellModel'
        - preferred_ori             : The preferred orientation (the one that elicited the greatest response). 
        - preferred_sf              : The preferred spatial frequency
        - x_pix                     : The x coordinates of the center of the neuron's receptive field (in pixel, can be a float)
        - y_pix                     : The y coordinates   ""        ""        ""
        - GSF                       : Grating Summation Field, it will be the center's radius
        - AMRF                      : Annular Minimum Response Field, it will be the inner edge radius of the surround (since the surround is a ring)
        - phases                    : An array containing the phases to try
        - ori_shifts                : An array containing the shift of orientation in Radiant to try (realtive to the preferred orientation) (Make sure that the middle value is 0 for better visualisation, don't expend over +-180  deg (+-pi))
        - pixel_min                 : Value of the minimal pixel that will serve as the black reference
        - pixel_max                 : Value of the maximal pixel that will serve as the white reference (NB : The gray value will be the mean of those two)
        - size                      : The size of the image in terms of visual field degrees
        - img_res                   : Resolution of the image in term of pixels shape : [pix_y, pix_x]
        - device                    : The device on which to execute the code, if set to "None" it will take the available one
        - do_surround_fixed_center  : Bool, whether or not the output should be the orientation tuning curve of the center alone or the fixed center and the surround
        - center_shift              : (Optional), The orientation shift to apply on the center for the surround orientation tuning with fixed center
    
    Outputs :

        If do_surround_fixed_center == True

            - orientation_tuning_curve : An array containing the response of the neuron dor the image which is a center grating patch with a fixed orientation and a surround with changing orientation

        If do_surround_fixed_center == False

            - orientation_tuning_curve : An array containing the response of the neuron for the image which is a center grating patch with changing orientation  
    '''

    ## Select device
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    single_model.to(device)

    ## Evaluation mode
    single_model.eval()

    ## Convert the size to the right shape ( [size_height, size_width] )
    if type(size) is int or type(size) is float: 
        size = [size]*2

    ## Avoid overlapping
    if GSF>AMRF :
        AMRF = GSF

    ## This will store the model responses for every phase (row) and every orientation (column) 
    all_orientation_curves = torch.zeros((len(phases), len(ori_shifts)))

    ## For every phase :
    for num_phase, phase in enumerate(phases) :

        ## For every orientation shift 
        for num_ori, ori_shift in enumerate(ori_shifts) : 
            
            ## Change the orientation
            orientation = preferred_ori + ori_shift

            ## Stimulus = center only
            if do_surround_fixed_center == False :
                
                ## Create the center stimulus
                stimulus = torch.Tensor(imagen.SineGrating(
                                    mask_shape=imagen.Disk(smoothing=0.0, size=GSF*2.0),
                                    orientation=orientation,
                                    frequency=preferred_sf,
                                    phase=phase,
                                    bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                                    offset = -1,
                                    scale=2,  
                                    xdensity=img_res[1]/size[1],
                                    ydensity=img_res[0]/size[0], 
                                    x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                                    y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())
                
                ## Mask the circular stimulus 
                center_mask = torch.Tensor(imagen.Disk(smoothing=0.0,
                        size= GSF*2.0, 
                        bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                        xdensity=img_res[1]/size[1],
                        ydensity=img_res[0]/size[0], 
                        x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                        y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())
    
                ## Add the gray background to the circular stimulus
                stimulus = stimulus * center_mask

                ## Change the contrast 
                stimulus *= contrast

                ## Convert to the right shape for the model
                stimulus = rescale(stimulus,-1,1,pixel_min,pixel_max).reshape(1,*img_res).to(device)

            else :

                center_ori = preferred_ori

                ## Change the center's orientation 
                if center_shift is not None :
                    center_ori = center_ori + center_shift
    
                stimulus = get_center_surround_stimulus(center_radius=GSF, center_ori=center_ori, center_sf=preferred_sf, center_phase=phase, center_contrast=contrast, surround_radius=AMRF, surround_ori=orientation, surround_sf=preferred_sf, surround_phase=phase, surround_contrast=contrast, x_pix=x_pix, y_pix=y_pix, pixel_min=pixel_min, pixel_max=pixel_max, size=size, img_res=img_res, device=device)    
            
            with torch.no_grad():
                
                ## Get the model's response
                response = single_model(stimulus).item()

            ## Save the response at the correct place
            all_orientation_curves[num_phase, num_ori] = response

    ## Get the maximum response accross phases
    orientation_tuning_curve = torch.max(all_orientation_curves, dim=0).values

    return orientation_tuning_curve


def orientation_tuning_experiment_all_phases(
    h5_file,
    all_neurons_model,
    neuron_ids,   
    overwrite = False, 
    phases = np.linspace(0, 2*np.pi, 37)[:-1],
    ori_shifts = np.linspace(-np.pi,np.pi,9),
    contrast = 1,
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.67,
    img_res = [93,93]
):
    ''' This function performs an orientation tuning experiment for the neurons accross multiple phases :

            - 1) It computes the orientation tuning curve for the center alone and save it in the '/orientation_tuning/curves_center/' folder
            - 2) It computes the orientation tuning curves for the surround orientation with a fixed center orientation
            - 3) It performs 2) for a center orientation of -45°, 0° and +45° relative to the preferred orientation and save the results in the '/orientation_tuning/curves_surround_fixed_center/' folder
            - 4) It saves the most suppressing surround orientation (don't look for orientations above 90 and below -90 degrees) for all 3 center orientation in the /orientation_tuning/results/' folder
        
        It saves the data with this architecture : 


                                        _________ SubGroup /orientation_tuning/curves_center                 --> neuron datasets
                                        |
        Group /orientation_tuning ______|________ SubGroup /orientation_tuning/curves_surround_fixed_center  --> neuron datasets
                                        |
                                        |________ SubGroup /orientation_tuning/results                       --> neuron datasets
        
        Prerequisite :

            - function 'get_all_grating_parameters' executed for the required neurons with matching arguments
            - function 'get_preferred_position'     executed for the required neurons with matching arguments
            - function 'size_tuning_experiment_all_phases' executed for the required neurons with matching arguments        

        Arguments :

            - ori_shifts : An array containing the shift of orientation (in radiant) to try (realtive to the preferred orientation)
            - other      : See the 'get_orientation_tuning_curves_all_phase' function

        Outputs :

            - datasets in ../curves_center                : An array containing the responses of the neuron to all orientation

            - datasets in ../curves_surround_fixed_center : A matrix containing the orientation tuning curves for the surround (col) for 3 differnet center orientation (-45°, +0°, +45°) (row) 

            - datasets in ../results                      : An array containing the most suppressive orientation (only for orientations in [-90°,+90°])
    '''

    print(' > Orientation tuning experiment')

    ## Required groups
    group_path_ff_params   = "/full_field_params"
    group_path_pos         = "/preferred_pos"
    group_path_st_results  = "/size_tuning/results"

    ## Groups to fill
    group_path_orientation_tuning = "/orientation_tuning"
    subgroup_path_results         = group_path_orientation_tuning + "/results"
    subgroup_path_curves_center   = group_path_orientation_tuning + "/curves_center"
    subgroup_path_curves_surround = group_path_orientation_tuning + "/curves_surround_fixed_center"

    ## Clear the group if requested
    if overwrite : 
        clear_group(h5_file,group_path_orientation_tuning)

    ## Initialize the Group and the subgroups
    args_str = f"phases={phases}/ori_shifts={ori_shifts}/contrast={contrast}/pixel_min={pixel_min}/pixel_max={pixel_max}/size={size}/img_res={img_res}"
    group_init(h5_file=h5_file, group_path=group_path_orientation_tuning, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_results, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_curves_center, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_curves_surround, group_args_str=args_str)


    ## Check compatibility between every experiment
    check_compatibility(h5_file=h5_file, list_group_path=[group_path_ff_params, group_path_pos, group_path_st_results], neuron_ids=neuron_ids, group_args_str=args_str)

    with h5py.File(h5_file,'a') as file :

        ## Access the groups 
        subgroup_results         = file[subgroup_path_results]
        subgroup_curves_center   = file[subgroup_path_curves_center]
        subgroup_curves_surround = file[subgroup_path_curves_surround]

        ## For every neuron
        for neuron_id in tqdm(neuron_ids):

            neuron = f"neuron_{neuron_id}"

            ## Check if the neuron data is not already present
            if neuron not in subgroup_results :

                ## Get the model for the neuron
                single_model = SingleCellModel(all_neurons_model,neuron_id )

                ## Get the neuron's preferred parameters 
                preferred_ori = file[group_path_ff_params][neuron][:][0]
                preferred_sf  = file[group_path_ff_params][neuron][:][1]
                x_pix         = file[group_path_pos][neuron][:][0]
                y_pix         = file[group_path_pos][neuron][:][1]
                GSF           = file[group_path_st_results][neuron][1]
                AMRF          = file[group_path_st_results][neuron][3]

                ## 1) We set 'do_surround_fixed_center' to False in order to have the center orientation tuning curve

                center_tuning_curve = get_orientation_tuning_curves_all_phase(single_model=single_model, preferred_ori=preferred_ori, preferred_sf=preferred_sf, contrast=contrast, x_pix=x_pix, y_pix=y_pix, GSF=GSF, AMRF=AMRF, phases=phases, ori_shifts=ori_shifts, pixel_min=pixel_min, pixel_max=pixel_max, size=size, img_res=img_res, device=device, center_shift=None,
                    do_surround_fixed_center = False)

                ## 2-3)
                ## Perform for three center orientation
                center_shifts = np.array([-(np.pi/4),0,np.pi/4])
                center_oris = preferred_ori + center_shifts

                ## This will be where the results will be saved, every row is a different center orientation (-45,0,+45 degrees), every column correspond to a surround orientation
                surround_tuning_curves = torch.zeros(len(center_oris), len(ori_shifts))

                ## This will be where the most suppressing surround orientation will be saved. 
                most_suppr_surr = torch.zeros(len(center_oris))


                for i, center_shift in enumerate(center_shifts):

                    ## Set 'do_surround_fixed_center' to True

                    surround_tuning_curve = get_orientation_tuning_curves_all_phase(single_model=single_model, preferred_ori=preferred_ori, preferred_sf=preferred_sf, contrast=contrast, x_pix=x_pix, y_pix=y_pix, GSF=GSF, AMRF=AMRF, phases=phases, ori_shifts=ori_shifts, pixel_min=pixel_min, pixel_max=pixel_max, size=size, img_res=img_res, device=device, 
                                                center_shift=center_shift,                                                            
                                                do_surround_fixed_center=True)

                    surround_tuning_curves[i] = surround_tuning_curve

                    ## 4) Select the values for a orientation shift >= -90 and <= 90 degrees

                    sub_array_id = np.where(np.abs(ori_shifts) < np.pi/2 + 0.01 )[0]
                    sub_ori = np.copy(ori_shifts[sub_array_id])
                    sub_tuning_curve = np.copy(surround_tuning_curve[sub_array_id])

                    ## Get the index of the most suppressing contrast orientation
                    argmin = np.argmin(sub_tuning_curve)

                    ## Get the corresponding orientation (TODO in radiant?) 
                    most_suppr_surr[i] = (sub_ori[argmin])

                ## Save every result in the HDF5 file 
                subgroup_curves_center.create_dataset(name=neuron, data=center_tuning_curve)
                subgroup_curves_surround.create_dataset(name=neuron, data=surround_tuning_curves)
                subgroup_results.create_dataset(name=neuron, data=most_suppr_surr)

                
            
def plot_orientation_tuning_curve(
    h5_file,
    neuron_id
):
    
    ''' This function goes into a HDF5 file, get the orientation tuning curves (center and surround with fixed center) for one neuron and plot them.

        Prerequisite :

            - function 'orientation_tuning_experiment_all_phases' executed for the required neuron    

        About the plot :

            - The dashed horizontal line correspond to the center max response at the preferred orientation
            - The x axis displays the orientation in degrees for better clarity
    '''

    group_path = '/orientation_tuning'
    subgroup_path_center = group_path + '/curves_center'
    subgroup_path_surround = group_path + '/curves_surround_fixed_center'

    ## Check if the orientation tuning experiment has been performed
    check_group_exists_error(h5_file, subgroup_path_surround)

    ## Check if the neuron is present in the data
    check_neurons_presence_error(h5_file, [subgroup_path_surround], [neuron_id])

    with h5py.File(h5_file, 'r') as f :

        ## Get the curves for the neuron
        neuron = f"neuron_{neuron_id}"
        center_tuning_curve = f[subgroup_path_center][neuron][:]
        surround_tuning_curve = f[subgroup_path_surround][neuron][:][1] ## Select the curve for a center orientation fixed at +0° relative to the preffered orientation

        ## Get the parameters used in the orientation tuning experiment
        group = f[group_path]
        group_args_str = group.attrs["arguments"]
        group_dico_args = get_arguments_from_str(group_args_str)
        ori_shifts = group_dico_args["ori_shifts"]
        ori_shifts = get_list_from_str(ori_shifts)

    ## Make the plot
    max_val_center = max(center_tuning_curve)
    plt.plot(np.degrees(ori_shifts), center_tuning_curve, label='Center only', color = 'darkslateblue', linewidth= 0.7)
    plt.plot(np.degrees(ori_shifts), surround_tuning_curve, label='Surround with fixed center', color = 'darkslateblue', linewidth=2.5,)
    plt.plot([min(np.degrees(ori_shifts)), max(np.degrees(ori_shifts))], [max_val_center, max_val_center], linestyle='--',color = 'k', linewidth=1)

    ## Change the ticks for better clarity
    custom_ticks = [-180,-135,-90,-45,0,45,90,135,180]
    plt.xticks(custom_ticks)

    ## Unzoom a little
    minresp = min(min(center_tuning_curve),min(surround_tuning_curve))
    maxresp = max(max(center_tuning_curve),max(surround_tuning_curve))
    plt.ylim(minresp-0.3, maxresp+0.3)

    plt.title(f"Orientation Tuning Curve for the neuron {neuron_id}")
    plt.xlabel("Orientation shift compared to the preferred orientation (degrees)")
    plt.ylabel("Response")
    plt.legend()
    plt.show()


def orientation_tuning_results_1(
    h5_file, 
    neuron_ids, 
    fit_err_thresh=0.2
):
    ''' This function aims to visualise the response of neurons to different center and surround orientations :
            
            - 1) It performs some filtering.
            - 2) It prints some informations in the terminal. 
            - 3) It plots the center orientation tuning curve and the surround orientation tuning curve with fixed center

        Prerequisite :
        
    TODO?   - function 'size_tuning_experiment_all_phases' executed for the required neurons
            - function 'orientation_tuning_experiment_all_phases' executed for the required neurons

        Filtering :
        
            - Exclude the neurons with receptive fields poorly fitted to a Gaussian model. function 'filter_fitting_error'
    TODO?   - Exclude the neurons that appeard to have no surround suppression nor response saturation. function 'filter_no_supp_neurons'

        Arguments :

            - fit_err_thresh : Fitting error threshold, exclude every neuron with an error above that value
    '''

    group_path    = '/orientation_tuning'
    subgroup_path = group_path + '/results'

    ## Check if the 'get_preferred_position' function has been computed
    check_group_exists_error(h5_file=h5_file, group_path=subgroup_path)
    
    ## Check if the neurons are present in the data
    check_neurons_presence_error(h5_file=h5_file, list_group_path=[subgroup_path], neuron_ids=neuron_ids)

    ## Filter on fitting error
    filtered_neuron_ids = filter_fitting_error(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh, print_results=False)

    ## Filter the neurons with no surround suppression and no response saturation
    filtered_neuron_ids = filter_no_supp_neurons(h5_file=h5_file, neuron_ids=filtered_neuron_ids, print_results = False)

    n = len(neuron_ids)
    n_new = len(filtered_neuron_ids)

    ## Initialise the arrays to fill
    with h5py.File(h5_file, 'r') as f :
        ## Get the parameters used in the experiment
        group_args_str = f[subgroup_path].attrs["arguments"]
        group_dico_args = get_arguments_from_str(group_args_str)
        ori_shifts = group_dico_args["ori_shifts"]
        ## Convert str to a list
        ori_shifts = get_list_from_str(ori_shifts)

        curves_center      = np.zeros((n_new,len(ori_shifts)))
        curves_surround    = np.zeros((n_new,len(ori_shifts)))

    ## Get the curves for every neuron
    with h5py.File(h5_file, 'r') as f :
        for i, neuron_id in enumerate(filtered_neuron_ids) :
            neuron = f"neuron_{neuron_id}"

            curve_center    = f[group_path + '/curves_center'][neuron][:]
            curve_surround  = f[group_path + '/curves_surround_fixed_center'][neuron][:][1]

            curves_center[i] = curve_center
            curves_surround[i] = curve_surround

    mean_curve_center = np.mean(curves_center, axis=0)
    mean_curve_surround = np.mean(curves_surround, axis=0)

    print("--------------------------------------")
    print("Visualisation of random orientation tuning curves :")
    print(f"    > Analysis made on {n} neurons")
    print(f"    > {n_new} neurons ({round((n_new/n*100), 2)}%) left after filtration")
    print()
    print(f"    > The mean curves accross every neuron :")

    ## Make the plot
    max_val_center = max(mean_curve_center)
    plt.plot(np.degrees(ori_shifts), mean_curve_center, label='Center only', color = 'darkgoldenrod', linewidth= 0.7)
    plt.plot(np.degrees(ori_shifts), mean_curve_surround, label='Surround with fixed center', color = 'darkgoldenrod', linewidth=2.5,)
    plt.plot([min(np.degrees(ori_shifts)), max(np.degrees(ori_shifts))], [max_val_center, max_val_center], linestyle='--',color = 'k', linewidth=1)

    ## Change the ticks for better clarity
    custom_ticks = [-180,-135,-90,-45,0,45,90,135,180]
    plt.xticks(custom_ticks)

    ## Unzoom a little
    minresp = min(min(mean_curve_center),min(mean_curve_surround))
    maxresp = max(max(mean_curve_center),max(mean_curve_surround))
    plt.ylim(minresp-0.3, maxresp+0.3)

    plt.title(f"Mean orientation tuning curves accross every neuron")
    plt.xlabel("Orientation shift compared to the preferred orientation (degrees)")
    plt.ylabel("Response")
    plt.legend()
    plt.show()

    print()
    print(f"    > Random neurons :")

    for i in range(3) : 

        ## Select a random neuron
        r = np.random.randint(0,n_new)
        neuron_id = filtered_neuron_ids[r]

        plot_orientation_tuning_curve(h5_file=h5_file, neuron_id=neuron_id)

    print("--------------------------------------")
    print()


def orientation_tuning_results_2(
    h5_file, 
    neuron_ids, 
    fit_err_thresh=0.2
):
    ''' This function to show the most suppressive surround orientations for different center orientations
            
            - 1) It performs some filtering.
            - 2) It prints some informations in the terminal. 
            - 3) It plot 3 histograms, one for each center orientation (-45°, 0 and +45°), the histograms shows the distribution of the most suppressive surround orientation
        NB : The orientation is the orientation compared to the preferred orientation

        Prerequisite :
        
    TODO?   - function 'size_tuning_experiment_all_phases' executed for the required neurons
            - function 'orientation_tuning_experiment_all_phases' executed for the required neurons

        Filtering :
        
            - Exclude the neurons with receptive fields poorly fitted to a Gaussian model. function 'filter_fitting_error'
    TODO?   - Exclude the neurons that appeard to have no surround suppression nor response saturation. function 'filter_no_supp_neurons'

    '''
    subgroup_path = '/orientation_tuning/results'

    ## Check if the 'get_preferred_position' function has been computed
    check_group_exists_error(h5_file=h5_file, group_path=subgroup_path)
    
    ## Check if the neurons are present in the data
    check_neurons_presence_error(h5_file=h5_file, list_group_path=[subgroup_path], neuron_ids=neuron_ids)

    ## Filter on fitting error
    filtered_neuron_ids = filter_fitting_error(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh, print_results=False)

    ## Filter the neurons with no surround suppression and no response saturation
    filtered_neuron_ids = filter_no_supp_neurons(h5_file=h5_file, neuron_ids=filtered_neuron_ids, print_results = False)

    ## Get the neurons results
    all_result_0 = []   #center orientation  : -45
    all_result_1 = []   #center orientation  :   0
    all_result_2 = []   #center orientation  :  45

    with h5py.File(h5_file, 'r') as f : 
        
        for neuron_id in filtered_neuron_ids :

            neuron = f"neuron_{neuron_id}"

            neuron_results = f[subgroup_path][neuron][:]
            result_0 = neuron_results[0]
            result_1 = neuron_results[1]
            result_2 = neuron_results[2]

            all_result_0.append(result_0)
            all_result_1.append(result_1)
            all_result_2.append(result_2)

            ## Get the parameters used in the orientation tuning experiment
            group = f['/orientation_tuning']
            group_args_str = group.attrs["arguments"]
            group_dico_args = get_arguments_from_str(group_args_str)
            ori_shifts = group_dico_args["ori_shifts"]
            ori_shifts = get_list_from_str(ori_shifts)

    ## Select the orientation shift >= -90 and <= 90 degrees
    sub_array_id = np.where(np.abs(ori_shifts) < np.pi/2 + 0.01 )[0] # In radiant for now
    sub_ori = ori_shifts[sub_array_id]
    sub_ori = np.degrees(sub_ori) #Convert in degrees

    n = len(neuron_ids)
    n_new = len(filtered_neuron_ids)

    ## Print in the terminal
    print("--------------------------------------")
    print("Visualisation of the most suppressive surround orientations :")
    print(f"    > Analysis made on {n} neurons")
    print(f"    > {n_new} neurons ({round((n_new/n*100), 2)}%) left after filtration")
    print(f"    > Plots :")

    all_results = [all_result_0, all_result_1, all_result_2]
    for i, orientation in enumerate([-45,0,45]):
        
        print()
        print(f"    > For center orientation of {orientation} deg :")
        print()

        results = np.degrees(all_results[i])

        ## Create the good amount of bins
        nbins = len(sub_ori)
        step  = np.abs(sub_ori[-1] - sub_ori[-2])
        edge  = sub_ori[-1] + (step/2)
        bins = np.linspace(- edge, edge,nbins+1) # The values should not be below -90 and above 90 deg
        weights = np.ones(results.shape) / len(results)
        plt.hist(results, bins = bins,edgecolor='black', density=False, weights=weights )

        ## Annotate the plot
        plt.xticks(sub_ori)
        plt.title(f"Distribution of the most suppressive surround orientation for a center at {orientation} deg")
        plt.ylabel("Distribution")
        plt.xlabel("Most suppressive surround direction\nrelative to the preferred one (deg)")
        plt.show()




def center_contrast_surround_suppression_experiment(
    h5_file,
    all_neurons_model,
    neuron_ids,
    overwrite = False,
    center_contrasts_ccss = np.array([0.06,0.12,0.25,0.5,1.0]),
    surround_contrast = 1,
    phases = np.linspace(0, 2*np.pi, 37)[:-1],
    pixel_min = -1.7876,
    pixel_max =  2.1919,
    device = None,
    size = 2.67,
    img_res = [93,93]  
):
    ''' This function aims to determine the link between the surround suppression and the center stimulus contrast
        For every center_contrast in center_contrasts_ccss :

            1) The center patch of each stimulus is at the preferred orientation and with a contrast = center_contrast
            2) It gets the response of each neuron to this center and surround at the parallel and orthogonal orientation + the response of the center alone
            3) It saves the best responses for the parallel, orthogonal and center alone stimulus accross every phase
            4) It saves the results in the HDF5 file as a matrix where every row is a center contrast 
        

        It saves the data with this architecture : 
                                                            
            Group /center_contrast_surround_suppression _______ SubGroup ../results     --> neuron datasets

        Prerequisite :

            - function 'get_all_grating_parameters' executed for the required neurons with matching arguments
            - function 'get_preferred_position'     executed for the required neurons with matching arguments
            - function 'size_tuning_experiment_all_phases' executed for the required neurons with matching arguments    

        Outputs :

            - datasets in ../results  : A matrix (shape [len(center_contrasts_ccss), 3]) containing the response of each neuron for a surround at the preferred orientation, orthogonal position and the response of the center alone.
                                        Each column is a condition and each row correspond to a center contrast

                                                format :             resp_iso | resp_ortho | resp_center_alone
                                                        _______________________________________________________
                                                        contrast 0 |          |            |
                                                        ______________________|____________|___________________
                                                        contrast...|          |            |
                                                        _______________________________________________________
    '''                                                 
    print(' > Center contrast surround suppression experiment')

    ## Required groups
    group_path_ff_params   = "/full_field_params"
    group_path_pos         = "/preferred_pos"
    group_path_st_results  = "/size_tuning/results"

    ## Groups to fill
    group_path      = "/center_contrast_surround_suppression"
    subgroup_path   = group_path + "/results"

    ## Clear the group if requested    
    if overwrite : 
        clear_group(h5_file,group_path)

    ## Initialize the Group and subgroup
    args_str = f"center_contrasts_ccss={center_contrasts_ccss}/surround_contrast={surround_contrast}/phases={phases}/pixel_min={pixel_min}/pixel_max={pixel_max}/size={size}/img_res={img_res}"
    group_init(h5_file=h5_file, group_path=group_path, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path, group_args_str=args_str)

    ## Check compatibility between every experiment
    check_compatibility(h5_file=h5_file, list_group_path=[group_path_ff_params, group_path_pos, group_path_st_results], neuron_ids=neuron_ids, group_args_str=args_str)

    ## Select device
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ## Convert the size to the right shape ( [size_height, size_width] )
    if type(size) is int or type(size) is float: 
        size = [size]*2

    with h5py.File(h5_file, 'a') as f :
        
        ## Access the subgroup
        subgroup = f[subgroup_path]

        for neuron_id in tqdm(neuron_ids) :

            neuron = f"neuron_{neuron_id}"

            ## Check if the neuron data is not already present
            if neuron not in subgroup :
                
                ## Get the model for the neuron
                single_model = SingleCellModel(all_neurons_model,neuron_id)
                single_model.to(device)
                single_model.eval()

                ## Get the neuron's preferred parameters 
                preferred_ori = f[group_path_ff_params][neuron][:][0]
                preferred_sf  = f[group_path_ff_params][neuron][:][1]
                x_pix         = f[group_path_pos][neuron][:][0]
                y_pix         = f[group_path_pos][neuron][:][1]
                GSF           = f[group_path_st_results][neuron][1]
                AMRF          = f[group_path_st_results][neuron][3]
                if AMRF<GSF :
                    AMRF=GSF

                max_resp = np.zeros((len(center_contrasts_ccss), 3))

                with torch.no_grad():

                    for num_contrast, center_contrast in enumerate(center_contrasts_ccss) : 

                        max_resp_row = max_resp[num_contrast]

                        ## The parallel and orthogonal orientation  
                        for i, shift_ori in enumerate([0,np.pi/2]) :
                            surround_ori = preferred_ori + shift_ori

                            for phase in phases : 

                                ## Create the stimulus
                                stimulus = get_center_surround_stimulus(center_radius=GSF, center_ori=preferred_ori, center_sf=preferred_sf, center_phase=phase, center_contrast=center_contrast, surround_radius=AMRF, surround_ori=surround_ori, surround_sf=preferred_sf, surround_phase=phase, surround_contrast=surround_contrast, x_pix=x_pix, y_pix=y_pix, pixel_min=pixel_min, pixel_max=pixel_max, size=size, img_res=img_res, device=device)
                                
                                ## Get the neuron's response
                                resp = single_model(stimulus).item()

                                ## Save the maximum response
                                if resp > max_resp_row[i] :
                                    max_resp_row[i] = resp
                            

                        ## For the center alone :
                        i = 2

                        for phase in phases :

                            ## Create the stimulus (set the surround contrast at zero)
                            stimulus = get_center_surround_stimulus(center_radius=GSF, center_ori=preferred_ori, center_sf=preferred_sf, center_phase=phase, center_contrast=center_contrast, surround_radius=AMRF, surround_ori=surround_ori, surround_sf=preferred_sf, surround_phase=phase, 
                                    surround_contrast=0, 
                                    x_pix=x_pix, y_pix=y_pix, pixel_min=pixel_min, pixel_max=pixel_max, size=size, img_res=img_res, device=device)

                            ## Get the neuron's response
                            resp = single_model(stimulus).item()

                            ## Save the maximum response
                            if resp > max_resp_row[i] :
                                max_resp_row[i] = resp

                    ## Update the row 
                    max_resp[num_contrast] = max_resp_row

                    ## Save the results the corresponding subgroup
                    subgroup.create_dataset(name=neuron, data=max_resp)


def plot_ccss_curves(
    h5_file,
    neuron_id
):  
    ''' This function goes into a HDF5 file, get every values for the required neuron and plot three curves :

            - One curve for the responses of the neuron for a surround at the preffered orientation
            - One curve for the responses of the neuron for a surround at the orthogonal orientation
            - One curve for the responses of the neuron for the center only

        NB1 : ccss stands for 'center contrast surround suppression'
        NB2 : yaxis  = Response of the single neuron modele 
              xaxis  = Center contrast

        Prerequisite :

            - function 'center_contrast_surround_suppression_experiment' executed for the required neuron
    '''

    subgroup_path = "/center_contrast_surround_suppression/results"

    ## Check if the experiment has been performed
    check_group_exists_error(h5_file, subgroup_path)

    ## Check if the neuron is present in the data
    check_neurons_presence_error(h5_file, [subgroup_path], [neuron_id])

    with h5py.File(h5_file, 'r') as f :
        
        neuron = f"neuron_{neuron_id}"
        neuron_curves = f[subgroup_path][neuron][:]

        curve_iso    = neuron_curves[:,0]
        curve_ortho  = neuron_curves[:,1]
        curve_center = neuron_curves[:,2]

        ## Get the parameters used in the experiment
        group_args_str = f[subgroup_path].attrs["arguments"]
        group_dico_args = get_arguments_from_str(group_args_str)
        center_contrasts_ccss = group_dico_args["center_contrasts_ccss"]
        ## Convert str to a list
        center_contrasts_ccss = get_list_from_str(center_contrasts_ccss)

        plt.plot(center_contrasts_ccss,curve_center, label = "Center alone" )
        plt.plot(center_contrasts_ccss,curve_ortho, label = "Surround ortho-oriented" )
        plt.plot(center_contrasts_ccss,curve_iso, label = "Surround iso-oriented" )
        
        plt.xscale("log")
        plt.xticks(center_contrasts_ccss)
        plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        
        plt.title(f"CCSS Curves for the neuron {neuron_id}")
        plt.xlabel("Contrast of the center")
        plt.ylabel("Response")
        plt.legend()

        plt.show()

def ccss_results_1(
    h5_file,
    neuron_ids,
    fit_err_thresh = 0.2    
):
    '''
    This function aims to visualise the relation between the center contrast and the surround suppression (with a surround at different orientations)
            
            - 1) It performs some filtering.
            - 2) It prints some useful informations in the terminal. 
            - 3) It plots 3 curves, one for the surround at the preffered orientation, one for the surround at the orthogonal orientation and one for the center alone
            - 4) It does 3) for 3 random neurons in the neuron ids left after filtering

        Prerequisite :
            
            - function 'center_contrast_surround_suppression_experiment' executed for the required neurons

        Filtering :
        
            - Exclude the neurons with receptive fields poorly fitted to a Gaussian model. function 'filter_fitting_error'
            - Exclude the neurons that appeard to have no surround suppression nor response saturation. function 'filter_no_supp_neurons'
    '''
    
    group_path = "/center_contrast_surround_suppression/results"

    ## Check if the 'get_preferred_position' function has been computed
    check_group_exists_error(h5_file=h5_file, group_path=group_path)
    
    # ## Check if the neurons are present in the data
    check_neurons_presence_error(h5_file=h5_file, list_group_path=[group_path], neuron_ids=neuron_ids)

    ## Filter on fitting error
    filtered_neuron_ids = filter_fitting_error(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh, print_results=False)

    ## Filter the neurons with no surround suppression and no response saturation
    filtered_neuron_ids = filter_no_supp_neurons(h5_file=h5_file, neuron_ids=filtered_neuron_ids, print_results = False)
    
    n = len(neuron_ids)
    n_new = len(filtered_neuron_ids)

    ## Initialise the arrays to fill
    with h5py.File(h5_file, 'r') as f :
        ## Get the parameters used in the experiment
        group_args_str = f[group_path].attrs["arguments"]
        group_dico_args = get_arguments_from_str(group_args_str)
        center_contrasts_ccss = group_dico_args["center_contrasts_ccss"]
        ## Convert str to a list
        center_contrasts_ccss = get_list_from_str(center_contrasts_ccss)

        curves_iso      = np.zeros((n_new,len(center_contrasts_ccss)))
        curves_ortho    = np.zeros((n_new,len(center_contrasts_ccss)))
        curves_center   = np.zeros((n_new,len(center_contrasts_ccss)))

    ## Get the curves for every neuron
    with h5py.File(h5_file, 'r') as f :
        for i, neuron_id in enumerate(filtered_neuron_ids) :
            neuron = f"neuron_{neuron_id}"

            results_neuron = f[group_path][neuron][:]

            curve_iso    = results_neuron[:,0]
            curve_ortho  = results_neuron[:,1]
            curve_center = results_neuron[:,2]

            curves_iso[i]    = curve_iso
            curves_ortho[i]  = curve_ortho
            curves_center[i] = curve_center

    mean_curve_iso    = np.mean(curves_iso, axis=0)
    mean_curve_ortho  = np.mean(curves_ortho, axis=0)
    mean_curve_center = np.mean(curves_center, axis=0)

    print("--------------------------------------")
    print("Visualisation of ccss curves for random neurons :")
    print(f"    > Analysis made on {n} neurons")
    print(f"    > {n_new} neurons ({round((n_new/n*100), 2)}%) left after filtration")
    print()
    print(f"    > The mean curves accross every neuron :")

    plt.plot(center_contrasts_ccss,mean_curve_center, label = "Center alone", color = 'gold')
    plt.plot(center_contrasts_ccss,mean_curve_ortho, label = "Surround ortho-oriented", color = 'goldenrod')
    plt.plot(center_contrasts_ccss,mean_curve_iso, label = "Surround iso-oriented", color = 'darkgoldenrod' )
    
    plt.xscale("log")
    plt.xticks(center_contrasts_ccss)
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    
    plt.title(f"Mean CCSS Curves")
    plt.xlabel("Contrast of the center")
    plt.ylabel("Response")
    plt.legend()

    plt.show()

    
    print()
    print(f"    > Random neurons :")

    for i in range(3) : 

        ## Select a random neuron
        r = np.random.randint(0,n_new)
        neuron_id = filtered_neuron_ids[r]

        plot_ccss_curves(h5_file=h5_file, neuron_id=neuron_id)

    print("--------------------------------------")
    print()

def ccss_results_2(
    h5_file,
    neuron_ids,
    contrast_id,
    norm_center_contrast_id,
    fit_err_thresh = 0.2    
):
    '''
    This function aims to visualise the distribution of response to surround iso and ortho-oriented to the center (the center is at the preferred orientation)
            
            - 1) It performs some filtering.
            - 2) It prints some useful informations in the terminal. 
            - 3) It plots a scatter plot to compare the normalised response for both conditions

        NB : The normalised response is done by dividing the response by the response to the center alone
        Prerequisite :
            
            - function 'center_contrast_surround_suppression_experiment' executed for the required neurons

        Filtering :
        
            - Exclude the neurons with receptive fields poorly fitted to a Gaussian model. function 'filter_fitting_error'
            - Exclude the neurons that appeard to have no surround suppression nor response saturation. function 'filter_no_supp_neurons'

        Arguments :

            - contrast_id             : The id of the contrast (corresponding to the id in the array center_contrasts_ccss used in the ccss experiment) to visualise
            - norm_center_contrast_id : The id of the center contrast that will serve for the normalisation  (corresponding to the id in the array center_contrasts_ccss used in the ccss experiment) 
    '''
    
    subgroup_path = '/center_contrast_surround_suppression/results'

    ## Check if the experiment has been run
    check_group_exists_error(h5_file=h5_file, group_path=subgroup_path)
    
    # ## Check if the neurons are present in the data
    check_neurons_presence_error(h5_file=h5_file, list_group_path=[subgroup_path], neuron_ids=neuron_ids)

    ## Filter on fitting error
    filtered_neuron_ids = filter_fitting_error(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh, print_results=False)

    ## Filter the neurons with no surround suppression and no response saturation
    filtered_neuron_ids = filter_no_supp_neurons(h5_file=h5_file, neuron_ids=filtered_neuron_ids, print_results = False)
    
    all_iso   = []
    all_ortho = []
    ## Get the data 
    with h5py.File(h5_file, 'r') as f :

        for neuron_id in filtered_neuron_ids :
            
            neuron = f"neuron_{neuron_id}"

            results_neuron = f[subgroup_path][neuron][:]

            resp_iso    = results_neuron[contrast_id,0]
            resp_ortho  = results_neuron[contrast_id,1]
            resp_center = results_neuron[norm_center_contrast_id,2]
            norm_resp_iso   = resp_iso / resp_center
            norm_resp_ortho = resp_ortho / resp_center
            all_iso.append(norm_resp_iso)
            all_ortho.append(norm_resp_ortho)
        all_iso   = np.array(all_iso)
        all_ortho = np.array(all_ortho)

        ## Get the center contrast that correspond to contrast_id
        group = f[subgroup_path]
        group_args_str = group.attrs["arguments"]
        group_dico_args = get_arguments_from_str(group_args_str)
        center_contrasts_ccss = group_dico_args["center_contrasts_ccss"]
        center_contrasts_ccss = get_list_from_str(center_contrasts_ccss)
        center_contrast = str(round(center_contrasts_ccss[contrast_id], 2))

    n = len(neuron_ids)
    n_new = len(filtered_neuron_ids)

    print("--------------------------------------")
    print(f"Visualisation of the responses to ortho and iso-oriented surround (center contrast = {center_contrast}):")
    print(f"    > Analysis made on {n} neurons")
    print(f"    > {n_new} neurons ({round((n_new/n*100), 2)}%) left after filtration")
    print(f"    > The mean response to the ortho-oriented surround is {str(round(np.mean(all_ortho),3))}")
    print(f"    > The mean response to the iso-oriented surround is {str(round(np.mean(all_iso),3))}")
    print(f"    > Scatter Plot :")

    plot_scatter_hist(
    x=all_ortho,
    y=all_iso,
    title=f'Distribution of the response depending on the surround orientation\nContrast of the center : {center_contrast}',
    x_label='Normalised response for the \n orthogonal surround',
    y_label='Normalised response for the \n parallel surround',
    log_axes=False)
    print("--------------------------------------")


#%%
# TODO DELETE  
from surroundmodulation.utils.misc import pickleread

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

overwrite = False
device = None
ori_shifts = np.linspace(-np.pi,np.pi,9)
center_contrasts_ccss = np.array([0.06,0.12,0.25,0.5,1.0])
fit_err_thresh = 0.2

contrast_id = 1
norm_center_contrast_id = contrast_id

# ccss_results_1(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh)
ccss_results_2(h5_file=h5_file, neuron_ids=neuron_ids, norm_center_contrast_id=norm_center_contrast_id, contrast_id=contrast_id, fit_err_thresh=fit_err_thresh)
#%%

# ## Name of the HDF5 file 
# h5_file = "/project/mathys_code/article1_ring.h5"

# ## Select the model of all neurons
# all_neurons_model = v1_convnext_ensemble

# ## Chose the number of neurons to work with
# n = 458
# neuron_ids = np.arange(n) ## Because 458 outputs in our model

# ## Parameters to test
# orientations = np.linspace(0, np.pi, 37)[:-1] 
# spatial_frequencies = np.linspace(1, 7, 25) 
# phases = np.linspace(0, 2*np.pi, 37)[:-1]
# radii = np.logspace(-2,np.log10(2),40)

# ## Parameters for the dot stimulation
# dot_size_in_pixels = 4
# num_dots=200000
# bs = 40
# seed = 0

# ## Fixed image parameters
# contrast = 1         
# img_res = [93,93] 
# pixel_min = -1.7876 # (the lowest value encountered in the data that served to train the model, serves as the black reference)
# pixel_max =  2.1919
# size = 2.67 #2.35

# ## Allow negative responses (because the response we show is the difference between the actual response and the response to a gray screen)
# neg_val = True

# ## Analysis1 arguments
# radii = radii

# ## Analysis2 arguments
# center_contrasts = np.logspace(np.log10(0.06),np.log10(1),18) #np.logspace(-2,np.log10(1),18) 
# surround_contrasts = np.logspace(np.log10(0.06),np.log10(1),6) #np.logspace(-2,np.log10(1),6) 

# ## Analysis3 arguments
# contrasts = np.logspace(np.log10(0.06),np.log10(1),5) 

# ##
# h5_file = "/project/mathys_code/test.h5"
# all_neurons_model = v1_convnext_ensemble
# device = None

# ## Parameters to test
# orientations = np.linspace(0, np.pi, 5)[:-1] 
# spatial_frequencies = np.linspace(1, 7, 5) 
# phases = np.linspace(0, 2*np.pi, 5)[:-1]
# radii = np.logspace(-2,np.log10(2),5)

# ## Parameters for the dot stimulation
# dot_size_in_pixels = 4
# num_dots=20000
# bs = 40
# seed = 0

# ## Fixed image parameters
# contrast = 1         
# img_res = [93,93] 
# pixel_min = -1.7876 # (the lowest value encountered in the data that served to train the model, serves as the black reference)
# pixel_max =  2.1919
# size = 2.67 #2.35

# ## Allow negative responses (because the response we show is the difference between the actual response and the response to a gray screen)
# neg_val = True

# ## Analysis1 arguments
# radii = radii

# ## Analysis2 arguments
# center_contrasts = np.logspace(np.log10(0.06),np.log10(1),5) #np.logspace(-2,np.log10(1),18) 
# surround_contrasts = np.logspace(np.log10(0.06),np.log10(1),5) #np.logspace(-2,np.log10(1),6) 

# ## Analysis3 arguments
# contrasts = np.logspace(np.log10(0.06),np.log10(1),5)  ## Same as in the article

# ## Chose the number of neurons to work with
# neuron_ids = np.arange(458) 


#%%

# overwrite = False
# device = None
# ## Get the preferred full field gratings parameters for the neuron (orientation, spatial frequency, full field phase)
# get_all_grating_parameters_fast(h5_file=h5_file, all_neurons_model=all_neurons_model, neuron_ids=neuron_ids, overwrite=overwrite, orientations = orientations, spatial_frequencies = spatial_frequencies, phases = phases, contrast = contrast, img_res = img_res, pixel_min = pixel_min, pixel_max = pixel_max, device = device, size = size )

# ## Find the preferred stimulus position by finding the center of the gaussian model fitted to the excitatory pixels for every neuron
# get_preferred_position(h5_file=h5_file, all_neurons_model=all_neurons_model, neuron_ids=neuron_ids, overwrite=overwrite, dot_size_in_pixels = dot_size_in_pixels, num_dots = num_dots, contrast = contrast, img_res = img_res, pixel_min = pixel_min, pixel_max =  pixel_max, device = device, bs = bs, seed = seed)

# ## Perform size tuning experiment
# size_tuning_experiment_all_phases(h5_file=h5_file, all_neurons_model=all_neurons_model, neuron_ids=neuron_ids, overwrite=overwrite, radii=radii, phases=phases, contrast=contrast, pixel_min=pixel_min, pixel_max=pixel_max, device=device, size=size, img_res=img_res, neg_val=neg_val)

# ## Perform contrast response experiment (15min)
# contrast_response_experiment(h5_file=h5_file, all_neurons_model=all_neurons_model, neuron_ids=neuron_ids, overwrite=overwrite, center_contrasts=center_contrasts, surround_contrasts=surround_contrasts, pixel_min=pixel_min, pixel_max=pixel_max, device=device, size=size, img_res=img_res, neg_val=neg_val)

# ## Perform contrast size tuning experiment 
# contrast_size_tuning_experiment_all_phases(h5_file=h5_file, all_neurons_model=all_neurons_model, neuron_ids=neuron_ids, overwrite=overwrite, phases=phases, contrasts=contrasts, radii=radii, pixel_min=pixel_min, pixel_max=pixel_max, device=device, size=size, img_res=img_res, neg_val=neg_val)

# %%

