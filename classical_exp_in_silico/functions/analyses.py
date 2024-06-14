#################################################################################
#####    THIS CODE CONTAINS EVERY FUNCTION USED TO PERFORM THE ANALYSES.    #####
#####             ------------------------------------------------          #####
#####           Those functions go get the results of the experiments       #####
#####         in the HDF5 file and perform the analyses on these results    #####
#################################################################################

############################
##### PART 0 : Imports #####
############################

## Useful
import numpy as np
import torch
import math
## Utils
from utils import *
from experiments import get_GSF_surround_AMRF
## Plots
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from surroundmodulation.utils.plot_utils import plot_img
## Data storage
import h5py



########################################
##### PART I : Filtering functions #####
########################################


def filter_fitting_error(
    h5_file, 
    neuron_ids,
    fit_err_thresh = 0.2,
    print_results = False
):
    ''' The goal of this function is to return only the neurons with a center that fits well to a Gaussian model.
        The function goes in the h5_file, searches for the '/preferred_pos' group and selects the neurons with an error below 
        the threshold. It will also exclude the neurons with error = np.nan

        Prerequisite :
            
            - function 'get_preferred_position' executed for the required neurons    

        Arguments : 

            - fit_err_thresh : The threshold of fitting error, a value above this will lead to the neuron to be excluded
            - print_results   : (Optional) If set to 'True', it will print the amount of neurons excluded in the terminal

        Outputs :

            - filtered_neuron_ids : An array containing the neurons that are well fitted
    '''
    
    group_path = '/preferred_pos'

    ## Check if the 'get_preferred_position' function has been computed
    check_group_exists_error(h5_file=h5_file, group_path=group_path)
    
    ## Check if the neurons are present in the data
    check_neurons_presence_error(h5_file=h5_file, list_group_path=[group_path], neuron_ids=neuron_ids)

    filtered_neuron_ids = []

    with h5py.File(h5_file,  'r') as file :

        for neuron_id in neuron_ids :
            
            ## name of the dataset 
            neuron = f"neuron_{neuron_id}"
            error = file[group_path][neuron][:][2]

            ## keep the neuron if its error is below the threshold
            if error < fit_err_thresh :
                filtered_neuron_ids.append(neuron_id)

    if print_results :

        n = len(neuron_ids)
        new_n = len(filtered_neuron_ids)

        ## Show the results
        print("--------------------------------------")
        print("Filtering the Neurons poorly fitted :")
        print(f"    > There were initially {n} neurons")
        print(f"    > {n-new_n} neurons were removed ({round(100*((n-new_n)/n),1)}%)")
        print(f"    > There is {new_n} neurons left")
        print("--------------------------------------")
        print()
        
    return np.array(filtered_neuron_ids)
    

def filter_low_supp_neurons(
    h5_file, 
    neuron_ids,
    supp_thresh = 0.1,
    print_results = False
    ):

    ''' The goal of this function is to select only the neurons with a decent surround suppression.
        The function goes in the h5_file, searches for the '/size_tuning/results' group and selects the neurons with 
        a suppression index above the threshold.

        Prerequisite :
            
            - function 'size_tuning_experiment_all_phases' executed for the required neurons

        Arguments :

            - supp_thresh      : Threshold, neurons with a suppression index below that value will be excluded
            - print_results    : (Optional) If set to 'True', it will print the amount of neurons excluded in the terminal

        Outputs :

            - filtered_neuron_ids : An array containing the neurons that have a decent suppression index

    '''

    group_path = '/size_tuning/results'

    ## Check if the 'size_tuning_experiment_all_phases' function has been computed
    check_group_exists_error(h5_file=h5_file, group_path=group_path)
    
    ## Check if the neurons are present in the data
    check_neurons_presence_error(h5_file=h5_file, list_group_path=[group_path], neuron_ids=neuron_ids)

    filtered_neuron_ids = []

    with h5py.File(h5_file,  'r') as file :

        for neuron_id in neuron_ids :
            
            ## name of the dataset 
            neuron = f"neuron_{neuron_id}"
            SI = file[group_path][neuron][:][4]

            ## keep the neuron if its error is below the threshold
            if SI > supp_thresh :
                filtered_neuron_ids.append(neuron_id)

    if print_results :

        n = len(neuron_ids)
        new_n = len(filtered_neuron_ids)

        ## Show the results
        print("--------------------------------------")
        print("Filtering the Neurons with low suppression :")
        print(f"    > There were initially {n} neurons")
        print(f"    > {n-new_n} neurons were removed ({round(100*((n-new_n)/n),1)}%)")
        print(f"    > There is {new_n} neurons left")
        print("--------------------------------------")
        print()
        
    return np.array(filtered_neuron_ids)


def filter_no_supp_neurons(
    h5_file, 
    neuron_ids,
    print_results = False
    ):
    ''' This function excludes unwanted neurons from our analyses.
        The unwanted neurons are neurons whith no surround suppression nor response saturation. (basically those are the neurons for which increasing the size of the stimulus will always lead to a greater response)
        The function goes in the h5_file, searches for the '/size_tuning/results' group and selects the neurons as so : 
            - It excludes neurons that have a negative Suppression Index
            - Having negative (or null) suppression index means that their is neither response saturation nor suppression

        Prerequisite :
            
            - function 'size_tuning_experiment_all_phases' executed for the required neurons

        Arguments :

            - see filter_low_supp_neurons     

        Outputs :

            - filtered_neuron_ids : An array containing the selected neurons
    '''
    ## Exclude neurons with a negative suppression index
    filtered_neuron_ids = filter_low_supp_neurons(h5_file=h5_file, neuron_ids=neuron_ids, supp_thresh=0.0001, print_results = False)

    filtered_neuron_ids = np.array(filtered_neuron_ids)
    
    if print_results :

        n = len(neuron_ids)
        new_n = len(filtered_neuron_ids)

        ## Show the results
        print("--------------------------------------")
        print("Filtering the Neurons with no suppression nor response saturation:")
        print(f"    > There were initially {n} neurons")
        print(f"    > {n-new_n} neurons were removed ({round(100*((n-new_n)/n),1)}%)")
        print(f"    > There is {new_n} neurons left")
        print("--------------------------------------")
        print()
        
    return np.array(filtered_neuron_ids)


def filter_SNR(
    h5_file, 
    neuron_ids,
    SNR_thresh = 2,
    print_results = False
):
    ''' This function is used in the third article only
        The goal of this function is to return only the neurons with a significant response to black or white dot stimulus
        The function goes in the h5_file, searches for the '/black_white_preference' group and selects the neurons with either SNRb or SNRw > SNR_thresh 

        Prerequisite :
            
            - function 'black_white_preference_experiment' executed for the required neurons    

        Arguments : 

            - SNR_thresh      : The threshold of SNR value, a value below this for both SNRb and SNRw will lead to the neuron to be excluded
            - print_results   : (Optional) If set to 'True', it will print the amount of neurons excluded in the terminal

        Outputs :

            - filtered_neuron_ids : An array containing the neurons that were kept

    '''

    group_path    = '/black_white_preference'
    subgroup_path = group_path + '/results'

    ## Check if the 'get_preferred_position' function has been computed
    check_group_exists_error(h5_file=h5_file, group_path=subgroup_path)
    
    ## Check if the neurons are present in the data
    check_neurons_presence_error(h5_file=h5_file, list_group_path=[subgroup_path], neuron_ids=neuron_ids)

    filtered_neuron_ids = []

    with h5py.File(h5_file,  'r') as file :
        
        subgroup = file[subgroup_path]
        for neuron_id in neuron_ids :
            
            ## name of the dataset 
            neuron = f"neuron_{neuron_id}"

            ## Get the SNR
            SNRb = subgroup[neuron][:][0]
            SNRw = subgroup[neuron][:][1]

            ## keep the neuron if its error is below the threshold
            if SNRb > SNR_thresh or SNRw > SNR_thresh :
                filtered_neuron_ids.append(neuron_id)

    if print_results :

        n = len(neuron_ids)
        new_n = len(filtered_neuron_ids)

        ## Show the results
        print("--------------------------------------")
        print("Filtering the Neurons with low SNR :")
        print(f"    > There were initially {n} neurons")
        print(f"    > {n-new_n} neurons were removed ({round(100*((n-new_n)/n),1)}%)")
        print(f"    > There is {new_n} neurons left")
        print("--------------------------------------")
        print()
        
    return np.array(filtered_neuron_ids)



##############################################################################
#####  PART II : Analyses for the first article, Cavanaugh et al., 2002  #####
#####   --------------------------------------------------------------   #####
#####        Nature and Interaction of Signals From the Receptive        #####
#####          Field Center and Surround in Macaque V1 Neurons           #####
#####   --------------------------------------------------------------   #####
#####              DOI : https://doi.org/10.1152/jn.00692.2001           #####
##################################################################################
                        

def plot_scatter_hist(
    x,
    y,
    title,
    x_label,
    y_label,
    log_axes = True
    ):

    ''' This function aims to create a scatter plot with two histograms on its sides
        It is used to compare the GSF diameter and the surround diameter

        Arguments :

            - x, y              : The arrays to compare
            - title             : The title on the plot
            - x_label, y_label  : The labels on the plot
            - log_axes          : If set to True, will plot with logarithmic axes
    '''

    ## Start with a square Figure.
    fig = plt.figure(figsize=(6, 6))

    ## Name the fig
    plt.suptitle(title)

    ## Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    ## the size of the marginal axes and the main axes in both directions.
    ## Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
    
    
    ## Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    ## Draw the scatter plot and the marginals
    ## The scatter plot:
    ax.scatter(x, y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if log_axes :
        ax.set_xscale("log")
        ax.set_yscale("log")

    # Set formatters
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())

    # Set limits
    
    xymin = min(np.min(np.abs(x)), np.min(np.abs(y)))
    minedge = max(0.06,xymin - 0.5) 
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    maxedge = xymax +1
    ax.set_xlim(minedge, maxedge)
    ax.set_ylim(minedge, maxedge)

    # Plot diagonal line
    ax.plot([minedge, maxedge], [minedge, maxedge], 'k--')

    ## The histograms
    nbins = 15

    ## To show proportions 
    weights_x = np.ones(x.shape) / len(x)
    weights_y = np.ones(y.shape) / len(y)

    if log_axes :
        ax_histx.hist(x, density = False, weights = weights_x, bins = np.logspace(np.log10(minedge), np.log10(maxedge),nbins))
        ax_histy.hist(y, density = False, weights = weights_y, orientation='horizontal', bins = np.logspace(np.log10(minedge), np.log10(maxedge),nbins))
    else :
        ax_histx.hist(x, density = False, weights = weights_x, bins = np.linspace(minedge, maxedge,nbins))
        ax_histy.hist(y, density = False, weights = weights_y, orientation='horizontal', bins = np.linspace(minedge, maxedge,nbins))
    
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    
    plt.show()


def plot_size_tuning_curve(
    h5_file,
    neuron_id
):
    ''' This function goes into a HDF5 file, get the size tuning curves for one neuron and plot them.

        Prerequisite :

            - function 'size_tuning_experiment_all_phases' executed for the required neuron

        NB  : The plot x axis shows diameters, not radii    
     
    '''
    group_size_tuning_path = "/size_tuning"
    subgroup_tuning_curves_path = group_size_tuning_path + "/curves"

    ## Check if the size tuning experiment has been performed
    check_group_exists_error(h5_file, group_size_tuning_path)

    ## Check if the neurons are present in the data
    check_neurons_presence_error(h5_file, [subgroup_tuning_curves_path], [neuron_id])

    with h5py.File(h5_file, 'r') as file :

        ## Get the parameters used in the size tuning experiment
        group = file[group_size_tuning_path]
        group_args_str = group.attrs["arguments"]
        group_dico_args = get_arguments_from_str(group_args_str)
        radii = group_dico_args["radii"]
        ## Convert str to a list
        radii = get_list_from_str(radii)

        ## Get the tuning curves
        neuron = f"neuron_{neuron_id}"
        subgroup = file[subgroup_tuning_curves_path]
        both_curves = subgroup[neuron][:]
        
        circular_tuning_curve = both_curves[0]
        annular_tuning_curve  = both_curves[1]

    
    ## Multiply the radii to get the diameters
    plt.plot(radii*2,circular_tuning_curve, label = "center")
    plt.plot(radii*2, annular_tuning_curve, label = "surround")
    plt.xlabel("Diameter (deg)")
    plt.ylabel('Response')
    plt.legend()
    plt.title(f"Size tuning curve of the Neuron {neuron_id}")
    plt.show()


def plot_contrast_response(
    h5_file,
    neuron_id,
    title = None
):

    ''' This function goes into a HDF5 file, get the contrast response curves for one neuron and plot them.
        
        NB :    yaxis  = Response of the single neuron modele , 
                xaxis  = Contrast of the center
                curves = One curve for each surround contrast

        Prerequisite :

            - function 'contrast_response_experiment' executed for the required neuron    
        
        Arguments :

            - title : (Optional) str : The customized title for the plot. If set to 'None', it will set a default title
    '''

    group_path_cr = "/contrast_response"
    group_path_cr_curves = group_path_cr + "/curves"

    ## Check if the contrast response experiment has been performed
    check_group_exists_error(h5_file, group_path_cr_curves)

    ## Check if the neuron is present in the data
    check_neurons_presence_error(h5_file, [group_path_cr_curves], [neuron_id])

    with h5py.File(h5_file, 'r') as file :

        ## Get the parameters used in the size tuning experiment
        group = file[group_path_cr]
        group_args_str = group.attrs["arguments"]
        group_dico_args = get_arguments_from_str(group_args_str)
        center_contrasts = group_dico_args["center_contrasts"]
        surround_contrasts = group_dico_args["surround_contrasts"]
        ## Convert str to a list
        center_contrasts = get_list_from_str(center_contrasts)
        surround_contrasts = get_list_from_str(surround_contrasts)
    
        ## Get the curves
        neuron = f"neuron_{neuron_id}"
        subgroup = file[group_path_cr_curves]
        contrast_response_curves = subgroup[neuron][:]


    ## plot each curve
    for i in range(len(contrast_response_curves)) :
        label = round(surround_contrasts[i],2)
        plt.plot(center_contrasts,contrast_response_curves[i], label = str(label))

    ## Set the title if no title was given
    if title is None :
        title = f"Response contrast of the neuron {neuron_id}"

    plt.legend(title='Surround contrast', title_fontsize='large')
    plt.title(title)
    plt.xlabel("Center contrast")
    plt.ylabel("Response")
    plt.xscale('log')

    plt.show()


def plot_contrast_size_tuning_curve(
    h5_file,
    neuron_id,
    title = None
    ):

    ''' This function goes into a HDF5 file, get the contrast size tuning response curves for one neuron and plot them.

        Prerequisite :

            - function 'contrast_size_tuning_experiment_all_phases' executed for the required neuron    
    
        NB : yaxis  = Response of the single neuron modele , 
             xaxis  = diameter (radius * 2)
             curves = One curve for each contrast

    '''

    group_path_cst = "/contrast_size_tuning"
    group_path_cst_curves = group_path_cst + "/curves"

    ## Check if the contrast response experiment has been performed
    check_group_exists_error(h5_file, group_path_cst_curves)

    ## Check if the neuron is present in the data
    check_neurons_presence_error(h5_file, [group_path_cst_curves], [neuron_id])

    with h5py.File(h5_file, 'r') as file :

        ## Get the parameters used in the size tuning experiment
        group = file[group_path_cst]
        group_args_str = group.attrs["arguments"]
        group_dico_args = get_arguments_from_str(group_args_str)
        contrasts = group_dico_args["contrasts"]
        radii = group_dico_args["radii"]
        ## Convert str to a list
        contrasts = get_list_from_str(contrasts)
        radii = get_list_from_str(radii)
    
        ## Get the curves
        neuron = f"neuron_{neuron_id}"
        subgroup = file[group_path_cst_curves]
        cst_curves = subgroup[neuron][:]

    ## Reverse the order for better visualisation
    for i, contrast in enumerate(contrasts[::-1]) :
        label = round(contrast,2)
        plt.plot(radii*2,cst_curves[- (i+1)], label = str(label) )
        plt.xlabel("diameter (deg)")
        plt.ylabel('response')

    ## Set the title if no title was given
    if title is None :
        title = f"Contrast Size Tuning Curves of the neuron {neuron_id}"


    plt.legend(title='Contrast', title_fontsize='large')
    plt.title(title)
    plt.show()
        

def size_tuning_results_1(  
    h5_file,
    neuron_ids,
    fit_err_thresh = 0.2,
    supp_thresh = 0.1
    ):

    ''' This function aims to visualise the relation between the GSF diameter and the surround extent diameter :
            
            - 1) It performs some filtering.
            - 2) It prints some useful informations in the terminal. (number of neurons filtered, mean GSF ...)
            - 3) It plots a scatter plot of the values, with histograms showing the distribution,

        Prerequisite :
            
            - function 'size_tuning_experiment_all_phases' executed for the required neurons

        Filtering :
        
            - Exclude the neurons with receptive fields poorly fitted to a Gaussian model. function 'filter_fitting_error'
            - Exclude the neurons that appeard to have no surround suppression nor response saturation. function 'filter_no_supp_neurons'
            - Exclude the neurons that have a suppression index too low. function 'filter_low_supp_neurons'

        Arguments :

            - fit_err_thresh : Fitting error threshold, exclude every neuron with an error above that value
            - supp_thresh    : Suppression index threshold, exclude every neuron with a SI below that value
    '''
    
    ## These functions verify if the prerequisite functions are performed for the requested neurons
    ## Filter on fitting error
    filtered_neuron_ids = filter_fitting_error(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh, print_results=False)

    ## Filter the neurons with no surround suppression and no response saturation
    filtered_neuron_ids = filter_no_supp_neurons(h5_file=h5_file, neuron_ids=filtered_neuron_ids, print_results = False)

    ## Filter the neurons with a surround suppression too low
    filtered_neuron_ids = filter_low_supp_neurons(h5_file=h5_file, neuron_ids=filtered_neuron_ids, supp_thresh=supp_thresh, print_results = False)

    group_path = "/size_tuning/results"

    with h5py.File(h5_file, 'r') as f :

        ## For each neuron, get the GSF and the surround extent
        all_GSF = []
        all_surr_ext = []

        for neuron_id in filtered_neuron_ids :

            neuron = f"neuron_{neuron_id}"

            GSF      = f[group_path][neuron][:][1]
            surr_ext = f[group_path][neuron][:][2]

            all_GSF.append(GSF)
            all_surr_ext.append(surr_ext)

    all_GSF = np.array(all_GSF)
    all_surr_ext = np.array(all_surr_ext)
    
    ## Multiply by 2 to get diameters
    all_GSF *= 2
    all_surr_ext *= 2
    
    ## Informations to print
    n = len(neuron_ids)
    n_new = len(all_GSF)
    mean_GSF = round(np.mean(all_GSF),3)
    mean_surr_ext = round(np.mean(all_surr_ext),2)
    mean_ratio    = round(np.mean(all_surr_ext/all_GSF),2)

    ## Parameters for the plot
    title = f"GSF diameter vs surround extent diameter for {n_new} neurons"
    x_label = "GSF diameter (deg)"
    y_label = "surround diameter (deg)"

    ## Show the results
    print("--------------------------------------")
    print("Comparison of the GSF and surround extent diameter :")
    print(f"    > Analysis made on {n} neurons")
    print(f"    > {n_new} neurons ({round((n_new/n*100), 2)}%) left after filtration")
    print(f"    > The mean diameter of the GSF is {mean_GSF}")
    print(f"    > The mean diameter of the surround_extent is {mean_surr_ext}")
    print(f"    > The mean ratio between the surround and the GSF is {mean_ratio}")
    print(f"    > Plot :")
    plot_scatter_hist(x=all_GSF, y=all_surr_ext,title=title, x_label=x_label, y_label=y_label)
    print("--------------------------------------")
    print()


    return
    

def size_tuning_results_2(
    h5_file,
    neuron_ids,
    fit_err_thresh = 0.2
    ):
    ''' This function aims to visualize the distribution of the suppression index in the requested neuron set :
            
            - 1) It performs some filtering.
            - 2) It prints some useful informations in the terminal. (number of neurons filtered, mean SI ...)
            - 3) It plots an histogram of the neuron SI values

        Prerequisite :
            
            - function 'size_tuning_experiment_all_phases' executed for the required neurons

        Filtering :
        
            - Exclude the neurons with receptive fields poorly fitted to a Gaussian model. function 'filter_fitting_error'
            - Exclude the neurons that appeard to have no surround suppression nor response saturation. function 'filter_no_supp_neurons'

        Arguments :

            - see 'size_tuning_results_1'
    '''
    
    ## These functions verify if the prerequisite functions are performed for the requested neurons
    ## Filter on fitting error
    filtered_neuron_ids = filter_fitting_error(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh, print_results=False)

    ## Filter the neurons with no surround suppression and no response saturation
    filtered_neuron_ids = filter_no_supp_neurons(h5_file=h5_file, neuron_ids=filtered_neuron_ids, print_results = False)

    group_path = "/size_tuning/results"

    with h5py.File(h5_file, 'r') as f :

        ## For each neuron, get the GSF and the surround extent
        all_SI = []

        for neuron_id in filtered_neuron_ids :

            neuron = f"neuron_{neuron_id}"

            SI = f[group_path][neuron][:][-1]

            all_SI.append(SI)
    
    all_SI = np.array(all_SI)
    n = len(neuron_ids)
    n_new = len(all_SI)
    mean_SI = round(np.mean(all_SI), 2)
    max_SI = round(max(all_SI),1)
    bins = np.linspace(0,max(max_SI,1),6)

    ## Show the results
    print("--------------------------------------")
    print("Distribution of the Suppression Index :")
    print(f"    > Analysis made on {n} neurons")
    print(f"    > {n_new} neurons ({round((n_new/n*100), 2)}%) left after filtration")
    print(f"    > The mean value of the SI is {mean_SI}")
    print(f"    > Plot :")

    weights = np.ones(all_SI.shape) / len(all_SI)
    plt.hist(all_SI, bins = bins,edgecolor='black', density=False, weights=weights)
    plt.xlabel("Suppression Index (SI)" )
    plt.ylabel('Distribution')
    plt.xticks(bins)
    plt.title(f"Distribution of the SI for the {len(all_SI)} neurons")

    plt.show()

    print("--------------------------------------")
    print()


def sort_by_spread(
    h5_file,
    neuron_ids,
    sort_by_std = False
):
    ''' This function sorts the neurons from the ones with the lowest spread values to the ones with the highest.  
        Spread is assessed thanks to the mean_std or maxmin_ratio values computed in the 'contrast_response_experiment' function

        Prerequisite :
            
            - function 'contrast_response_experiment' executed for the required neurons

        Arguments :

            - sort_by_std   : If set to true, sort with the mean_std value, if set to false, sort with max/min value.

        Outputs :

            - sorted_neuron_ids    : The sorted neuron ids
            - sorted_spread        : The sorted spread values (either std or maxmin)
    '''
    group_path = "/contrast_response/results"

    ## Check if the 'contrast_response_experiment' function has been computed
    check_group_exists_error(h5_file=h5_file, group_path=group_path)
    
    ## Check if the neurons are present in the data
    check_neurons_presence_error(h5_file=h5_file, list_group_path=[group_path], neuron_ids=neuron_ids)

    sorted_neuron_ids = np.copy(neuron_ids)

    with h5py.File(h5_file, 'r') as f :

        if sort_by_std :
            
            all_std = []

            for neuron_id in sorted_neuron_ids : 

                neuron = f"neuron_{neuron_id}"

                ## Get the neurons mean_std
                mean_std = f[group_path][neuron][:][1]
                all_std.append(mean_std)

            ## Sort accross mean_std
            order = np.argsort(all_std)

            ## Order the spread values
            sorted_spread = np.array(all_std)[order]

        else :

            all_maxmin = []

            for neuron_id in sorted_neuron_ids : 

                neuron = f"neuron_{neuron_id}"

                ## Get the neurons mean_std
                maxmin = f[group_path][neuron][:][0]
                all_maxmin.append(maxmin)

            ## Sort accross max/min
            order = np.argsort(all_maxmin)

            ## Order the spread values
            sorted_spread = np.array(all_maxmin)[order]

    ## Order the neuron_ids
    sorted_neuron_ids = sorted_neuron_ids[order]

    return sorted_neuron_ids, sorted_spread


def contrast_response_results_1(
    h5_file,
    neuron_ids,
    fit_err_thresh = 0.2,
    sort_by_std = False,
    spread_to_plot = [15, 50, 85]
    ):
    ''' This function aims to visualize some representative contrast response curves for the requested neuron set :
            
            - 1) It performs some filtering.
            - 2) It prints some useful informations in the terminal. (number of neurons filtered, mean spread ...)
            - 3) It sorts the array of neurons according to the how spread their curves are
            - 4) It plots the contrast response curves for neurons representing different contrasts

        To estimate how spread the curves are for a neuron, there are two possibilities :

            - Either take the last points of the curves and calculate the ratio : max_response/min_response
            - Either compute the mean standard deviation for every points

        Prerequisite :
        
            - function 'size_tuning_experiment_all_phases' executed for the required neurons
            - function 'contrast_response_experiment' executed for the required neurons

        Filtering :
        
            - Exclude the neurons with receptive fields poorly fitted to a Gaussian model. function 'filter_fitting_error'
            - Exclude the neurons that appeard to have no surround suppression nor response saturation. function 'filter_no_supp_neurons'

        Arguments :

            - sort_by_std    : Bool, decides which method to use to estimate how the curves are spread. If set to 'True', it will use the second method and sort with the mean_std value
            - spread_to_plot : An array containing the position of the neuron in the sorted spread array (in percentage). 100 means that it's the neuron with the highest spread, 50 means that 50% of the neurons have a smaller spread.
            - other          : see 'size_tuning_results_1'
    '''
     
    ## These functions verify if the prerequisite functions are performed for the requested neurons
    ## Filter on fitting error
    filtered_neuron_ids = filter_fitting_error(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh, print_results=False)

    ## Filter the neurons with no surround suppression and no response saturation
    filtered_neuron_ids = filter_no_supp_neurons(h5_file=h5_file, neuron_ids=filtered_neuron_ids, print_results = False)

    ## Sort the neurons
    sorted_neuron_ids, sorted_spread = sort_by_spread(h5_file=h5_file, neuron_ids=filtered_neuron_ids, sort_by_std=sort_by_std)

    n = len(neuron_ids)
    n_new = len(sorted_neuron_ids)

    ## Show the results
    mean_spread = round(np.mean(np.array(sorted_spread)),2)
    median_spread = round(np.median(np.array(sorted_spread)),2)

    print("--------------------------------------")
    print("Visualisation of representative contrast response curves :")
    print(f"    > Analysis made on {n} neurons")
    print(f"    > {n_new} neurons ({round((n_new/n*100), 2)}%) left after filtration")
    if sort_by_std :
        print(f"    > The median of mean_std is {median_spread}")
    else :
        print(f"    > The median of max/min is {median_spread}")

    print(f"    > Plots :")

    for spread_percent in spread_to_plot :

        neuron_pos = int((spread_percent/100) * (n_new-1))
        neuron_id  = sorted_neuron_ids[neuron_pos]
        
        print(f"    > {spread_percent}% of spread, neuron {neuron_id} :")

        title = f"Response contrast for a spread above {spread_percent}% of the neurons (neuron {neuron_id})"

        plot_contrast_response(h5_file=h5_file, neuron_id=neuron_id, title=title)

    print("--------------------------------------")
    print()


def sort_by_shift(
    h5_file,
    neuron_ids,
    low_contrast_id = None,
    high_contrast_id = None
):
    ''' This function sorts the neurons from the ones with the lowest shift value to the ones with the highest.  
        shift is assessed thanks to the GSF at low contrast divided by the GSF at high contrast. Computed in the 'contrast_size_tuning_experiment_all_phases' function

        Prerequisite :
            
            - function 'contrast_size_tuning_experiment_all_phases' executed for the required neurons

        Arguments :

            - low_contrast_id  : Int (Optional) If None, it will take the low contrast to be the lowest contrast computed in the experiment. If not None, the low contrast will be the corresponding id in the 'contrasts' array (the array containing every contrasts tested)
            - high_contrast_id : Same for high contast
            
        Outputs :

            - sorted_neuron_ids    : The sorted neuron ids
            - sorted_shift         : The sorted shift values
    '''
    group_path = "/contrast_size_tuning/results"
    curves_path= "/contrast_size_tuning/curves"

    ## Check if the 'contrast_size_tuning_experiment_all_phases' function has been computed
    check_group_exists_error(h5_file=h5_file, group_path=group_path)
    
    ## Check if the neurons are present in the data
    check_neurons_presence_error(h5_file=h5_file, list_group_path=[group_path], neuron_ids=neuron_ids)

    sorted_neuron_ids = np.copy(neuron_ids)

    with h5py.File(h5_file, 'r') as f :
        
        ## Access the groups
        group_results = f[group_path]
        group_curves  = f[curves_path]
        
        ## Get the radii (useful if low_contrast_id or high_contrast_id is not None)
        group_args_str = group_curves.attrs["arguments"]
        group_dico_args = get_arguments_from_str(group_args_str)
        radii = group_dico_args["radii"]
        ## Convert str to a list
        radii = get_list_from_str(radii)

        all_shift = []

        for neuron_id in sorted_neuron_ids : 

            neuron = f"neuron_{neuron_id}"

            ## Get the neurons shift
            shift = group_results[neuron][:][0]

            ## Get low_contrast_id or high_contrast_id are not None
            if low_contrast_id is not None or high_contrast_id is not None :

                ## Assert none of them are None
                if low_contrast_id is None :
                    low_contrast_id = 0
                if high_contrast_id is None :
                    high_contrast_id = -1
                
                ## Get the corresponding curves
                low_contrast_curve = group_curves[neuron][:][low_contrast_id]
                high_contrast_curve = group_curves[neuron][:][high_contrast_id]
                GSF_low,_,_,_,_  = get_GSF_surround_AMRF(radii = radii,circular_tuning_curve=low_contrast_curve, annular_tuning_curve = None)
                GSF_high,_,_,_,_ = get_GSF_surround_AMRF(radii = radii,circular_tuning_curve=high_contrast_curve, annular_tuning_curve = None)

                shift = np.float64(GSF_low/GSF_high)

            
            all_shift.append(shift)

        ## Sort accross mean_std
        order = np.argsort(all_shift)
    
    ## Order the neuron ids and the shift values
    sorted_neuron_ids = sorted_neuron_ids[order]
    sorted_shift = np.array(all_shift)[order]

    return sorted_neuron_ids, sorted_shift


def contrast_size_tuning_results_1(
    h5_file,
    neuron_ids,
    fit_err_thresh = 0.2,
    shift_to_plot = [15,50,85],
    low_contrast_id = None,
    high_contrast_id= None
): 
    ''' This function aims to visualize what happens when a size tuning experiment is performed at different contrasts :

            - 1) It performs some filtering.
            - 2) It gets the ratio of the GSF at the lowest contrast divided by the GSF at the highest contrast (GSFlow/GSFhigh)
            - 3) It prints the mean value of the GSFlow/GSFhigh, which basically represents how the receptive field radius change when lowering the contrast
            - 4) It plots the curves for neurons representing different shifts
        NB : 'shift' refers to the 'GSFlow/GSFhigh ratio'

        Prerequisite :
        
            - function 'size_tuning_experiment_all_phases' executed for the required neurons
            - function 'contrast_size_tuning_experiment_all_phases' executed for the required neurons

        Filtering :
        
            - Exclude the neurons with receptive fields poorly fitted to a Gaussian model. function 'filter_fitting_error'
            - Exclude the neurons that appeard to have no surround suppression nor response saturation. function 'filter_no_supp_neurons'

        Arguments :

            - shift_to_plot    : An array containing the position of the neurons to plot in the sorted shift array (in percentage). 100 means that it's the neuron with the highest shift, 50 means that 50% of the neurons have a smaller shift.
            - low_contrast_id  : Int (Optional) If None, it will take the low contrast to be the lowest contrast computed in the experiment. If not None, the low contrast will be the corresponding id in the 'contrasts' array (the array containing every contrasts tested)
            - high_contrast_id : Same for high contast
            - other            : see 'size_tuning_results_1'
    '''
    
    ## These functions verify if the prerequisite functions are performed for the requested neurons
    ## Filter on fitting error
    filtered_neuron_ids = filter_fitting_error(h5_file=h5_file, neuron_ids=neuron_ids, fit_err_thresh=fit_err_thresh, print_results=False)

    ## Filter the neurons with no surround suppression and no response saturation
    filtered_neuron_ids = filter_no_supp_neurons(h5_file=h5_file, neuron_ids=filtered_neuron_ids, print_results = False)

    ## Sort the neurons
    sorted_neuron_ids, sorted_shift = sort_by_shift(h5_file=h5_file, neuron_ids=filtered_neuron_ids, low_contrast_id=low_contrast_id, high_contrast_id=high_contrast_id)    

    ## Show the results
    mean_shift = round(np.mean(np.array(sorted_shift)),2)
    n = len(neuron_ids)
    n_new = len(sorted_shift)
    print("--------------------------------------")
    print("Visualisation of representative contrast size tuning curves :")
    print(f"    > Analysis made on {n} neurons")
    print(f"    > {n_new} neurons ({round((n_new/n*100), 2)}%) left after filtration")
    print(f"    > The mean shift value is {mean_shift}")

    print(f"    > Plots :")

    for shift_percent in shift_to_plot :

        neuron_pos = int((shift_percent/100) * (n_new-1))
        neuron_id  = sorted_neuron_ids[neuron_pos]
        
        print(f"    > {shift_percent}% of shift, neuron {neuron_id}, GSFlow/high = {round(sorted_shift[neuron_pos],2)} :")

        title = f"Response contrast for a shift above {shift_percent}% of the neurons (neuron {neuron_id})"

        plot_contrast_size_tuning_curve(h5_file=h5_file, neuron_id=neuron_id, title=title)

    print("--------------------------------------")
    print()



##################################################################################
#####  PART III : Experiments for the second article, Cavanaugh et al., 2002 #####
#####   ------------------------------------------------------------------   #####
#####        Selectivity and Spatial Distribution of Signals From the        #####
#####            Receptive Field   Surround in Macaque V1 Neurons            #####
#####   ------------------------------------------------------------------   #####
#####              DOI : https://doi.org/10.1152/jn.00693.2001               #####
##################################################################################


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
        
            - function 'size_tuning_experiment_all_phases' executed for the required neurons
            - function 'orientation_tuning_experiment_all_phases' executed for the required neurons

        Filtering :
        
            - Exclude the neurons with receptive fields poorly fitted to a Gaussian model. function 'filter_fitting_error'
            - Exclude the neurons that appeard to have no surround suppression nor response saturation. function 'filter_no_supp_neurons'

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
        
            - function 'size_tuning_experiment_all_phases' executed for the required neurons
            - function 'orientation_tuning_experiment_all_phases' executed for the required neurons

        Filtering :
        
            - Exclude the neurons with receptive fields poorly fitted to a Gaussian model. function 'filter_fitting_error'
            - Exclude the neurons that appeard to have no surround suppression nor response saturation. function 'filter_no_supp_neurons'

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



###################################################################################
#####  PART IV : Experiments for the third article, Chun-I Yeh et al., 2009  #####
#####   -------------------------------------------------------------------   #####
#####        “Black” Responses Dominate Macaque Primary Visual Cortex V1      #####
#####   -------------------------------------------------------------------   #####
#####              DOI : https://doi.org/10.1523/JNEUROSCI.1991-09.2009       #####
###################################################################################


def plot_logSNRwb(
    logSNR_OFF,
    logSNR_ON,
    neuron_depths_OFF = None,
    neuron_depths_ON = None
    ):
    ''' This functions plots the Signal Noise Ratios 

        Arguments : 

            - logSNR_OFF : the values of log(SNRwhite / SNRblack) that are < 0
            - logSNR_ON  :  ""  ""  ""  ""  ""  ""  ""  ""  ""  ""  ""  "" > 0

        NB : make sure the depths are discrete values and not continuous to plot the correct mean
    '''

    if neuron_depths_OFF is not None :
        y_OFF = neuron_depths_OFF
        y_ON  = neuron_depths_ON

    else :
        ## Create random y values to faciliate the reading of the plot
        y_OFF = np.random.rand(len(logSNR_OFF))
        y_ON  = np.random.rand(len(logSNR_ON))

    ## MAKE THE PLOT

    ## Start with a square Figure.
    fig = plt.figure(figsize=(8, 6))

    ## Name the fig
    plt.suptitle('Visualisation of the neurons color preference')

    ## Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    ## the size of the marginal axes and the main axes in both directions.
    ## Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 1,  height_ratios=(1, 4),
                    left=0.1, right=0.9, bottom=0.1, top=0.9,
                    wspace=0.05, hspace=0.05)
    
    ## Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_hist = fig.add_subplot(gs[0, 0], sharex=ax)
    
    ## Scatter plot
    ax.scatter(logSNR_OFF, y_OFF, color='0', label = 'black-dominant')
    ax.scatter(logSNR_ON, y_ON, color='1', edgecolor ='k', label = 'white-dominant')
    

    # ## Set nice y axis
    max_y = max(max(y_OFF), max(y_ON))
    ax.set_ylim(max_y + 0.1 * max_y, - 0.10)

    ## Add vertical dashed line at zero
    ax.plot([0,0],[max_y + 0.1 * max_y, - 0.10], linestyle='--', color = 'k')
    
    # Calculate the bin edges (make an edge be 0)
    min_edge = np.min(logSNR_OFF) - (np.min(logSNR_OFF)/20)
    bin_edges = np.linspace(min_edge,0,11)
    gap = abs(bin_edges[1]-bin_edges[0])
    max_val = np.max(logSNR_ON)
    n_gap = round(max_val/gap)
    max_edge =  n_gap * gap
    bin_edges = np.concatenate((bin_edges, np.linspace(0,max_edge,n_gap+1))) 

    
    all_logSNR = logSNR_OFF+logSNR_ON
    weights = np.ones(len(all_logSNR)) / len(all_logSNR)
    ax_hist.hist(all_logSNR, bins=bin_edges, density = False, edgecolor='k', weights=weights)
    ax_hist.tick_params(axis="x", labelbottom=False)

    ax.set_xlabel('log(SNR_white / SNR_black)')

    if neuron_depths_OFF is None :
        ax.set_yticks([])
    else :

        depth_unique = np.union1d(np.unique(neuron_depths_OFF),np.unique(neuron_depths_ON))
        
        ## Key of the dict is the depth, value is the count or the sum of SNR
        dict_count = {}
        dict_sum   = {}

        ## Initialise the dict values
        for depth in depth_unique :
            dict_count[depth] = 0
            dict_sum[depth]   = 0

        ## For every off neuron :
        for i, depth in enumerate(neuron_depths_OFF) :

            logSNR = logSNR_OFF[i] 

            dict_count[depth] += 1
            dict_sum[depth]   += logSNR

        ## For every on neuron :
        for i, depth in enumerate(neuron_depths_ON) :

            logSNR = logSNR_ON[i] 

            dict_count[depth] += 1
            dict_sum[depth]   += logSNR

        mean_val = np.zeros(len(depth_unique))

        for i, depth in enumerate(depth_unique) :
            
            mean_val[i] = dict_sum[depth] / dict_count[depth]

        ## Plot the mean curve
        ax.plot(mean_val, depth_unique, label = 'Mean curve')

        ## Name the y axis
        ax.set_ylabel('Depth')
    
    ax_hist.set_ylabel('Distribution')
    
    ax.legend()
    plt.show()


def black_white_results_1(
    h5_file,
    neuron_ids,
    neuron_depths = None,
    SNR_thresh = 2
):  
    ''' This function aims to visualise whether there is a black or white preference for the selected neurons

        Prerequisite :
            
            - function 'black_white_preference_experiment' executed for the required neurons

        Filtering :

            - Exclude the neurons with neither SNRw nor SNRb that are > SNR_thresh. function 'filter_SNR'

        Arguments :

            - neuron_depths : If not None, will plot the results including depth
                              IMPORTANT : neuron_depths should contain ALL NEURONS DEPTHS, so that the neuron_depths[neuron_id] correspond to the depth of that specific neuron

    '''

    ## This function verify if the prerequisite function has been performed for the requested neurons
    filtered_neuron_ids = filter_SNR(h5_file=h5_file, neuron_ids=neuron_ids, SNR_thresh=SNR_thresh, print_results=False)

    group_path = '/black_white_preference'
    

    ## Get the results
    with h5py.File(h5_file, 'r') as f:
        
        subgroup_results = f[group_path + '/results']
        
        ## Get the log_SNRwb value of every neuron, and sort them whether this value is negative (black preference = OFF) or positive (white preference = ON)
        logSNR_OFF = []
        logSNR_ON  = []

        if neuron_depths is not None :
            neuron_depths_OFF = []
            neuron_depths_ON = []

        all_SNRb   = []
        all_SNRw   = []

        for neuron_id in filtered_neuron_ids :

            neuron = f"neuron_{neuron_id}"
            
            results_neuron = subgroup_results[neuron][:]

            SNR_b    = results_neuron[0]
            SNR_w    = results_neuron[1]
            logSNRwb = results_neuron[2]

            if logSNRwb < 0 : 
                logSNR_OFF.append(logSNRwb)
                if neuron_depths is not None :
                    neuron_depths_OFF.append(neuron_depths[neuron_id])

            else :
                logSNR_ON.append(logSNRwb)
                if neuron_depths is not None :
                    neuron_depths_ON.append(neuron_depths[neuron_id])

            all_SNRb.append(SNR_b)
            all_SNRw.append(SNR_w)
    
    ## Informations to print
    n = len(neuron_ids)
    n_new = len(filtered_neuron_ids)
    mean_SNRb   = round(np.mean(all_SNRb),3)
    mean_SNRw   = round(np.mean(all_SNRw),3)
    mean_logSNR = round(np.mean(logSNR_OFF + logSNR_ON),3)

    ## Show the results
    print("--------------------------------------")
    print("Visualisation of the neurons Black or white preference :")
    print(f"    > Analysis made on {n} neurons")
    print(f"    > {n_new} neurons ({round((n_new/n*100), 2)}%) left after filtration")
    print(f"    > The mean SNR for black stimuli is {mean_SNRb}")
    print(f"    > The mean SNR for white stimuli is {mean_SNRw}")
    print(f"    > The mean log SNR white over black is {mean_logSNR}")
    print(f"    > Plot :")
    if neuron_depths is not None :
        plot_logSNRwb(logSNR_OFF=logSNR_OFF,logSNR_ON=logSNR_ON, neuron_depths_OFF=neuron_depths_OFF, neuron_depths_ON=neuron_depths_ON)
    else :
        plot_logSNRwb(logSNR_OFF=logSNR_OFF,logSNR_ON=logSNR_ON)

    print("--------------------------------------")
    print()
        
    return


def texture_noise_response_results_1(
    h5_file, 
    neuron_ids,
    wanted_fam_order = None
):  
    ''' This function aims to visualise if the neurons product the same response to texture or noise images
    
        Prerequisite :
            
            - function 'texture_noise_response_experiment' executed for the required neurons

        Filtering :
            
            - None
        
        Arguments :

            - dict_fam_order : (Optional) An array containing the families name in the desired order (make sure the names of the families match). If set to False, random order, 

    '''

    group_path    = '/texture_noise_response'
    subgroup_tex_path   = group_path + "/texture"
    subgroup_noise_path = group_path + "/noise"

    ## Check if the 'texture_noise_response' function has been computed
    check_group_exists_error(h5_file=h5_file, group_path=subgroup_noise_path)
    
    ## Check if the neurons are present in the data
    check_neurons_presence_error(h5_file=h5_file, list_group_path=[subgroup_noise_path,subgroup_tex_path], neuron_ids=neuron_ids)

    ## Show the results
    print("--------------------------------------")
    print("Visualisation of neurons response to texture and noise images :")
    print(f"    > Analysis made on {len(neuron_ids)} neurons")
    print(f"    > Plot :")

    ## Get the results for every neuron, shape = (n_family, n_sample, n_neurons)
    all_tex_resp   = []
    all_noise_resp = []

    with h5py.File(h5_file,  'r') as file :
        
        subgroup_tex = file[subgroup_tex_path]
        subgroup_noise = file[subgroup_noise_path]

        ## Get the families id (for the plot)
        description = subgroup_tex.attrs["description"]
        family_ids  = description.split('-')
        family_ids  = np.asarray(family_ids)

        for neuron_id in neuron_ids :
            
            ## name of the dataset 
            neuron = f"neuron_{neuron_id}"

            ## Get the responses to texture and noise for this neuron
            neuron_results_tex    = subgroup_tex[neuron][:]
            neuron_results_noise  = subgroup_noise[neuron][:]

            all_tex_resp.append(neuron_results_tex)
            all_noise_resp.append(neuron_results_noise)

    all_tex_resp   = torch.Tensor(np.stack(all_tex_resp))
    all_noise_resp = torch.Tensor(np.stack(all_noise_resp))
    

    ## Computes the mean responses accross every neuron
    mean_all_tex_resp   = torch.mean(all_tex_resp, dim=0)
    mean_all_noise_resp = torch.mean(all_noise_resp, dim=0)
    
    ## For every family, computes the mean response and the standard deviation to textures and the mean response to noise
    mean_tex_resp   = torch.zeros(len(all_tex_resp[0]))
    mean_noise_resp = torch.zeros(mean_tex_resp.shape)
    std_tex_resp    = torch.zeros(mean_tex_resp.shape)
    std_noise_resp  = torch.zeros(mean_tex_resp.shape)

    n_sample = len(all_tex_resp[0])

    ## Plot for the mean accross neurons
    for i in range(len(mean_all_tex_resp)) :
            
        ## For the texture
        mean_tex_resp[i]   = torch.mean(mean_all_tex_resp[i,:])
        std_tex_resp[i]    = torch.std(mean_all_tex_resp[i,:]) 
        ## For the noise
        mean_noise_resp[i] = torch.mean(mean_all_noise_resp[i,:])
        std_noise_resp[i]  = torch.std(mean_all_noise_resp[i,:])
    
    ## Computes the standart deviation of the mean
    sem_tex_resp = std_tex_resp / math.sqrt(n_sample)
    sem_noise_resp = std_noise_resp / math.sqrt(n_sample)

    ## Change the order if required
    if wanted_fam_order is not None :
                
        dict_old_order = {}

        ## Get a dictionnary dict[name_family] = position in array
        pos = 0
        for i in family_ids :
            dict_old_order[i] = pos
            pos += 1

        wanted_fam_order = np.asarray(wanted_fam_order)
        
        ## Get the new order
        new_order = []
        for fam in wanted_fam_order :
            new_order.append(dict_old_order[fam])
        new_order = np.array(new_order)


    print('Mean accross every neuron')
    plt.title('Mean responses to texture and noise images')


    x_ticks = np.copy(family_ids)

    if wanted_fam_order is not None : 
        x_ticks = x_ticks[new_order]
        mean_tex_resp = mean_tex_resp[new_order]
        mean_noise_resp = mean_noise_resp[new_order]
        
    max_val = max(max(mean_tex_resp),max(mean_noise_resp))
    max_error = max(max(sem_tex_resp), max(sem_noise_resp))
    max_plot  = max_val + max_error
    max_plot  = max_plot + 0.05* max_plot
    y_ticks = np.arange(0,max_plot,0.5)


    plt.errorbar(x_ticks, mean_tex_resp, yerr=sem_tex_resp, fmt='o', capsize=7, capthick=2, elinewidth=2, markersize=12, color='darkgoldenrod', label='Texture')    
    plt.errorbar(x_ticks, mean_noise_resp, yerr=sem_noise_resp, fmt='o', capsize=7, capthick=2, elinewidth=2, markersize=12, color='orange', label='Noise', alpha=0.65)    

    plt.xlabel('Texture family')
    plt.ylabel('Mean response')
    
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.legend()

    plt.show()
    
    ## Plot for 3 random neurons
    for iter in range(3) :

        ## For every family, computes the mean response and the standard deviation to textures and the mean response to noise
        mean_tex_resp   = torch.zeros(len(all_tex_resp[0]))
        mean_noise_resp = torch.zeros(mean_tex_resp.shape)
        std_tex_resp    = torch.zeros(mean_tex_resp.shape)
        std_noise_resp  = torch.zeros(mean_tex_resp.shape)

        r = np.random.randint(0,len(neuron_ids))
        r = [0,1,2][iter]
        for i in range(len(all_tex_resp[0])) :
            
            ## For the texture
            mean_tex_resp[i]   = torch.mean(all_tex_resp[r,i,:])
            std_tex_resp[i]    = torch.std(all_tex_resp[r,i,:])
            ## For the noise
            mean_noise_resp[i] = torch.mean(all_noise_resp[r,i,:])
            std_noise_resp[i]  = torch.std(all_noise_resp[r,i,:])

        ## Computes the standart deviation of the mean
        sem_tex_resp = std_tex_resp / math.sqrt(n_sample)
        sem_noise_resp = std_noise_resp / math.sqrt(n_sample)

        rand_neuron_id = neuron_ids[r]

        print(f'Random neuron : {rand_neuron_id}')
        plt.title(f'Responses to texture and noise images for the neuron {rand_neuron_id}')

        x_ticks = family_ids 

        if wanted_fam_order is not None : 
            x_ticks = x_ticks[new_order]
            mean_tex_resp = mean_tex_resp[new_order]
            mean_noise_resp = mean_noise_resp[new_order]
            
        max_val = max(max(mean_tex_resp),max(mean_noise_resp))
        max_error = max(max(sem_tex_resp), max(sem_noise_resp))
        max_plot  = max_val + max_error
        max_plot  = max_plot + 0.05* max_plot
        y_ticks = np.arange(0,max_plot,0.5)

        plt.errorbar(x_ticks, mean_tex_resp, yerr=sem_tex_resp, fmt='o', capsize=7, capthick=2, elinewidth=2, markersize=12, color='darkgreen', label='Texture')    
        plt.errorbar(x_ticks, mean_noise_resp, yerr=sem_noise_resp, fmt='o', capsize=7, capthick=2, elinewidth=2, markersize=12, color='limegreen', label='Noise', alpha=0.7)    

        plt.xlabel('Texture family')
        plt.ylabel('Response')

        plt.xticks(x_ticks)
        plt.yticks(y_ticks)
        plt.legend()
    
        plt.show()

    print("--------------------------------------")


def texture_noise_response_results_2(
    h5_file, 
    neuron_ids,
    wanted_fam_order = None
):  
    ''' This function aims to visualise the average modulation index accross every neuron for each texture family.
        The modulation index is defined as so : (response_texture - response_noise) / (response_texture + response_noise)

        Prerequisite :
            
            - function 'texture_noise_response_experiment' executed for the required neurons

        Filtering :
            
            - None 
    '''

    group_path    = '/texture_noise_response'
    subgroup_tex_path   = group_path + "/texture"
    subgroup_noise_path = group_path + "/noise"

    ## Check if the 'texture_noise_response' function has been computed
    check_group_exists_error(h5_file=h5_file, group_path=subgroup_noise_path)
    
    ## Check if the neurons are present in the data
    check_neurons_presence_error(h5_file=h5_file, list_group_path=[subgroup_noise_path,subgroup_tex_path], neuron_ids=neuron_ids)

    ## Show the results
    print("--------------------------------------")
    print("Visualisation of the average modulation index for each texture family:")
    print(f"    > Analysis made on {len(neuron_ids)} neurons")
    print(f"    > Plot :")

    ## Get the results for every neuron shape = (n_family, n_sample, n_neurons)
    all_tex_resp   = []
    all_noise_resp = []

    with h5py.File(h5_file,  'r') as file :
        
        subgroup_tex = file[subgroup_tex_path]
        subgroup_noise = file[subgroup_noise_path]

        ## Get the families id (for the plot)
        description = subgroup_tex.attrs["description"]
        family_ids  = description.split('-')

        for neuron_id in neuron_ids :
            
            ## name of the dataset 
            neuron = f"neuron_{neuron_id}"

            ## Get the responses to texture and noise for this neuron
            neuron_results_tex    = subgroup_tex[neuron][:]
            neuron_results_noise  = subgroup_noise[neuron][:]

            all_tex_resp.append(neuron_results_tex)
            all_noise_resp.append(neuron_results_noise)

    all_tex_resp   = torch.Tensor(np.stack(all_tex_resp))
    all_noise_resp = torch.Tensor(np.stack(all_noise_resp))
    
    ## Change the order if required
    if wanted_fam_order is not None :
                
        dict_old_order = {}
        ## Get a dictionnary dict[name_family] = position in array
        pos = 0
        for i in family_ids :
            dict_old_order[i] = pos
            pos += 1

        wanted_fam_order = np.asarray(wanted_fam_order)
        
        ## Get the new order
        new_order = []
        for fam in wanted_fam_order :
            new_order.append(dict_old_order[fam])
        new_order = np.array(new_order)

    ## Compute every modulation index
    all_modulation_index = (all_tex_resp - all_noise_resp) / (all_tex_resp + all_noise_resp)

    ## Average the modulation accross every neuron
    all_modulation_index = torch.mean(all_modulation_index, dim=0)

    ## Average the modulation index accross every sample
    all_modulation_index = torch.mean(all_modulation_index, dim=1)
    
    if wanted_fam_order is not None : 

        family_ids = np.array(family_ids)[new_order]
        all_modulation_index = all_modulation_index[new_order]

    plt.bar(family_ids, all_modulation_index, width=0.8, color = 'limegreen', edgecolor='black')

    minval = min(all_modulation_index)
    maxval = max(all_modulation_index)
    minlim = min(-0.15, minval - abs(0.1 * minval))
    maxlim = max(0.3, maxval + abs(0.1 * maxval))

    plt.ylim([minlim, maxlim])
    plt.xlabel('Texture family')
    plt.ylabel('Modulation index')
    plt.title('Modulation index of every texture family\n averaged accross every neuron')

    plt.show()

    print("--------------------------------------")


def texture_noise_response_results_3(
    h5_file, 
    neuron_ids
):  
    ''' This function aims to visualise the distribution of the modulation index averaged accross every texture family and every sample
        The modulation index is defined as so : (response_texture - response_noise) / (response_texture + response_noise)

        Prerequisite :
            
            - function 'texture_noise_response_experiment' executed for the required neurons

        Filtering :
            
            - None
    '''

    group_path    = '/texture_noise_response'
    subgroup_tex_path   = group_path + "/texture"
    subgroup_noise_path = group_path + "/noise"

    ## Check if the 'texture_noise_response' function has been computed
    check_group_exists_error(h5_file=h5_file, group_path=subgroup_noise_path)
    
    ## Check if the neurons are present in the data
    check_neurons_presence_error(h5_file=h5_file, list_group_path=[subgroup_noise_path,subgroup_tex_path], neuron_ids=neuron_ids)

    ## Show the results
    print("--------------------------------------")
    print("Visualisation of distribution of the mean Modulation index for every neuron :")
    print(f"    > Analysis made on {len(neuron_ids)} neurons")
    print(f"    > Plot :")

    ## Get the results for every neuron shape = (n_family, n_sample, n_neurons)
    all_tex_resp   = []
    all_noise_resp = []

    with h5py.File(h5_file,  'r') as file :
        
        subgroup_tex = file[subgroup_tex_path]
        subgroup_noise = file[subgroup_noise_path]

        for neuron_id in neuron_ids :
            
            ## name of the dataset 
            neuron = f"neuron_{neuron_id}"

            ## Get the responses to texture and noise for this neuron
            neuron_results_tex    = subgroup_tex[neuron][:]
            neuron_results_noise  = subgroup_noise[neuron][:]

            all_tex_resp.append(neuron_results_tex)
            all_noise_resp.append(neuron_results_noise)

    all_tex_resp   = torch.Tensor(np.stack(all_tex_resp))
    all_noise_resp = torch.Tensor(np.stack(all_noise_resp))
    
    ## Compute every modulation index
    all_modulation_index = (all_tex_resp - all_noise_resp) / (all_tex_resp + all_noise_resp)

    ## Average the modulation accross every neuron
    all_modulation_index = torch.mean(all_modulation_index, dim=2)

    ## Average the modulation index accross every sample
    all_modulation_index = torch.mean(all_modulation_index, dim=1)
    
    meanval = torch.mean(all_modulation_index).item()

    weights = np.ones(all_modulation_index.shape) / len(all_modulation_index)
    values, bins, _ = plt.hist(all_modulation_index, edgecolor='black', bins=10, density=False, weights=weights, color= 'limegreen')
    max_hist = max(values)

    plt.plot([0,0], [0,max_hist + 0.1 * max_hist], 'k--', label='zero')
    plt.plot([meanval,meanval], [0, max_hist + 0.1 * max_hist], color = 'darkgoldenrod', label = 'mean')

    minval = min(bins)
    maxval = max(bins)
    minlim = min(-0.5, minval - abs(0.1 * minval))
    maxlim = max( 0.5, maxval + abs(0.1 * maxval))

    plt.xlim([minlim,maxlim])
    plt.xlabel('Modulation Index')
    plt.ylabel('Distribution')
    plt.title('Distribution of the Modulation index in the neurons')
    plt.legend()
    plt.show()

    print("--------------------------------------")