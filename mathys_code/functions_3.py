#%%
import numpy as np
import torch
from tqdm import tqdm
from itertools import combinations
## Plots
import matplotlib.pyplot as plt
## Important functions
from surroundmodulation.utils.misc import rescale
from scipy.optimize import curve_fit
## Data storage
import h5py
## Import functions
from functions import *
from functions_2 import *

# %%

##################################################################
## Functions for the ARTICLE III :                              ##
## “Black” Responses Dominate Macaque Primary Visual Cortex V1  ##
## Chun-I Yeh, Dajun Xing, and Robert M. Shapley                ##
## DOI : https://doi.org/10.1523/JNEUROSCI.1991-09.2009         ##
##################################################################

def black_white_preference_experiment(
    h5_file,
    all_neurons_model,
    neuron_ids,   
    overwrite = False, 
    dot_size_in_pixels=5,
    contrast = 1, 
    img_res = [93,93], 
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None
    ):  
    ''' This function aims to find the Signal Noise Ratio (SNR) for black and white dot stimuli for the selected neurons :

            - 1) It creates the stimulations for every possible dot position
            - 2) It gets the response of the neuron of the model to the stimulations
            - 3) It creates a "position-response" image (matrix) where every pixel represents the summed response of the neuron to a dot at this position
            - 4) It creates a noise image similarly to 3) but the response is being shuffled
            - 5) Performs 1-4 for black and white stimuli and computes the Signal Noise Ratio as the variance of the position-response matrix divided by the variance of the noise
            - 6) Saves the position-response images, the noise images and the SNR in three subgroups in the HDF5 file

    
    It saves the data with this architecture : 


                                                _________ SubGroup ../position_response_img   --> neuron datasets
                                                |
        Group /black_white_preference   ________|________ SubGroup ../noise_img                  --> neuron datasets
                                                |
                                                |________ SubGroup ../results                 --> neuron datasets
    Prerequisite :

        - None

    Arguments :

        - dot_size_in_pixels : Size of the pixels, corresponding to the size of the sides of the square
        - others             : explained in other functions

    Outputs :

        - datasets in ../position_response_img  : A tensor containing two images (matrices). The first one is for black dots stimulation, the second one for white dot stimulation 
                                                  every pixel of the images correspond to the summed response of the neuron to this pixel

        - datasets in ../noise_img              :  A tensor containing two images (matrices). The first one is for black dots stimulation, the second one for white dot stimulation 
                                                   every pixel of the images correspond to the summed shuffled response of the neuron to this pixel (a shuffled response means that the response assigned to every dot can now be the response to another dot)


        - datasets in ../results                : An array containing the Signal noise ratios values and the log10(SNRw / SNRb)
                                         format : [SNR_b, SNR_w, logSNRwb]

    '''

    print(' > Black or white preference experiment')

    ## Groups to fill
    group_path = "/black_white_preference"
    subgroup_path_resp_img  = group_path + "/position_response_img"
    subgroup_path_noise_img = group_path + "/noise_img"
    subgroup_path_results   = group_path + "/results"

    ## Clear the group if requested
    if overwrite : 
        clear_group(h5_file,group_path)

    ## Initialize the Group and the subgroups
    args_str = f"dot_size_in_pixels={dot_size_in_pixels}/contrast={contrast}/img_res={img_res}/pixel_min={pixel_min}/pixel_max={pixel_max}"
    group_init(h5_file=h5_file, group_path=group_path, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_resp_img, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_results, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_noise_img, group_args_str=args_str)


    ## Select device
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_neurons_model.to(device)
    all_neurons_model.eval()
    
    
    ## Generate the objects to fill
    sum_dot_pos   = torch.zeros((img_res[1], img_res[0]), dtype=int).to(device)
    black_stimuli = []
    white_stimuli = []
    resp_b = []
    resp_w = []

    with torch.no_grad():
 
        ## Perform the dot stimulation
        for y in range(img_res[0] - dot_size_in_pixels + 1):
            
            for x in range(img_res[1] - dot_size_in_pixels + 1) :
            
                ## Create an image with values = 0
                black_stim = np.zeros(img_res, dtype=int)
                white_stim = np.zeros(img_res, dtype=int)

                ## Save the dot position
                sum_dot_pos[y:y+dot_size_in_pixels, x:x+dot_size_in_pixels] += 1

                ## Do one black and one white stimulus
                black_value = -1
                white_value = 1

                ## Create the dots
                black_stim[y:y+dot_size_in_pixels, x:x+dot_size_in_pixels] = black_value*contrast
                white_stim[y:y+dot_size_in_pixels, x:x+dot_size_in_pixels] = white_value*contrast

                ## Save the dots
                black_stimuli.append(torch.Tensor(black_stim))
                white_stimuli.append(torch.Tensor(white_stim))

        ## Convert the list of tensors to a unique tensor
        black_stimuli = torch.stack(black_stimuli)
        white_stimuli = torch.stack(white_stimuli)

        num_dots = len(black_stimuli)

        ## For every input get the responses of the all_neurons_model
        ## Rescale the image to pixel_min and pixel_max
        model_input_b = rescale(black_stimuli, -1, 1, pixel_min, pixel_max).reshape(num_dots, 1, 93,93).to(device)
        model_input_w = rescale(white_stimuli, -1, 1, pixel_min, pixel_max).reshape(num_dots, 1, 93,93).to(device)

        ## Get the responses
        resp_b = all_neurons_model(model_input_b)
        resp_w = all_neurons_model(model_input_w)
        
        ## Tensor 1 is a tensor containing every dot position of the images (1 if the dot is here, else 0). shape = (num_dots * img_res) 
        ## Tensor 2 is a tensor containing the responses of every neuron to the dot images,  shape = (num_dots * nNeuron) #nNeuron = len(output_of_all_neurons_model)
        ## This function creates for every dot (first dimension = 'b') the following tensor :
        ## 'nxy' Basically, for each neuron, resp_to_the_image * dot_position_matrix_in_image
        ## It can be visualised as an array of matrices. Each matrix contains zeros where there isn't a dot and the value of the response where the dot is
        ## Then it sums this Tensor for every dot image that was presented
        ## The result image is what is called 'position response image' and baically contains the summed response of the dots at every position (pixel)
        all_position_resp_b = torch.einsum(
                            'bxy,bn->nxy', 
                            torch.abs(black_stimuli).to(device),  ## torch.abs because black values is -1
                            (resp_b).to(device)
                            )
        
        all_position_resp_w = torch.einsum(
                            'bxy,bn->nxy', 
                            white_stimuli.to(device),
                            (resp_w).to(device)
                            )
        
        ## Shuffle the responses for each neuron to create noise
        shuffle_resp_b = np.copy(resp_b.cpu())
        shuffle_resp_w = np.copy(resp_w.cpu())
        np.apply_along_axis(np.random.shuffle, axis=0, arr=shuffle_resp_b)
        np.apply_along_axis(np.random.shuffle, axis=0, arr=shuffle_resp_w)


        ## Put everything back in a tensor on the correct device
        shuffle_resp_b = torch.Tensor(shuffle_resp_b).to(device)
        shuffle_resp_w = torch.Tensor(shuffle_resp_b).to(device)
        
        all_noise_b = torch.einsum(
                            'bxy,bn->nxy', 
                            torch.abs(black_stimuli).to(device),  ## torch.abs because black values is -1
                            (shuffle_resp_b).to(device)
                            )
        
        all_noise_w = torch.einsum(
                            'bxy,bn->nxy', 
                            white_stimuli.to(device),  
                            (shuffle_resp_w).to(device)
                            )
        
    ## Fill the HDF5 file 
    with h5py.File(h5_file, 'a') as f :

        ## Save everything in a dictionnary where every key is a neuron index
        for neuron_id in tqdm(neuron_ids):

            neuron = f"neuron_{neuron_id}"

            if neuron not in f[subgroup_path_results] :
                
                ## Create datasets
                pos_resp_imgs = torch.stack([all_position_resp_b[neuron_id], all_position_resp_w[neuron_id]]).to('cpu')
                noises        = torch.stack([all_noise_b[neuron_id], all_noise_w[neuron_id]]).to('cpu')
        
                SNR_b = torch.var(pos_resp_imgs[0]).item() / torch.var(noises[0]).item()
                SNR_w = torch.var(pos_resp_imgs[1]).item() / torch.var(noises[1]).item()

                logSNRwb = np.log10(SNR_w/SNR_b)
                
                f[subgroup_path_resp_img].create_dataset(name=neuron, data=pos_resp_imgs)
                f[subgroup_path_noise_img].create_dataset(name=neuron, data=noises)
                f[subgroup_path_results].create_dataset(name=neuron, data=[SNR_b, SNR_w, logSNRwb])


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
    

    ## Set nice y axis
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

    print(len(logSNR_OFF))
    print(len(logSNR_ON))
    all_logSNR = logSNR_OFF+logSNR_ON
    print(len(all_logSNR))
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

def filter_SNR(
    h5_file, 
    neuron_ids,
    SNR_thresh = 2,
    print_results = False
):
    ''' The goal of this function is to return only the neurons with a significant response to black or white dot stimulus
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
                              IMPORTANT : neuronn_depths should contain ALL NEURONS DEPTHS, so that the neuron_depths[neuron_id] correspond to the depth of that specific neuron

        ##TODO : Plot only an histogram when the neuron depths is not included
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



## TODO DELETE

import numpy as np
## Models
from nnvision.models.trained_models.v1_task_fine_tuned import v1_convnext_ensemble
## Functions
from functions   import *
from functions_2 import *
## To select the well predicted neurons 
from surroundmodulation.utils.misc import pickleread
from surroundmodulation.utils.plot_utils import plot_img
h5_file='/project/mathys_code/article1.h5'
n = 458
neuron_ids = np.arange(n) ## Because 458 outputs in our model
corrs = pickleread("/project/mathys_code/avg_corr.pkl") ## The correlation score of the neurons
neuron_ids = neuron_ids[corrs>0.75]

black_white_preference_experiment(
    h5_file=h5_file,
    all_neurons_model=v1_convnext_ensemble,
    neuron_ids=neuron_ids,   
    overwrite=True, 
    dot_size_in_pixels=5,
    contrast=1, 
    img_res = [93,93], 
    pixel_min = -1.7876,
    pixel_max =  2.1919,
    device = None
    )
# %%
neuron_depths = pickleread("/project/mathys_code/depth_info.pickle") 

black_white_results_1(
    h5_file,
    neuron_ids,
    neuron_depths = neuron_depths,
    SNR_thresh = 2
)
# %%
