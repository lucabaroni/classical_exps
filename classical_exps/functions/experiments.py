####################################################################################
#####    THIS CODE CONTAINS EVERY FUNCTION USED TO PERFORM THE EXPERIMENTS.    #####
#####             ------------------------------------------------             #####
#####     Those functions works with a HDF5 file where they store the data     #####
####################################################################################

############################
##### PART 0 : Imports #####
############################

## Useful
import numpy as np
import torch
from tqdm import tqdm
import os
## Utils
from classical_exps.functions.utils import *
## Image generation
import imagen
from imagen.image import BoundingBox
import matplotlib.image as mpimg
## Import models 
from classical_exps.functions.utils import SingleCellModel
## Data storage
import h5py
from scipy.optimize import curve_fit

########################################################
##### PART I : Functions that perform pre-analyses #####
########################################################


def find_preferred_grating_parameters_full_field(
    single_model, 
    orientations = np.linspace(0, np.pi, 37)[:-1], 
    spatial_frequencies = np.linspace(1, 7, 25), 
    phases = np.linspace(0, 2*np.pi, 37)[:-1],
    contrast = 1,         
    img_res = [93,93], 
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.67,           
    ):

    ''' 
        This function generates full field grating images with every parameters combination and 
        returns the parameters that lead to the greatest activation in the Single Neuron Model.

        Arguments : 

            - single_model        : A single cell model of the class 'surroundmodulation.models.SingleCellModel'
            - orientations        : An array containing the orientations of the grating to test (in rad)
            - spatial_frequencies : An array containing the spatial frequencies to test (it correspond to the number of grating cycles per degree of excentricity)
            - phases              : An array containing the phases of the grating to test
            - contrast            : The value that will multiply the grating image's values (lower than 1 will reduce the contrast and greater than 1 will increase the contrast)
            - img_res             : Resolution of the image in term of pixels [nb_y_pix, nb_x_pix]
            - pixel_min           : Value of the minimal pixel that will serve as the black reference
            - pixel_max           : Value of the maximal pixel that will serve as the white reference (NB : The gray value will be the mean of those two)
            - device              : The device on which to execute the code, if set to "None" it will take the available one
            - size                : The size of the image in terms of degree of visual angle (DVA)

        Outputs : 

            - max_ori             : Preferred orientation
            - max_sf              : Preferred spatial frequency
            - max_phase           : Preferred phase
            - stim_max            : Preferred grating image (with the preferred parameters)
            - resp_max            : Response of the model to the preferred image

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
    
    ## Initialisation
    resp_max = 0

    ## Get every parameters combination
    with torch.no_grad():
        for orientation in orientations :
            for sf in spatial_frequencies:
                for phase in phases: 
                
                    ## Creation of the grating image
                    grating = torch.Tensor(imagen.SineGrating(
                        orientation = orientation, 
                        frequency = sf,
                        phase = phase,
                        bounds = BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))), #The bounds of the image. Warning, the shape is : ((width_left_side, height_bottom_side) , width_right_side, height_top_side)). 
                        offset = 0, #?
                        scale = 1,  #?
                        xdensity = img_res[1]/size[1], #pixels/degree
                        ydensity = img_res[0]/size[0],
                    )())

                    grating = grating.reshape(1,*img_res).to(device)
                    ## Rescale because the output of imagen has values from 0 to 1 and we want 
                    ## values from pixel_min to pixel_max (be carful to use contrast on centered values)
                    grating = rescale(grating, 0, 1, -1, 1)*contrast
                    grating = rescale(grating, -1, 1, pixel_min, pixel_max)

                    resp = single_model(grating.reshape(1,1,*img_res))

                    if resp>resp_max:
                        resp_max = resp
                        max_ori = orientation
                        max_phase = phase
                        max_sf = sf
                        stim_max = grating
        
    return max_ori, max_sf, max_phase, stim_max, resp_max


def get_all_grating_parameters(
    h5_file,
    all_neurons_model,
    neuron_ids,
    overwrite = False,
    orientations = np.linspace(0, np.pi, 37)[:-1], 
    spatial_frequencies = np.linspace(1, 7, 25), 
    phases = np.linspace(0, 2*np.pi, 37)[:-1],
    contrast = 1,         
    img_res = [93,93], 
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.67,           
    ):

    '''
        This function uses the 'find_preferred_grating_parameters_full_field' on every single cell model (neuron)
        and write the preferred parameters for the full field grating image of the selected neurons in the HDF5 file
        

        Arguments :

            - h5_file           : The HDF5 file containing the data
            - overwrite         : If set to True, will erase the data in the group "full_field_params" and then fill it again. 
                                If set to False, the function will conserve the existing data and only add the one not already present
            - all_neurons_model : The full model containing every neurons (In our analysis it correspond to the v1_convnext_ensemble)
            - neuron_ids        : The list (or array) of the neurons we wish to perform the analysis on
            - others            : Explained in 'find_preferred_grating_parameters_full_field'

        Outputs :

            - datasets in /full_field_params : An array containing the preferred full field parameters
                                               Format = [pref_orientation, pref_spatial_frequency, pref_phase]
    '''
    print(' > Get grating parameters')

    ## Create the group path
    group_path = "/full_field_params"

    ## Clear the group if requested
    if overwrite : 
        clear_group(h5_file,group_path)

    ## This will serve to create and verify the arguments of the group
    args_str = f"orientations={orientations}/spatial_frequencies={spatial_frequencies}/phases={phases}/contrast={contrast}/img_res={img_res}/pixel_min={pixel_min}/pixel_max={pixel_max}/size={size}"

    ## Create the group if it doesn't exist, and, if it already exists, check if the arguments are matching
    group_init(h5_file, group_path, args_str)
    
    ## Select device
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_neurons_model.to(device)

    ## Evaluation mode
    all_neurons_model.eval()

    ## Convert the size to the right shape ( [size_height, size_width] )
    if type(size) is int or type(size) is float: 
        size = [size]*2
    
    ## Initialisation of the maximal response of every neuron
    resp_max = torch.zeros(len(neuron_ids)).to(device)

    ## Save the preferred parameters (an array where each row is a neuron and each column is a parameter)
    preferred_params = torch.zeros((len(neuron_ids),3)) ## Columns = [ori, sf, phase]

    ## If the neurons are not all in the data : 
    if check_neurons_presence(h5_file, [group_path], neuron_ids) : 
        return
        
    else :

        ## Get every parameters combination
        with torch.no_grad():
            for orientation in tqdm(orientations) :
                for sf in spatial_frequencies:
                    for phase in phases: 
                    
                        ## Creation of the grating image
                        grating = torch.Tensor(imagen.SineGrating(
                            orientation = orientation, 
                            frequency = sf,
                            phase = phase,
                            bounds = BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))), #The bounds of the image. Warning, the shape is : ((width_left_side, height_bottom_side) , width_right_side, height_top_side)). 
                            offset = 0, #?
                            scale = 1,  #?
                            xdensity = img_res[1]/size[1], #pixels/degree
                            ydensity = img_res[0]/size[0],
                        )())

                        grating = grating.reshape(1,*img_res).to(device)
                        ## Rescale because the output of imagen has values from 0 to 1 and we want 
                        ## values from pixel_min to pixel_max (be carful to use contrast on centered values)
                        grating = rescale(grating, 0, 1, -1, 1)*contrast
                        grating = rescale(grating, -1, 1, pixel_min, pixel_max)

                        ## Get the response for the selected neurons
                        resp = all_neurons_model(grating.reshape(1,1,*img_res))[0][neuron_ids]
                        
                        ## Get the indices of the neurons that yield the maximum response
                        condition = torch.where(resp>resp_max)[0]

                        ## Get the current parameters 
                        current_params = torch.Tensor([orientation,sf,phase])

                        ## Save the preferred parameters for these neurons
                        preferred_params[condition] = current_params

                        ## Update the maximal response
                        resp_max[condition] = resp[condition]

        ## Now save the parameters in the file
        with h5py.File(h5_file, 'a') as file :

            ## Access the group
            group = file[group_path]

            ## Add description
            group.attrs["description"] = "datasets = [orientation, sf, phase_full_field]"

            for i, id in enumerate(neuron_ids):

                ## Put everything into a list
                data = preferred_params[i]
            
                ## Create a dataset for the neuron if it doen't already exists
                try :
                    group.create_dataset(f"neuron_{id}", data=data)

                except ValueError :
                    pass


def dot_stimulation(
    all_neurons_model,
    neuron_ids,  
    dot_size_in_pixels=4,
    contrast = 1, 
    num_dots=200000,
    img_res = [93,93], 
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    bs = 40,
    seed = 0,
    ):  
    ''' This function aims to find, for the neurons of the all_neurons_model,
    their sensivity to points presented at different positions of the image.

    
    Arguments :

        - dot_size_in_pixels : Size of the pixels, corresponding to the size of the sides of the square
        - num_dots           : IMPORTANT : must be a multiple of bs | Number of dot images to present to the neurons. 
        - bs                 : IMPORTANT : num_dots%bs == 0.        | Batch size
        - seed               : random seed 
        - others             : explained in other functions

    Output :

        - dot_stim_dict : dictionnary containing the results of the dot stimulation for every selected neuron. The key are the neuron ids and the values are images where the responses of neurons to the dots where summed 
    '''

    ## Select device
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_neurons_model.to(device)
    
    ## Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate the tensors
    tensors = []
    resp = []
    

    with torch.no_grad():
        ## Generate the responses to gray screens
        no_stim_resp= all_neurons_model(torch.ones(1,1,*img_res, device=device)* (pixel_min + pixel_max)/2)

        ## Do the dot stimulation
        for _ in range(num_dots):

            ## Create an image with values = 0
            tensor = np.zeros(img_res, dtype=int)

            ## Get a random position for the dot 
            x = np.random.randint(0, img_res[0] - dot_size_in_pixels + 1)
            y = np.random.randint(0, img_res[1] - dot_size_in_pixels + 1)

            ## Make the dot either black or white
            square_values = np.random.choice([-1, 1])

            ## Create the dot
            tensor[x:x+dot_size_in_pixels, y:y+dot_size_in_pixels] = square_values*contrast

            ## Save the dot
            tensors.append(torch.Tensor(tensor))
        
        ## Convert the list of tensors to a unique tensor
        tensors_array = torch.stack(tensors)
    
        ## For every batch, get the responses of the all_neurons_model
        for i in tqdm(range(0, num_dots, bs)):
            ## Rescale the image to pixel_min and pixel_max
            model_input = rescale(tensors_array[i:i+bs], -1, 1, pixel_min, pixel_max).reshape(bs, 1, *img_res).to(device)
            resp.append(all_neurons_model(model_input))
        
        ## Merge the batches together
        resp = torch.cat(resp, dim=0)


        ## Tensor 1 is a tensor containing every dot position of the images (1 if the dot is here, else 0)                                         shape = (num_dots * img_res) 
        ## Tensor 2 is a tensor containing the responses of every neuron to the dot images, with a substraction of the response to gray stimulus   shape = (num_dots * nNeuron) #nNeuron = len(output_of_all_neurons_model)
        ## This function creates for every dot (first dimension = 'b') the following tensor :
        ## 'nxy' Basically, for each neuron, resp_to_the_image * dot_position_in_image
        ## It can be visualised as an array of matrices. Each matrix contains of zeros where there isn't a dot and the normalised value of the resp where the dot is
        ## Then it sums this Tensor for every dot image that was presented
        all_neurons_outputs = torch.einsum(
                            'bxy,bn->nxy', 
                            torch.abs(tensors_array).to(device),
                            (resp-no_stim_resp).to(device)
                            )
        ## For this we assume that a large amount of pixels presented would lead to a uniform presentation accross every position,
        ## So the results we obtain after this are accounting for the strength of the response of the neurons to every position, and thus their receptive field 

    dot_stim_dict = {}

    ## Save everything in a dictionnary where every key is a neuron index
    for id in neuron_ids:
        dot_stim_dict[id] = all_neurons_outputs[id]

    return dot_stim_dict


def gaussian2D_with_correlation(xy, A, x0, y0, sigma_x, sigma_y, rho):
    ''' The model of the gaussian function'''
    x, y = xy
    a = 1.0 / (2 * (1 - rho**2))
    b = ((x - x0)**2) / (sigma_x**2)
    c = 2 * rho * (x - x0) * (y - y0) / (sigma_x * sigma_y)
    d = ((y - y0)**2) / (sigma_y**2)
    return A * np.exp(-a * (b - c + d))


def gauss_fit(dot_stim_img, img_size=[93,93]): 
    ''' This function takes images coming from a dot stimulation experiment and tries to 
    make a 2D gaussian model fit to it.
    The images are the output of the dot_stimulation function, contained as values in the dictionnary
    
    Arguments :

        - dot_stim_img : Should be a numpy array. This is the values contained in the dictionary which is the output of the 'dot_stimulation' function

    Outputs :

        - A_opt, x0_opt, y0_opt, sigma_x_opt, sigma_y_opt, rho_opt : Fitted parameters

    '''

    with torch.no_grad():
        data = np.clip(dot_stim_img, a_min=0, a_max=None)
        data = data/data.std()
        x_data = np.arange(0, img_size[1])
        y_data = np.arange(0, img_size[0])
        x, y = np.meshgrid(x_data, y_data)

        # Flatten for fitting
        x_data, y_data = x.ravel(), y.ravel()
        z_data = data.ravel()

        # Initial guess [A, x0, y0, sigma_x, sigma_y, rho]
        initial_guess = [5, 93/2, 93/2, 10, 10, 0]

        # Fit the model
        params, covariance = curve_fit(gaussian2D_with_correlation, (x_data, y_data), z_data, p0=initial_guess)

        # Extract the optimized parameters
        A_opt, x0_opt, y0_opt, sigma_x_opt, sigma_y_opt, rho_opt = params

        fitted_data = gaussian2D_with_correlation((x_data, y_data), *params)
        fitted_data = fitted_data.reshape(*img_size)

        return A_opt, x0_opt, y0_opt, sigma_x_opt, sigma_y_opt, rho_opt, fitted_data


def get_preferred_position(
    h5_file,
    all_neurons_model,
    neuron_ids,
    overwrite = False,
    dot_size_in_pixels=4,
    contrast = 1, 
    num_dots=200000,
    img_res = [93,93], 
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    bs = 40,
    seed = 0
):
    ''' This function performs multiple things :
        1) Multiple verifications to see if the group "perferred_pos" exists in the file and if so, check if the parameters are compatible
        2) Uses the function "dot_stimulation" to obtain a dictionnary containing the excitatory pixels for every neuron in "neuron_ids"
        3) Uses the function "gauss_fit" to try to fit the data to a 2D Gaussian model
        4) If it fits : save x0_opt and y0_opt and the error. x0_opt and y0_opt are the preferred position for the stimulus
        5) If the fitting doesn't converges : Save an error = np.nan and the middle point of the image for x0_opt and y0_opt
        6) Save the results in HDF5 file

        Outputs :

            - datasets in /preferred_pos : An array containing the prefered position (x and y) and the fitting error 
                                           Format = [x_pix, y_pix, error]
        '''
    
    print(' > Get preferred position')

    ## Step 1

    ## Create the group path
    group_path = "/preferred_pos"

    ## Clear the group if requested
    if overwrite : 
        clear_group(h5_file,group_path)
    
    ## This will serve to create and verify the arguments of the group
    args_str = f"dot_size_in_pixels={dot_size_in_pixels}/contrast={contrast}/num_dots={num_dots}/img_res={img_res}/pixel_min={pixel_min}/pixel_max={pixel_max}/bs={bs}/seed={seed}"

    ## Create the group if it doesn't exist, and, if it already exists, check if the arguments are matching
    group_init(h5_file, group_path, args_str)

    ## Step 2)
    dot_stim_dict = dot_stimulation(
        all_neurons_model = all_neurons_model,
        neuron_ids = neuron_ids,  
        dot_size_in_pixels = dot_size_in_pixels,
        contrast = contrast, 
        num_dots = num_dots,
        img_res = img_res, 
        pixel_min = pixel_min, 
        pixel_max = pixel_max, 
        device = device,
        bs = bs,
        seed = seed)

    with h5py.File(h5_file, 'a') as file :

        ## Select the group
        group = file[group_path]
        ## Add description
        group.attrs["description"] = "datasets = [x_pix, y_pix, error]"     

        for id in neuron_ids:

            ## Steps 3)
            ## Try to fit
            try: 
                ## Normalise the dot stimulation result and put them in the right format
                dot_stim_norm = dot_stim_dict[id].detach().cpu().numpy()/dot_stim_dict[id].detach().cpu().numpy().std()
                
                ## Fit to the 2D Gaussian model
                A_opt, x0_opt, y0_opt, sigma_x_opt, sigma_y_opt, rho_opt, fitted_dot_stim = gauss_fit(dot_stim_norm)
                
                ## Calculate the error between the model and the data
                error = np.mean((fitted_dot_stim - dot_stim_norm)**2)

                ## Step 5) Save x0_opt, y0_opt and error
        
                try :
                    ## Create a dataset for the neuron if it doen't already exists
                    data = [x0_opt,y0_opt,error]
                    group.create_dataset(f"neuron_{id}", data=data)

                except ValueError :
                    pass
                
      
            ## Step 4)
            ## If it do not converge, that mean that the receptive field could not be fitted to a gaussian model
            except: 
                ## Step 5) Save x0_opt, y0_opt and error = np.nan
                try :
                    ## Create a dataset for the neuron if it doen't already exists
                    data=[img_res[1]/2,img_res[0]/2,np.nan]
                    group.create_dataset(f"neuron_{id}", data=data )
                    

                except ValueError :
                    pass



##################################################################################
#####  PART II : Experiments for the first article, Cavanaugh et al., 2002  #####
#####   -----------------------------------------------------------------   #####
#####        Nature and Interaction of Signals From the Receptive Field     #####
#####               Center and Surround in Macaque V1 Neurons               #####
#####   -----------------------------------------------------------------   #####
#####              DOI : https://doi.org/10.1152/jn.00693.2001              #####
##################################################################################


def get_size_tuning_curves(
    single_model,
    x_pix,
    y_pix,
    preferred_ori,
    preferred_sf,
    preferred_phase,
    radii = np.logspace(-2,np.log10(2),40) ,
    contrast = 1,
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.67,
    img_res = [93,93],
    neg_val = True,     #TODO never used
    compute_annular = True
    ):
    
    ''' This function is a size tuning function that makes the size of circular and annular grating images vary 
        with their prefered parameters fixed. The difference between the response of the model for each image and the response for a gray image 
        is calculated and saved into an array of the same size of the argument "radii". You can afterwards plot these arrays
        to visualise the effect of the radii on the model's response. If neg_val is set to 'False', the negative values
        are set to 0.
        

        NB : We substract the response to gray image to the response of stimuli because it is necessary for the rest of our analysis, it normalises the values.
        NB2: What is called here 'annular' is actually not a ring since there is no outer boundary, the 'ring' fills the outer space of the image. 

        Arguments : 

            - single_model    : A single cell model of the class 'surroundmodulation.models.SingleCellModel'
            - x_pix           : The x coordinates of the center of the neuron's receptive field (in pixel, can be a float)
            - y_pix           : The y coordinates   ""        ""        ""
            - preferred_ori   : The preferred orientation (the one that elicited the greatest response)
            - preferred_sf    : The preferred spatial frequency
            - preferred_phase : The preferred phase
            - radii           : An array containing every radius size to test, NB : here the radius is from the center to the inner edge of the ring
            - Thickness       : The thickness of the annular item (half of the full thickness)
            - contrast        : The value that will multiply the image's values 
            - pixel_min       : Value of the minimal pixel that will serve as the black reference
            - pixel_max       : Value of the maximal pixel that will serve as the white reference (NB : The gray value will be the mean of those two)
            - device          : The device on which to execute the code, if set to "None" it will take the available one
        ?   - size            : The size of the image in terms of visual field degrees
            - img_res         : Resolution of the image in term of pixels shape : [pix_y, pix_x]
            - neg_val         : If set to 'False', the outputs won't include negative values but will replace them by 0.
            - compute_annular : If set to 'False', the function will only perform size tuning on a circular grating image

        Outputs : 

            - circular_tuning_curve  : An array containing the responses for every center stimulus minus the response to gray image
            
            - If compute_annular == True :
                
                - annular_tuning_curve   : An array containing the responses for every surround stimulus minus the response to gray image
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

    ## To save the responses
    circular_tuning_curve   = torch.zeros(len(radii))
    annular_tuning_curve = torch.zeros(len(radii))


    with torch.no_grad():

        ## Create Gray stimulus to substract to the non gray stimuli
        gray_stim = torch.ones((1, *img_res)).to(device) * ((pixel_min + pixel_max)/2)
        gray_resp = single_model(gray_stim)

        for i, radius in enumerate(radii) :
            
            ## Make the circular stimulus
            grating_center = torch.Tensor(imagen.SineGrating(mask_shape=imagen.Disk(smoothing=0.0, size=radius*2.0),
                                orientation=preferred_ori,
                                frequency=preferred_sf,
                                phase=preferred_phase,
                                bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                                offset = -1,
                                scale=2,  
                                xdensity=img_res[1]/size[1],
                                ydensity=img_res[0]/size[0], 
                                x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                                y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())
            
            ## Mask the circular stimulus 
            center_mask = torch.Tensor(imagen.Disk(smoothing=0.0,
                                size=radius*2.0, 
                                bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                                xdensity=img_res[1]/size[1],
                                ydensity=img_res[0]/size[0], 
                                x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                                y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())
            
            ## Add the gray background to the circular stimulus
            grating_center = grating_center * center_mask

            ## Change the contrast for the circle
            grating_center *= contrast
            
            ## Convert to the right shape for the model
            grating_center = rescale(grating_center,-1,1,pixel_min,pixel_max).reshape(1,*img_res).to(device)


            ## Save responses for this radius, substract the gray response
            if neg_val == False :
                ## Avoid negative values
                circular_tuning_curve[i] = torch.maximum(single_model(grating_center) - gray_resp, torch.tensor(0))

            else : 
                ## Allow negative values
                circular_tuning_curve[i] = single_model(grating_center) - gray_resp

            ## Same for the annular stimulus
            if compute_annular : 
                ## Make the annular stimulus
                ## Create a grating background
                grating_background = torch.Tensor(imagen.SineGrating(
                                    orientation=preferred_ori,
                                    frequency=preferred_sf,
                                    phase=preferred_phase,
                                    bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                                    offset = -1,
                                    scale=2,   
                                    xdensity=img_res[1]/size[1],
                                    ydensity=img_res[0]/size[0], 
                                    x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                                    y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())
                

                ## Create the inner edge of the ring
                inner_edge = (torch.Tensor(imagen.Disk(
                                    smoothing=0.0, 
                                    size=(radius)*2.0,
                                    bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))), 
                                    xdensity=img_res[1]/size[1],
                                    ydensity=img_res[0]/size[0], 
                                    x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                                    y = -get_offset_in_degr(y_pix, img_res[0], size[0]))()) * -1) + 1
                grating_ring = grating_background * inner_edge

                ## Change the contrast for the ring
                grating_ring   *= contrast
                
                ## Convert to the right shape for the model 
                grating_ring = rescale(grating_ring,-1,1,pixel_min,pixel_max).reshape(1,*img_res).to(device)
                
                ## Save responses for this radius, substract the gray response
                if neg_val == False :
                    ## Avoid negative values
                    annular_tuning_curve[i]  = torch.maximum(single_model(grating_ring) - gray_resp,torch.tensor(0))   

                else : 
                    ## Allow negative values
                    annular_tuning_curve[i]  = single_model(grating_ring) - gray_resp
                   

    if compute_annular :
        return circular_tuning_curve, annular_tuning_curve
    
    else :
        return circular_tuning_curve
    

def get_GSF_surround_AMRF(
    radii,
    circular_tuning_curve,
    annular_tuning_curve = None
    ): 

    ''' This function takes the circular and annular tuning curves outputs of the function "get_size_tuning_curves".
        For the circular stimuli : 
            -It finds the Grating Summation Field (GSF) defined as the "diameter of the smallest stimulus that elicited at least 95% of the
            neuron’s maximum response".
            -It finds the surround extent radius defined as the "inhibitory surround extent as the diameter of the smallest stimulus 
            for which the neuron’s response was reduced to within 5% of its asymptotic value for the largest gratings".
            -It computes the supression index defined as "(Ropt - Rsupp) / Ropt", where Ropt is the response at the GSF and Rsupp is the asymptotic response.
        For the annular stimuli :
            -It finds the Annular Minimum Response Field (AMRF) defined as "the point at which the response to the annular stimulus reached
            a value of at most 5% of the neuron’s maximum response to a circular patch of grating."
        
        Optionally, if no annular_tuning_curve is given, this function will not returns the AMRF

        Arguments : 

            - radii                  : The array that contains the radii that were used to obtain circular_tuning_curve and annular_tuning_curve
            - circular_tuning_curve  : An array, the center stimulus size tuning curve
            - annular_tuning_curve   : An array, the surround stimulus size tuning curve

        Outputs : 

            - GSF             : Optimal circular radius that elicits approximately the maximal response
            - surround_extent : Optimal circular radius that elicits approximately the asymptotic response (suppression)
            - AMRF            : Optimal annular radius that elicits approximately the lowest response
            - Ropt            : The response of the model at the GSF for the circular stimuli
            - Rsupp           : The asymptotic response of the model for the circular stimuli

            NB : outputs are radii, not diameters
    '''
    
    ## Make sure the elements are tensors
    circular_tuning_curve = torch.as_tensor(circular_tuning_curve)
    if annular_tuning_curve is not None :
        annular_tuning_curve = torch.as_tensor(annular_tuning_curve)

    ## Avoid zeros
    GSF             = 1.e-9
    surround_extent = 1.e-9
    AMRF            = 1.e-9
    Ropt            = 0
    Rsupp           = circular_tuning_curve[-1] #The last response = asymptotic value

    ## Select the thresholds 
    GSF_thresh   = (95 * torch.max(circular_tuning_curve)) / 100
    a = Rsupp - ((5 * Rsupp) /100)
    b = Rsupp + ((5 * Rsupp) /100)
    surround_thresh_min = min(a,b)
    surround_thresh_max = max(a,b)
    AMRF_thresh  = 5 * torch.max(circular_tuning_curve) / 100
    
    ## For the GSF
    for i, resp in enumerate(circular_tuning_curve):
        if resp > GSF_thresh :
            GSF = radii[i]
            Ropt = resp
            break

    ## For the surround extent
    for i, resp in enumerate(circular_tuning_curve):
        ## Avoid to take values with smaller radius
        if radii[i] > GSF :
            ## Assert that the response is decreasing
            if resp < Ropt :
                if resp >= surround_thresh_min and resp <= surround_thresh_max:
                    surround_extent = radii[i]
                    break
    
    if annular_tuning_curve != None :
        ## For the AMRF
        for i, resp in enumerate(annular_tuning_curve):
            if resp < AMRF_thresh:
                AMRF = radii[i]
                break
    
    
    SI = ((Ropt - Rsupp) / Ropt).item()


    if annular_tuning_curve is not None :
    
        return GSF, surround_extent, AMRF, SI, Ropt, Rsupp
    
    else :
        
        return GSF, surround_extent, SI, Ropt, Rsupp
    

def size_tuning_experiment_all_phases(
    h5_file,
    all_neurons_model,
    neuron_ids,
    overwrite = False, 
    radii = np.logspace(-2,np.log10(2),40),
    phases = np.linspace(0, 2*np.pi, 37)[:-1],
    contrast = 1,
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.67,
    img_res = [93,93],
    neg_val = True
    ):

    ''' This function performs a size tuning experiment for the neurons accross multiple phases :
        (we have to compute accross every phase because changing the images radius shifts the phase, and we want to counter that)  

            1) For each phase this function :                                                              

                - Gets the tuning curves for two grating images, one circular and one annular (see 'get_size_tuning_curves')

                - Computes the Grating Summation Field (GSF) and other elements (see 'get_GSF_surround_AMRF')
            
            2) It then takes the results obtained with the phase that lead to the maximal response 

            3) Finally it saves the size tuning curves and the results for each neuron in two subgroups in the HDF5 file.

        It saves the data with this architecture : 


                                _________ SubGroup /size_tuning/curves   --> neuron datasets
                                |
        Group /size_tuning  ____|
                                |
                                |________ SubGroup /size_tuning/results  --> neuron datasets


        Prerequisite :

            - function 'get_all_grating_parameters' executed for the required neurons with matching arguments
            - function 'get_preferred_position'     executed for the required neurons with matching arguments


        Arguments :

            - radii                     : An array containing every radius to test
            - others                    : Described in other functions
            - neg_val                   : If set to 'False', this will set the negative responses encountered to zero (it should not change the results, only the curves). Negative values occurs when a stimulus elicits less response than the baseline (gray screen)
            

        Outputs :

            - datasets in ../results       : An array containing the results of this analysis for a neuron 
                                             format = [preferred_phase, GSF, surround_extent, AMRF, SI]
            NB  : The Value of GSF, surround_extant and AMRF are radii size (in deg), not diameters

            - datasets in ../curves : A matrix containing the circular patch tuning curve in the first row
                                                                 the annular patch tuning curve in the second row
    '''
    
    print(' > Size tuning experiment')
    
  
    ## Required groups
    group_path_ff_params   = "/full_field_params"
    group_path_pos         = "/preferred_pos"

    ## Groups to fill
    group_path_size_tuning      = "/size_tuning"
    subgroup_path_results       = group_path_size_tuning + "/results"
    subgroup_path_tuning_curves = group_path_size_tuning + "/curves"

    ## Clear the group if requested
    if overwrite : 
        clear_group(h5_file,group_path_size_tuning)

    ## Initialize the Group and the subgroups
    args_str = f"radii={radii}/phases={phases}/contrast={contrast}/pixel_min={pixel_min}/pixel_max={pixel_max}/size={size}/img_res={img_res}/neg_val={neg_val}"
    group_init(h5_file=h5_file, group_path=group_path_size_tuning, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_results, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_tuning_curves, group_args_str=args_str)

    ## Check compatibility between every experiment
    check_compatibility(h5_file=h5_file, list_group_path=[group_path_ff_params, group_path_pos], neuron_ids=neuron_ids, group_args_str=args_str)


    with h5py.File(h5_file,'a') as file :

        ## Access the groups 
        subgroup_results  = file[subgroup_path_results]
        subgroup_tc       = file[subgroup_path_tuning_curves]
        group_size_tuning = file[group_path_size_tuning]
        group_ff_params   = file[group_path_ff_params]
        group_pos         = file[group_path_pos]

        ## For every neuron
        for neuron_id in tqdm(neuron_ids):

            neuron = f"neuron_{neuron_id}"

            ## Check if the neuron data is not already present
            if neuron not in subgroup_results :

                ## Get the model for the neuron
                single_model = SingleCellModel(all_neurons_model,neuron_id )

                ## Get the neuron's preferred parameters 
                max_ori = group_ff_params[neuron][:][0]
                max_sf  = group_ff_params[neuron][:][1]
                max_x   = group_pos[neuron][:][0]
                max_y   = group_pos[neuron][:][1]
                            
                ## Initialise the parameters to find
                GSF_preferred_phase = 0
                surround_extent_preferred_phase = 0
                AMRF_preferred_phase = 0
                SI_preferred_phase = 0

                Rmax = 0 

                for phase in phases :
                    ## Get the tuning curve for the current phase
                    circular_tuning_curve, annular_tuning_curve = get_size_tuning_curves(
                        single_model = single_model,
                        x_pix = max_x,
                        y_pix = max_y,
                        preferred_ori = max_ori,
                        preferred_sf = max_sf,
                        preferred_phase = phase,
                        radii = radii,
                        contrast = contrast,
                        pixel_min = pixel_min,
                        pixel_max = pixel_max,
                        device = device,
                        size = size,
                        img_res = img_res,
                        neg_val = neg_val,
                        compute_annular = True ## This function needs the annular tuning curve
                        )
                    
                    ## Perform some analysis
                    GSF, surround_extent, AMRF, SI, Ropt, _ = get_GSF_surround_AMRF(
                        radii = radii,
                        circular_tuning_curve = circular_tuning_curve,
                        annular_tuning_curve = annular_tuning_curve
                        )
                    
                    if Ropt>Rmax :
                        ## Results
                        preferred_phase = phase
                        GSF_preferred_phase = GSF
                        surround_extent_preferred_phase = surround_extent
                        AMRF_preferred_phase = AMRF
                        SI_preferred_phase = SI
                        Rmax = Ropt

                        ## Curves
                        circular_tc_pref = circular_tuning_curve
                        annular_tc_pref  = annular_tuning_curve



                ## Merge the circular and annular tuning curves together (row 1 = circular, row 2 = annular)
                both_curves = torch.stack((circular_tc_pref, annular_tc_pref))
 
                ## Save the tuning curve in the subgroup '/size_tuning/curves
                subgroup_tc.create_dataset(name=neuron, data=both_curves.cpu())

                ## Save the results ([preferred_phase, GSF, surround_extent, AMRF, SI])
                subgroup_results.create_dataset(name=neuron, data=[preferred_phase, GSF_preferred_phase, surround_extent_preferred_phase, AMRF_preferred_phase, SI_preferred_phase])


def get_contrast_response(
    single_model,
    x_pix,
    y_pix,
    center_radius,
    surround_radius,
    preferred_ori,
    preferred_sf,
    preferred_phase,
    center_contrasts = np.logspace(-2,np.log10(1),18),  
    surround_contrasts = np.logspace(-2,np.log10(1),18),
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.67,
    img_res = [93,93],
    neg_val = True
    ):
    
    ''' This function takes the preferred parameters for the orientation, spatial frequency and phase to create images with a circular center 
        and an annular surround. The radius of the center and annular is fixed, we advice to take the GSF for the center radius and the AMRF 
        for the surround radius (or the GSF if the AMRF is smaller than the GSF). 
        The goal of this function is to try different combination of center and surround contrast for the given model (unique neuron),
        get their response, then normalise the values by substracting the response to gray stimulus.
        Optionally (if neg_val == False) it replaces the negative values by zero (neg values are coming from the normalisation).

        NB  : This function uses the SingleNeuronModel class and the v1_convnext_ensemble
        NB2 : Surround_radius is the radius from the center to the inner edge of the ring
        NB3 : For this function, you can't use surround diameters smaller than center diameter (if overlapping, the center takes advantage)

        Arguments :

            - neg_val : Bool, if set to false, replace every negative value to 0
        
        Outputs : 

            - contrast_resps_mat : A matrix that contains the response of the model (neuron) for every surround contrast (rows) and center contrast (cols)
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
    
    ## Avoid surround and center to overlap
    if surround_radius < center_radius :
        surround_radius = center_radius


    contrast_resps_mat = torch.zeros((len(surround_contrasts), len(center_contrasts))).to(device)

    with torch.no_grad():
        ## Get the response of the model to gray stimulus
        gray_stim = torch.ones((1,93,93)).to(device) * ((pixel_min + pixel_max)/2)
        gray_resp = single_model(gray_stim)

        ## For every combination of contrasts
        for i, surround_contrast in enumerate(surround_contrasts):
            for j, center_contrast in enumerate(center_contrasts) : 
              
                ## Get the image in the right shape for the model
                image = get_center_surround_stimulus(center_radius=center_radius, center_ori=preferred_ori, center_sf=preferred_sf, center_phase=preferred_phase, center_contrast=center_contrast, surround_radius=surround_radius, surround_ori=preferred_ori, surround_sf=preferred_sf, surround_phase=preferred_phase, surround_contrast=surround_contrast, x_pix=x_pix, y_pix=y_pix, pixel_min=pixel_min, pixel_max=pixel_max, size=size, img_res=img_res, device=device)

                ## Get the model's response 
                resp = single_model(image)

                ## Normalise the response
                resp_norm = resp - gray_resp

                ## Save the response
                contrast_resps_mat[i,j] = resp_norm

    ## Optional : replace negative values by zero
    if neg_val == False :     
        contrast_resps_mat = torch.maximum(contrast_resps_mat, torch.zeros(contrast_resps_mat.shape).to(device))

    return contrast_resps_mat


def contrast_response_experiment(
    h5_file,
    all_neurons_model,
    neuron_ids,   
    overwrite = False, 
    center_contrasts = np.logspace(-2,np.log10(1),18),
    surround_contrasts = np.logspace(-2,np.log10(1),6),
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.67,
    img_res = [93,93],
    neg_val = True
    ):
    ''' For the selected neurons this function :

            1) Performs 'get_contrast_response' function to get the response of each neuron to different center and surround contrasts.

            2) Analysis for each result : it gets the last point of every curves and calculate the difference between the highest and
            the lowest value ('max/min'), also computes the mean standard deviation accross every points ('mean_std')
            Those values aim to characterise how the neuron is affected by the surround contrast (if the points are far from each other, that means that the response is affected by the surround contrast, the std and max/min ratio values will then be higher)

            3) For each neuron it saves the contrast response curves and the results of the analysis in different subgroups

        It saves the data with this architecture : 


                                      _________ SubGroup /contrast_response/curves   --> neuron datasets
                                      |
        Group /contrast_response  ____|
                                      |
                                      |________ SubGroup /contrast_response/results  --> neuron datasets
        
        Prerequisite : 

            - function 'get_all_grating_parameters'        executed for the required neurons with matching arguments
            - function 'get_preferred_position'            executed for the required neurons with matching arguments
            - function 'size_tuning_experiment_all_phases' executed for the required neurons with matching arguments

        Arguments :

            - center_contrasts   : An array containing the contrast values to test for the circular (center) stiumulus
            - surround_contrasts :      ""      ""      ""      ""      ""      ""      "" annular (surround) stiumulus
            - others             : Described in other functions

        Outputs :


            - datasets in ../curves     : A matrix that contains the response of the model (neuron) for every surround contrast (rows) and center contrast (cols) (see 'get_contrast_response')

            - datasets in ../results    : An array containing the results of this analysis for each neuron
                                          format = [max/min, mean_std] 
                                                
    '''
    print(' > Contrast response experiment')
    
    ## Required groups
    group_path_ff_params   = "/full_field_params"
    group_path_pos         = "/preferred_pos"
    group_path_st_results  = "/size_tuning/results"

    ## Groups to fill
    group_path_cr            = "/contrast_response"
    subgroup_path_cr_results = group_path_cr + "/results"
    subgroup_path_cr_curves  = group_path_cr + "/curves"

    ## Clear the group if requested
    if overwrite : 
        clear_group(h5_file,group_path_cr)

    ## Initialize the Group and the subgroups
    args_str = f"center_contrasts={center_contrasts}/surround_contrasts={surround_contrasts}/pixel_min={pixel_min}/pixel_max={pixel_max}/size={size}/img_res={img_res}/neg_val={neg_val}"
    group_init(h5_file=h5_file, group_path=group_path_cr, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_cr_results, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_cr_curves, group_args_str=args_str)

    ## Check compatibility between every experiment
    check_compatibility(h5_file=h5_file, list_group_path=[group_path_ff_params, group_path_pos,group_path_st_results], neuron_ids=neuron_ids, group_args_str=args_str)


    with h5py.File(h5_file,'a') as file :

        ## Access the groups 
        group_cr            = file[group_path_cr]
        subgroup_cr_results = file[subgroup_path_cr_results]
        subgroup_cr_curves  = file[subgroup_path_cr_curves]
        group_ff_params     = file[group_path_ff_params]
        group_pos           = file[group_path_pos]
        group_st_results    = file[group_path_st_results]

        
        for neuron_id in tqdm(neuron_ids) :
            
            neuron = f"neuron_{neuron_id}"

            ## Check if the neuron data is not already present
            if neuron not in subgroup_cr_results :

                ## Get the single_model of the neuron
                single_model = SingleCellModel(all_neurons_model, neuron_id)

                ## Get the preferred parameters of the neuron
                preferred_phase = group_st_results[neuron][:][0]
                GSF             = group_st_results[neuron][:][1]
                AMRF            = group_st_results[neuron][:][3]
                preferred_ori   = group_ff_params[neuron][:][0]
                preferred_sf    = group_ff_params[neuron][:][1]
                x_pix           = group_pos[neuron][:][0]
                y_pix           = group_pos[neuron][:][1]

                ## Avoid overlapping
                if GSF>AMRF :
                    AMRF = GSF
                
                ## Get the contrast response curves (matrix)
                contrast_response_mat = get_contrast_response(
                    single_model = single_model,
                    x_pix = x_pix,
                    y_pix = y_pix,
                    center_radius = GSF,
                    surround_radius = AMRF,
                    preferred_ori = preferred_ori,
                    preferred_sf = preferred_sf,
                    preferred_phase = preferred_phase,
                    center_contrasts = center_contrasts,
                    surround_contrasts = surround_contrasts,
                    pixel_min = pixel_min,
                    pixel_max = pixel_max,
                    device = device,
                    size = size,
                    img_res = img_res,
                    neg_val = neg_val
                    )

                ## Capture how the curves are spread
                last_points = contrast_response_mat[:,-1]
                min = torch.min(last_points)
                max = torch.max(last_points)
                ratio = (max/min).cpu().item()
                
                mean_std = torch.mean(torch.std(contrast_response_mat, axis = 0)).cpu().item()
                
                ## Save the curves
                subgroup_cr_curves.create_dataset(name=neuron, data=contrast_response_mat.cpu())

                ## Save the results
                data = [ratio, mean_std]
                subgroup_cr_results.create_dataset(name=neuron, data=data)


def get_contrast_size_tuning_curve_all_phases(
    single_model,
    x_pix,
    y_pix,
    preferred_ori,
    preferred_sf,
    phases = np.linspace(0, 2*np.pi, 37)[:-1],
    contrasts = np.logspace(np.log10(0.06),np.log10(1),5),
    radii = np.logspace(-2,np.log10(2),40) ,
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.67,
    img_res = [93,93],
    neg_val = True
    ):
    ''' For a given neuron, this function aims to create multiple size tuning curves, for different contrasts.
        This function uses the 'get_size_tuning_curves' to get the tuning curves.
        The outputs of this function are two matrices, one for the circular stimulus tuning curve, the other for the annular stimulus one.
        The matrices are created as so : every row correspond to a contrast value, every column correspond to a radius.

        Arguments :

            - contrasts : Every contrast to test (will be the rows of the matrices)
            - others    : See 'get_size_tuning_curves' arguments

        Outputs :

            - all_circular_curves : For the circular stimuli : the matrix containing the response of the neuron for every combination of contrast (row) and radius (column)
            - all_annular_curves  : For the annular stimuli  : Same

        NB : since we perform accross every phase, the responses set in the matrices are the best response accross every phase
    '''

    all_circular_curves = torch.zeros((len(phases),len(contrasts), len(radii)))
    
    ## For every phase :
    for num_phase in range(len(phases)) :
        
        ## For every contrast (every row of the matrices)
        for i,contrast in enumerate(contrasts) : 
            
            ## Get the size tuning curve
            circular_curve = get_size_tuning_curves(
                single_model = single_model,
                x_pix = x_pix,
                y_pix = y_pix,
                preferred_ori = preferred_ori,
                preferred_sf = preferred_sf,
                preferred_phase = phases[num_phase],
                radii = radii,
                contrast = contrast,
                pixel_min = pixel_min,
                pixel_max =  pixel_max, 
                device = device,
                size = size,
                img_res = img_res,
                neg_val = neg_val,
                compute_annular = False ## This function does not need the annular tuning curve
                )

            ## Fill the corresponding rows of the matrices
            all_circular_curves[num_phase ,i ,:] = circular_curve

    ## Get the maximum response accross phases
    all_circular_curves = torch.max(all_circular_curves, dim=0).values
    
    return all_circular_curves


def contrast_size_tuning_experiment_all_phases(
    h5_file,
    all_neurons_model,
    neuron_ids,   
    overwrite = False, 
    phases = np.linspace(0, 2*np.pi, 37)[:-1],
    contrasts = np.logspace(np.log10(0.06),np.log10(1),5),
    radii = np.logspace(-2,np.log10(2),40) ,
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.67,
    img_res = [93,93],
    neg_val = True
):
    ''' For the selected neurons this function :

            1) Performs 'get_contrast_size_tuning_curve_all_phases' function to get the size tuning curves of each neuron for different contrasts

            2) Analysis for each result : This function gets for each neuron the GSF values computed from size tuning experiments for the highest contrast (100%) and for the lowest contrast (6%)
                                          It then computes the shift of the GSF as a ratio GSFlow/GSFhigh
            
            3) For each neuron it saves the curves and the results of the analysis in different subgroups

        It saves the data with this architecture : 


                                         _________ SubGroup /contrast_size_tuning/curves   --> neuron datasets
                                         |
        Group /contrast_size_tuning  ____|
                                         |
                                         |________ SubGroup /contrast_size_tuning/results  --> neuron datasets
        
        Prerequisite : 

            - function 'get_all_grating_parameters'        executed for the required neurons with matching arguments
            - function 'get_preferred_position'            executed for the required neurons with matching arguments

        Arguments :

            - contrasts : Every contrast to test (will be the rows of the matrices)
            - others    : See other functions

        Outputs :


            - datasets in ../curves     : A matrix that contains the response of the model (neuron) for every contrast (row) and radius (col)

            - datasets in ../results    : An array containing the results of this analysis for each neuron
                                          format = [GSFs_ratio] with GSFs_ratio = GSFlow/GSFhigh
                                                
    '''
    print(' > Contrast size tuning experiment')

    ## Required groups
    group_path_ff_params   = "/full_field_params"
    group_path_pos         = "/preferred_pos"

    ## Groups to fill
    group_path_cst            = "/contrast_size_tuning"
    subgroup_path_cst_results = group_path_cst + "/results"
    subgroup_path_cst_curves  = group_path_cst + "/curves"

    ## Clear the group if requested
    if overwrite : 
        clear_group(h5_file,group_path_cst)

    ## Initialize the Group and the subgroups
    args_str = f"phases={phases}/contrasts={contrasts}/radii={radii}/pixel_min={pixel_min}/pixel_max={pixel_max}/size={size}/img_res={img_res}/neg_val={neg_val}"
    group_init(h5_file=h5_file, group_path=group_path_cst, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_cst_results, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_path_cst_curves, group_args_str=args_str)

    ## Check compatibility between every experiment
    check_compatibility(h5_file=h5_file, list_group_path=[group_path_ff_params, group_path_pos], neuron_ids=neuron_ids, group_args_str=args_str)

    with h5py.File(h5_file,'a') as file :

        ## Access the groups 
        group_cst                 = file[group_path_cst]
        subgroup_cst_results      = file[subgroup_path_cst_results]
        subgroup_cst_curves  = file[subgroup_path_cst_curves]
        group_ff_params     = file[group_path_ff_params]
        group_pos           = file[group_path_pos]

        for neuron_id in tqdm(neuron_ids) :
            
            neuron = f"neuron_{neuron_id}"

            ## Check if the neuron data is not already present
            if neuron not in subgroup_cst_results :

                ## Get the single_model of the neuron
                single_model = SingleCellModel(all_neurons_model, neuron_id)

                ## Get the preferred parameters of the neuron
                preferred_ori   = group_ff_params[neuron][:][0]
                preferred_sf    = group_ff_params[neuron][:][1]
                x_pix           = group_pos[neuron][:][0]
                y_pix           = group_pos[neuron][:][1]
                
                cst_curves = get_contrast_size_tuning_curve_all_phases(
                    single_model=single_model,
                    x_pix=x_pix,
                    y_pix=y_pix,
                    preferred_ori=preferred_ori,
                    preferred_sf=preferred_sf,
                    phases=phases,
                    contrasts=contrasts,
                    radii=radii,
                    pixel_min=pixel_min,
                    pixel_max=pixel_max, 
                    device=device,
                    size=size,
                    img_res=img_res,
                    neg_val=neg_val
                    )


                ## For low contrast  
                low_contrast_curve = cst_curves[0]
                GSF_low,_,_,_,_ = get_GSF_surround_AMRF(radii = radii,circular_tuning_curve=low_contrast_curve, annular_tuning_curve = None)
                ## For contrast =1
                high_contrast_curve = cst_curves[-1]
                GSF_high,_,_,_,_ = get_GSF_surround_AMRF(radii = radii,circular_tuning_curve=high_contrast_curve, annular_tuning_curve = None)

                GSFs_ratio = GSF_low/GSF_high

                ## Save the curves
                subgroup_cst_curves.create_dataset(name=neuron, data=cst_curves.cpu())

                ## Save the results
                data = [GSFs_ratio]
                subgroup_cst_results.create_dataset(name=neuron, data=data)



##################################################################################
#####  PART III : Experiments for the second article, Cavanaugh et al., 2002 #####
#####   ------------------------------------------------------------------   #####
#####        Selectivity and Spatial Distribution of Signals From the        #####
#####            Receptive Field   Surround in Macaque V1 Neurons            #####
#####   ------------------------------------------------------------------   #####
#####              DOI : https://doi.org/10.1152/jn.00693.2001               #####
##################################################################################


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

                    ## Get the corresponding orientation  
                    most_suppr_surr[i] = (sub_ori[argmin])

                ## Save every result in the HDF5 file 
                subgroup_curves_center.create_dataset(name=neuron, data=center_tuning_curve)
                subgroup_curves_surround.create_dataset(name=neuron, data=surround_tuning_curves)
                subgroup_results.create_dataset(name=neuron, data=most_suppr_surr)


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



##################################################################################
#####  PART IV : Experiments for the third article, Chun-I Yeh et al., 2009  #####
#####   -------------------------------------------------------------------  #####
#####        “Black” Responses Dominate Macaque Primary Visual Cortex V1     #####
#####   -------------------------------------------------------------------  #####
#####              DOI : https://doi.org/10.1523/JNEUROSCI.1991-09.2009      #####
##################################################################################


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
    device = None,
    seed = 42
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
        Group /black_white_preference   ________|________ SubGroup ../noise_img               --> neuron datasets
                                                |
                                                |________ SubGroup ../results                 --> neuron datasets
    Prerequisite :

        - None

    Arguments :

        - dot_size_in_pixels : Size of the pixels, corresponding to the size of the sides of the square
        - seed               : Random seed for reproducibility
        - others             : explained in other functions

    Outputs :

        - datasets in ../position_response_img  : A tensor containing two images (matrices). The first one is for black dots stimulation, the second one for white dot stimulation 
                                                  every pixel of the images correspond to the summed response of the neuron to this pixel

        - datasets in ../noise_img              :  A tensor containing two images (matrices). The first one is for black dots stimulation, the second one for white dot stimulation 
                                                   every pixel of the images correspond to the summed shuffled response of the neuron to this pixel (a shuffled response means that the response assigned to every dot can now be the response to another dot)


        - datasets in ../results                : An array containing the Signal noise ratios values and the log10(SNRw / SNRb)
                                         format : [SNR_b, SNR_w, logSNRwb]

    '''

    np.random.seed(seed)

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
        shuffle_resp_w = torch.Tensor(shuffle_resp_w).to(device)
        
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



###############################################################################
#####  PART V : Experiments for the fourth article, Freeman et al., 2013  #####
#####   ---------------------------------------------------------------   #####
#####             A functional and perceptual signature of the            #####
#####                    second visual area in primates                   #####
#####   ---------------------------------------------------------------   #####
#####              DOI : https://doi.org/10.1038/nn.3402                  #####
###############################################################################


def random_crop_img(
    img, 
    target_res=[93,93]
):
    ''' This function aims to crop a subpart of the image in order to have the texture at the desired resolution 

        Arguments : 

            - img         : The source img, it's resolution should be greater than the target resolution
            - target_res  : The desired resolution for the cropped image

        Outputs :

            - cropped_img : The cropped img
    '''

    ## Get the former resolution
    y_res, x_res = img.shape

    ## Get a random initial position (top left pixel of the image)
    init_y = np.random.randint(0,y_res - target_res[0] +1)
    init_x = np.random.randint(0,x_res - target_res[1] +1)

    ## Crop the image
    cropped_img = img[init_y:init_y+target_res[0], init_x:init_x+target_res[1]]

    return cropped_img    


def load_imgs(
    directory_imgs, 
    target_res = [93,93],
    contrast = 1,  
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    num_samples = 15,
    device = None
):  
    ''' This function takes a folder path, loads and sorts every images by category (texture or noise), family, and sample
        For the images it crops them to the correct resolution, changes the contrast and rescale the images to the wanted values
    
        The name of the images should be formated as so : 

            'tex-320x320-im13-smp2.png'
            1     2      3    4   5

            - 1 : 'tex' if texture, 'noise' if noise
            - 2 : 'resolution_y' + 'x' + 'resolution_x'
            - 3 : 'im'  + number, the number shows to which texture familiy it corresponds
            - 4 : 'smp' + number, the number shows to which sample inside this family it corresponds
            - 5 : '.png' image should be in png
            - between each, there should be '-'
        
        Important :

            - In your folder, there should be exactly num_samples samples for every family
            - The resolution selected in 'target_res' should be equal or lower than the loaded images resolution

        Arguments :

            - directory_imgs : The path of the folder containing every images to load
            - target_res     : The desired resolution of the images (if the loaded images are in a higher dimention, it crops the image)
            - contrast       : The contrast to apply on the images, 1 means it does not change
            - pixel_min      : Value of the minimal pixel that will serve as the black reference
            - pixel_max      : Value of the maximal pixel that will serve as the white reference (NB : The gray value will be the mean of those two)
            - num_samples    : The number of samples for each texture family
            - device         : The device on which to execute the code, if set to "None" it will take the available one
            
            - TODO           : Add random seed to avoid randomness

        Outputs : 

            - tex_imgs       : The tensor containing the texture images,              shape = [number of family, number of samples, resolution y, resolution x]
            - noise_imgs     : The tensor containing the corresponding noise images,     ''  ''  ''  ''  ''  ''  ''  ''  ''  ''  ''  ''  ''  ''  ''  ''  ''  ''
    
    '''

    ## Select device
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Get a list of all files in the directory
    list_imgs_names = os.listdir(directory_imgs)
    
    # Filter out only the image files
    list_imgs_names = [f for f in list_imgs_names if f.endswith(('.png'))] #TODO maybe allow other formats 
    
    ## Get every unique family
    dict_fam = {}
    for i, img_name in enumerate(list_imgs_names):

        ## Get the family for each image
        _, _, family, _ = img_name.split('-')
        family = int(family[2:])
        
        dict_fam[family] = 0

    ## Set the ids for every family
    for i, family in enumerate(dict_fam.keys()) : 
        dict_fam[family] = i

    ## Create a Tensor containing every image and noise sorted correctly
    tex_imgs   = torch.zeros((len(dict_fam.keys()), num_samples, *target_res)).to(device)
    noise_imgs = torch.zeros(tex_imgs.shape).to(device)

    print("   > loading images ...")
    ## Get every unique family
    for i, img_name in tqdm(enumerate(list_imgs_names)):
                
        ## Get the image informations
        category, _, family, sample = img_name.split('-')
        # load_img_res = [int(load_img_res.split('x')[0]), int(load_img_res.split('x')[1])]
        family = int(family[2:])
        sample = int(sample[3:-4])
        fam_id = dict_fam[family]
        smp_id = sample -1

        img_path = directory_imgs + '/' + img_name

        img = torch.Tensor(mpimg.imread(img_path))

        ## Crop the image to the correct dimention
        cropped_img = random_crop_img(img=img,target_res=target_res)

        if category == 'tex' :
            tex_imgs[fam_id, smp_id] = cropped_img
        else : 
            noise_imgs[fam_id, smp_id] = cropped_img

    ## Change the contrast and rescale to the wanted values
    min_val = torch.min(torch.min(tex_imgs), torch.min(noise_imgs))
    max_val = torch.max(torch.max(tex_imgs), torch.max(noise_imgs))
    
    for i in range(len(tex_imgs)) :

        for j in range(num_samples) :
            
            ## Get the images
            tex_img   = tex_imgs[i,j]
            noise_img = noise_imgs[i,j]

            # ## Change the contrast and rescale
            tex_img   = rescale(tex_img, min_val, max_val, -1, 1)*contrast
            tex_img   = rescale(tex_img, -1, 1, pixel_min, pixel_max)
            noise_img = rescale(noise_img, min_val, max_val, -1, 1)*contrast
            noise_img = rescale(noise_img, -1, 1, pixel_min, pixel_max)

            ## Update the tensor
            tex_imgs[i,j]   = tex_img
            noise_imgs[i,j] = noise_img

    return tex_imgs, noise_imgs, dict_fam

 
def texture_noise_response_experiment(
    h5_file,
    all_neurons_model,
    neuron_ids,
    directory_imgs,
    overwrite = False,
    contrast = 1,
    pixel_min = -1.7876, 
    pixel_max =  2.1919, 
    num_samples = 15,
    img_res = [93,93],
    device = None

):
    ''' This function aims to get the responses of a model to texture and noise images :

            1) Take a folder directory and load every images into two tensors, one for the textures one fore the noises.
            2) Get the responses of every image
            3) Save the responses in a HDF5 file

        It saves the data with this architecture :         

                                                    _________ SubGroup ../texture   --> neuron datasets
                                                    |                        
            Group /texture_noise_response    _______| 
                                                    |
                                                    |________ SubGroup ../noise     --> neuron datasets
                                                  
        Prerequisite :

            - None

        Arguments : 

            - h5_file      : Path to the HDF5 file
            - all_neurons_model : The full model containing every neurons (In our analysis it correspond to the v1_convnext_ensemble)
            - neuron_ids        : The list (or array) of the neurons we wish to perform the analysis on
            - overwrite         : If set to True, will erase the data in the group "full_field_params" and then fill it again. 
            - other             : See 'load_imgs' description

        Outputs :

            - datasets in ../texture       : A matrix containing the responses of a neuron to the texture images. Each row correspond to a texture family, each column correspond to a sample

            - datasets in ../noise         : Same for noise images

    '''
    
    print(' > Texture and noise response experiment')

    ## Groups to fill
    group_path          = "/texture_noise_response"
    subgroup_tex_path   = group_path + "/texture"
    subgroup_noise_path = group_path + "/noise"

    ## Clear the group if requested    
    if overwrite : 
        clear_group(h5_file,group_path)


    ## Initialize the Group and subgroup
    args_str = f"contrast={contrast}/pixel_min={pixel_min}/pixel_max{pixel_min}/num_samples={num_samples}/img_res={img_res}"
    group_init(h5_file=h5_file, group_path=group_path, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_tex_path, group_args_str=args_str)
    group_init(h5_file=h5_file, group_path=subgroup_noise_path, group_args_str=args_str)

    ## Select device
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_neurons_model.to(device)
    all_neurons_model.eval()

    ## Load the images
    tex_imgs, noise_imgs, dict_fam = load_imgs(directory_imgs=directory_imgs, target_res=img_res, contrast=contrast, pixel_min=pixel_min, pixel_max=pixel_max, num_samples=num_samples, device=device)

    tex_resp   = []
    noise_resp = []

    ## Get the model's responses
    with torch.no_grad():

        ## Get the responses
        for i in range(len(tex_imgs)) :
            
            tex_resp.append(all_neurons_model(tex_imgs[i]))
            noise_resp.append(all_neurons_model(noise_imgs[i]))

    ## Convert the lists of tensors into single tensors
    tex_resp   = torch.stack(tex_resp).cpu()
    noise_resp = torch.stack(noise_resp).cpu()

    ## Fill the HDF5 file
    with h5py.File(h5_file, 'a') as f :
            
        ## Access the groups
        subgroup_tex = f[subgroup_tex_path]
        subgroup_noise = f[subgroup_noise_path]

        ## Add a description for the families
        description = ''
        for family in dict_fam.keys() :
            description += f'{family}-'
        description = description[:-1]
            

        f[group_path].attrs["description"]  = description
        subgroup_tex.attrs["description"]   = description
        subgroup_noise.attrs["description"] = description

        for neuron_id in tqdm(neuron_ids) : 
            
            neuron = f"neuron_{neuron_id}"

            ## Check if the neuron data is not already present
            if neuron not in subgroup_noise :

                neuron_results_tex   = tex_resp[:,:,neuron_id]
                neuron_results_noise = noise_resp[:,:,neuron_id]

                subgroup_tex.create_dataset(name=neuron, data=neuron_results_tex)
                subgroup_noise.create_dataset(name=neuron, data=neuron_results_noise)