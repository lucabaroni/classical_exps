#%%

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
## Image generation
import imagen
from imagen.image import BoundingBox
## Important functions
from surroundmodulation.utils.misc import rescale
from surroundmodulation.utils.plot_utils import plot_img
from scipy.optimize import curve_fit
## Import models
from nnvision.models.trained_models.v1_task_fine_tuned import v1_convnext_ensemble
from surroundmodulation.models import SingleCellModel


####################################################################################
#####    THIS CODE CONTAINS EVERY FUNCTION USED TO PERFORM THE EXPERIMENTS.    #####
#####           ----------------------------------------------------           #####
##### Those function works with a HDF5 file where they all store and get data  #####
####################################################################################



########################################################
##### PART I : Functions that perform pre-analyses #####
########################################################

def find_preferred_grating_parameters_background(
    model, 
    orientations = np.linspace(0, np.pi, 37)[:-1], 
    spatial_frequencies = np.linspace(1, 7, 25), 
    phases = np.linspace(0, 2*np.pi, 37)[:-1],
    contrast = 0.2,         
    img_res = [93,93], 
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.35,           
    ):

    ''' 
        This function generates grating images with every parameters combination and 
        returns the parameters that lead to the greatest activation in the Single Neuron Model.

        This function differs from "find_preferred_grating_parameters" only in the fact
        that the grating is not only presented in a small part of the image but in the 
        whole image.

        Arguments : 

            - model               : A single cell model of the class 'surroundmodulation.models.SingleCellModel'
            - orientations        : An array containing the orientations of the grating to test (in rad)
        ?   - spatial_frequencies : An array containing the spatial frequencies to test (it correspond to the number of grating cycles per degree of excentricity)
            - phases              : An array containing the phases of the grating to test
            - contrast            : The value that will multiply the grating image's values (lower than 1 will reduce the contrast and greater than 1 will increase the contrast)
            - img_res             : Resolution of the image in term of pixels [nb_y_pix, nb_x_pix]
        ?   - pixel_min           : Value of the minimal pixel that will serve as the black reference
        ?   - pixel_max           : Value of the maximal pixel that will serve as the white reference (NB : The gray value will be the mean of those two)
            - device              : The device on which to execute the code, if set to "None" it will take the available one
        ?   - size                : The size of the image in terms of visual field degrees

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
    model.to(device)

    ## Evaluation mode
    model.eval()

    ## Convert the size to the right shape ( [size_height, size_width] )
    if type(size) is int or type(size) is float: 
        size = [size]*2
    
    resp_max = 0


    ## Get every parameters combination
    with torch.no_grad():
        for orientation in tqdm(orientations):
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

                    resp = model(grating.reshape(1,1,*img_res))

                    if resp>resp_max:
                        resp_max = resp
                        max_ori = orientation
                        max_phase = phase
                        max_sf = sf
                        stim_max = grating
        
    return max_ori, max_sf, max_phase, stim_max, resp_max

def find_all_grating_parameters(
    all_neurons_model,
    neuron_ids,
    orientations = np.linspace(0, np.pi, 37)[:-1], 
    spatial_frequencies = np.linspace(1, 7, 25), 
    phases = np.linspace(0, 2*np.pi, 37)[:-1],
    contrast = 1,         
    img_res = [93,93], 
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.35,           
    ):

    '''
    This function returns a DataFrame containing the preferred
      parameters for every neuron of the general model

    '''

    ori = []
    sf = []
    phase = []
    stim = []
    resp = []

    for id in tqdm(neuron_ids) :

        model = SingleCellModel(all_neurons_model,id)

        max_ori, max_sf, max_phase, stim_max, resp_max = find_preferred_grating_parameters_background(
            model = model, 
            orientations = orientations, 
            spatial_frequencies = spatial_frequencies, 
            phases = phases,
            contrast = contrast,         
            img_res = img_res, 
            pixel_min = pixel_min,
            pixel_max = pixel_max,
            device = device,
            size = size,           
            )

        ori.append(max_ori)
        sf.append(max_sf)
        phase.append(max_phase)
        stim.append(stim_max)
        resp.append(resp_max)

    

    all_params = pd.DataFrame({
    'index': neuron_ids,
    'orientation': ori,
    'sf': sf,
    'phase_full_field': phase,
    })

    return all_params


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

        - all_neurons_model  : In our anaalyse it correspond to the v1_convnext_ensemble
        - dot_size_in_pixels : Size of the pixels, corresponding to the size of the sides of the square
        - num_dots           : IMPORTANT : must be a multiple of bs | Number of dot images to present to the neurons. 
        - bs                 : IMPORTANT : num_dots%bs == 0.        | Batch size
        - seed               : random seed 
        - others             : explained in other functions

    Output :

        - dot_stim_dict : dictionnary containing the results of the dot stimulation for every selected neuron. The key are the neurons ids and the values are images where the responses of neurons to the dots where summed 
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
        no_stim_resp= all_neurons_model(torch.ones(1,1,93,93, device=device)* (pixel_min + pixel_max)/2)

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
        
       
        # Convert the list of tensors to a NumPy array then tensor, allow to have only one Tensor in the end
        tensors_array = torch.stack(tensors)
    
        ## For every batch, get the responses of the all_neurons_model
        for i in tqdm(range(0, num_dots, bs)):
            ## Rescale the image to pixel_min and pixel_max
            model_input = rescale(tensors_array[i:i+bs], -1, 1, pixel_min, pixel_max).reshape(bs, 1, 93,93).to(device)
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
        - fitted_data : TODO idkckoi


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
        print(f"Optimized parameters: A = {A_opt}, x0 = {x0_opt}, y0 = {y0_opt}, sigma_x = {sigma_x_opt}, sigma_y = {sigma_y_opt}, rho = {rho_opt}")

        fitted_data = gaussian2D_with_correlation((x_data, y_data), *params)
        fitted_data = fitted_data.reshape(*img_size)

        return A_opt, x0_opt, y0_opt, sigma_x_opt, sigma_y_opt, rho_opt, fitted_data
#%%

def get_preferred_position(
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
    seed = 0
):
    ''' This function performs multiple things :
        1) Uses the function "dot_stimulation" to obtain a dictionnary containing the excitatory pixels for every neuron in "neuron_ids"
        2) Uses the function "gauss_fit" to try to fit the data to a 2D Gaussian model
        3) If it fits : save x0_opt and y0_opt and the error. x0_opt and y0_opt are the preferred position for the stimulus
        4) If the fitting doesn't converges : Save an error = None and the middle point of the image for x0_opt and y0_opt
        5) Return everything as a dataframe

        '''
    
    ## Step 1)
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
    

    ## Step 2

    all_x0 = []
    all_y0 = []
    all_error = []

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

            ## Save x0_opt, y0_opt, error
            all_x0.append(x0_opt)
            all_y0.append(y0_opt)
            all_error.append(error)

        ## Step 4)
        ## If it do not converge, that mean that the receptive field could not be fitted to a gaussian model
        except: 
            all_x0.append(img_res[1]/2)
            all_y0.append(img_res[0]/2)
            all_error.append(None)

    preferred_pos_df = pd.DataFrame({
        "index" : neuron_ids,
        "x_pix" : all_x0,
        "y_pix" : all_y0,
        "error" : all_error},
    )

    return preferred_pos_df



def size_tuning_all_phases(
    model, 
    x_pix, 
    y_pix, 
    preferred_ori,
    preferred_sf, 
    contrast=0.2, 
    radii = np.linspace(0.05, 2.5, 40), 
    phases= np.linspace(0, 2*np.pi, 37)[:-1],
    return_all=False,
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.35,
    img_res = [93,93]
    ):
    ''' 
        This function tries to find the radius size and grating phase of a circular
        grating patch that maximises the response of a single neuron model. 
        Other parameters such as grating orientaion, contrast, and spatial frequency are fixed.
        
        This function is an annotated copy of  "analyses.size_tuning_all_phases".

        Arguments : 

            - model               : A single cell model of the class 'surroundmodulation.models.SingleCellModel'
            - x_pix               : The x coordinates of the center of the circle
            - y_pix               : The y coordinates   ""        ""        ""
            - preferred_ori       : The preferred orientation (the one that elicited the greatest response)
            - preferred_sf        : The preferred spatial frequency
            - contrast            : The value that will multiply the grating image's values (lower than 1 will reduce the contrast and greater than 1 will increase the contrast)
            - radii               : An array containing every radius size to test
            - phases              : An array containing every phase to test
            - return_all          : If set to true, the function will also return the matrix containing every response and every image
            - pixel_min           : Value of the minimal pixel that will serve as the black reference
            - pixel_max           : Value of the maximal pixel that will serve as the white reference (NB : The gray value will be the mean of those two)
            - device              : The device on which to execute the code, if set to "None" it will take the available one
        ?   - size                : The size of the image in terms of visual field degrees
            - img_res             : Resolution of the image in term of pixels shape : [pix_y, pix_x]

        Outputs : 

            - top_radius          : Preferred radius
            - top_phase           : Preferred phase
            - top_grating         : Image corresponding to those parameters 
            - top_resp            : Model's response to this image 

        If return_all == True :

            - resp                : Matrix containing the response for every radius (row) and phase (col)
            - gratings            : Matrix containing the images for every radius (row) and phase (col)

    '''



    ## Select device
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    ## Evaluation mode
    model.eval()

    ## Convert the size to the right shape ( [size_height, size_width] )
    if type(size) is int or type(size) is float: 
        size = [size]*2
    
    ## Matrix containing every image and response for every radius and phase combination (rows = radius, col = phase)
    resp = np.zeros([len(radii), len(phases)])
    gratings = np.zeros([len(radii), len(phases), *img_res])

    ## Try every combination of radius and phase
    with torch.no_grad():
        for i, radius in enumerate(radii):
            for j, phase in enumerate(phases):

                ## Creates a grating background
                a = imagen.SineGrating(orientation= preferred_ori,
                                        frequency= preferred_sf,
                                        phase= phase,
                                        bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                                        offset = 0, # That means that the lowest point will be 0 ?
                                        scale=1 ,
                                        xdensity = img_res[1]/size[1],
                                        ydensity = img_res[0]/size[0])()
                
                ## Creates a uniform gray background
                b = imagen.Constant(scale=0.5,
                                bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                                xdensity = img_res[1]/size[1],
                                ydensity = img_res[0]/size[0])()
                
                ## Creates the circular patch of given radius
                c = imagen.Disk(smoothing=0.0,
                                size=radius*2,
                                scale =1.0,
                                bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                                xdensity = img_res[1]/size[1],
                                ydensity = img_res[0]/size[0], 
                                x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                                y = -get_offset_in_degr(y_pix, img_res[0], size[0]))()    
                
                ## Combine everything to get a grating circular patch with gray background
                d1 = np.multiply(a,c)
                d2 = np.multiply(b,-(c-1.0))

                d = np.add(d1,d2) 
                d = torch.Tensor(d).reshape(1, 1, *img_res).to(device)
                input = rescale(d, 0, 1, -1, 1)*contrast
                input = rescale(d, -1, 1, pixel_min, pixel_max)*contrast

                ## Save the results
                resp[i, j] = model(input)
                gratings[i, j] = input.squeeze().reshape(*img_res).cpu().numpy()

                # TODO improve
                ## Finds the coordinates of the top scoring parameters 
                idx = np.unravel_index(resp.argmax(), [len(radii), len(phases)])

                ## Get the top parameters
                top_grating = gratings[idx]
                top_radius = radii[idx[0]]
                top_phase = phases[idx[1]]
                top_resp = resp[idx]
           
        

    if return_all:
        return top_radius, top_phase, top_grating, top_resp, resp, gratings
    else: 
        return top_radius, top_phase, top_grating, top_resp

def get_offset_in_degr(x_pix, x_size_in_pix, x_size_in_deg):
    ''' This function calculates, along one axis, the distance in degrees between a point and
        the center of the image. 

        Arguments :

            - x_pix         : the pixel position on the x axis
            - x_size_in_pix : the width of the image in term of pixels 
            - x_size_in_deg : the width of the image in term of degrees

        Outputs :

            - x_deg         : the position of the point in degree
    '''
    ## First  : Calculate the difference in pixels from the center and the point
    ##          If the point is on the left : negative values, right : positive values
    ## Second : Make a ratio and convert in degrees to have the shift in degrees
    x_deg = ((x_pix - x_size_in_pix/2) / (x_size_in_pix/2)) * x_size_in_deg/2
  
    return x_deg

def get_orientation_contrast_stim(
    x_pix, 
    y_pix, 
    center_radius, 
    center_orientation,
    spatial_frequency, 
    phase, 
    surround_radius, 
    surround_orientation, 
    gap, 
    contrast, 
    size, 
    num_pix_x,
    num_pix_y, 
):
    '''
    This function takes every parameters needed and returns a stimulus like this :
    A center grating patch with his own orientation.
    A surround ring with his own orientation, but the same phase, spatial frequency as the center
    A gap between those two items.

    Arguments : 

        - x_pix                : The x position that will be the center of our items
        - y_pix                : The y position    ""      ""      ""      ""
        - center_radius        : The radius of the central patch 
        - center_orientation   : The orientation of the center's grating
        - spatial_frequency    : The spatial frequency for both center and surround
        - phase                : The phase for both center and surround
       ?- surround_radius      : The radius of the surround ##IDK HOW THIS WORKS
        - surround_orientation : The orientation of the surround's grating
       ?- gap                  : The size of half of the gap 
        - contrast             : The contrast for both center and surround
       ?- size                 : The (horizontal) size of the image in terms of visual field degrees
        - num_pix_x            : The horizontal size of the image in term of pixels
        - num_pix_y            : The vertical size of the image in term of pixels

    Outputs : 

        - stim  : The image stimulation that is created with those parameters

    TO ME IT SHOULD HAVE GRAY BACKGROUND ??? AND INCLUDE PIXELMIN AND MAX TO BE MORE ADAPATIVE
    why surround AND center have the same phase ??? shouldent we be able to change it ?
    What is radius, and thickness ?
    NB rajouter pixel_min, pixel_max ?
    '''

    ## Convert the size to the right shape ( [size_height, size_width] )
    if type(size) is int or type(size) is float: 
        size = [size]*2

    ## Create the center patch
    center = imagen.SineGrating(mask_shape=imagen.Disk(smoothing=0.0, size=center_radius*2.0),
                                orientation=center_orientation,
                                frequency=spatial_frequency,
                                phase=phase,
                                bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                                offset = -1,
                                scale=2,  
                                xdensity=num_pix_x/size[1],
                                ydensity=num_pix_y/size[0], 
                                x = get_offset_in_degr(x_pix, num_pix_x, size[1]), 
                                y = -get_offset_in_degr(y_pix, num_pix_y, size[0]))()

    ## Total radius. NB : For the annular surround, the radius is from the center to the middle of the thickness
    r = (center_radius + surround_radius + gap)/2.0
    ## Surround thickness
    t = (surround_radius - center_radius - gap)/2.0

    ## Create the surround 
    surround = imagen.SineGrating(mask_shape=imagen.Ring(thickness=t*2.0, smoothing=0.0, size=r*2.0),
                                    orientation=surround_orientation + center_orientation,
                                    frequency=spatial_frequency,
                                    phase=phase,
                                    bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                                    offset = -1,
                                    scale=2,   
                                    xdensity=num_pix_x/size[1],
                                    ydensity=num_pix_y/size[0],
                                    x = get_offset_in_degr(x_pix, num_pix_x, size[1]), 
                                    y = -get_offset_in_degr(y_pix, num_pix_y, size[0]))()

    ## Delimit the inner edge of the gap
    center_disk = (imagen.Disk(smoothing=0.0,
                            size=center_radius*2.0, 
                            bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                            xdensity=num_pix_x/size[1],
                            ydensity=num_pix_y/size[0],
                            x = get_offset_in_degr(x_pix, num_pix_x, size[1]), 
                            y = -get_offset_in_degr(y_pix, num_pix_y, size[0]))()-1)*-1

    ## Delimit the outer edge of the gap
    outer_disk = imagen.Disk(smoothing=0.0,
                            size=(center_radius+gap)*2,
                            bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                            xdensity=num_pix_x/size[1],
                            ydensity=num_pix_y/size[0],
                            x = get_offset_in_degr(x_pix, num_pix_x, size[1]), 
                            y = -get_offset_in_degr(y_pix, num_pix_y, size[0]))()

    ## Delimit the background of the image
    background = imagen.Disk(smoothing=0.0,
                            size= t*2+r*2.0,
                            bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                            xdensity=num_pix_x/size[1],
                            ydensity=num_pix_y/size[0],
                            x = get_offset_in_degr(x_pix, num_pix_x, size[1]), 
                            y = -get_offset_in_degr(y_pix, num_pix_y, size[0]))()

    ## Combine inner and outer parts to create the gap ring
    ring = center_disk*outer_disk
    ## Add every compound together why np.maximum ?
#Modified:
    #NB gray center bc -1 + 1 = 0
    stim = np.add(np.maximum(center, surround), ring)*contrast 
    stim = np.multiply(stim,background)
    plot_img(stim, -1, 1)
    return stim


def orientation_contrast(
    model, 
    x_pix, 
    y_pix, 
    preferred_ori,
    preferred_sf, 
    center_radius,
    orientation_diffs, 
    phases,
    gap = 0.2,
    contrast=0.2,
    img_res = [93,93], 
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.35
    ):

    ''' This function takes a Single Neuron Model and returns an array of the stimulations
        and the model's response for every combination of phase and surround orientation.
        This function uses the "get_orientation_contrast_stim" function in order to create the stimuli.

        Arguments :
            - model             : A single cell model of the class 'surroundmodulation.models.SingleCellModel'
            - x_pix             : The x position that will be the center of our items
            - y_pix             : The y position    ""      ""      ""      ""
            - preferred_ori     : The preferred orientation of the center (the one that elicited the greatest response)
            - preferred_sf      : The preferred spatial frequency of the center (This will also be the spatial frequency of the surround)
            - center_radius     : The radius of the center
           ?- orientation_diffs : An array containing every (surround orientation/ or surround diff of orientation) to use
            - phases            : An array containing every phase to use
           ?- gap               : The size of half of the gap 
            - contrast          : The value that will multiply the image's values (center and surround)
            - img_res           : The resolution of the image in term of pixels, shape : [pix_y, pix_x]
            - pixel_min         : Value of the minimal pixel that will serve as the black reference
            - pixel_max         : Value of the maximal pixel that will serve as the white reference (NB : The gray value will be the mean of those two)
            - device            : The device on which to execute the code, if set to "None" it will take the available one
        ?   - size              : The size of the image in terms of visual field degrees
    
        Outputs :

            - oc_stims : A matrix containing the images obtained with every combination of surround orientation (rows) and phase (col) 
            - oc_resps : A matrix containing the responses obtained for every image

            Why try every phase ?
        This funtion takes the preferred center's parameters (except phase)
        and it uses the function "get_orientation_contrast_stim" to get the stimulations
        for every phase and every surround orientation. 
        return every stimulations and every response
    '''

    ## Select device
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    ## Evaluation mode
    model.eval()

    ## Convert the size to the right shape ( [size_height, size_width] )
    if type(size) is int or type(size) is float: 
        size = [size]*2

    oc_stims = []
    oc_resps = []

    with torch.no_grad():
        ## For every surround orientation
        for od in orientation_diffs:
            ## List of stimuli for every phase
            oc_stim = [get_orientation_contrast_stim(
                    x_pix=x_pix, 
                    y_pix=y_pix, 
                    center_radius = center_radius, 
                    center_orientation=preferred_ori,
                    spatial_frequency = preferred_sf, 
                    phase = phase, 
                    surround_radius = 100,       #Why 100
                    surround_orientation = od, 
                    gap = gap,  
                    contrast = contrast, 
                    size = size, 
                    num_pix_x = img_res[1],
                    num_pix_y = img_res[0],
                ) for phase in phases]
            
            ## Reshape the stimuli and get their response
            oc_stim = np.stack(oc_stim).reshape(-1, 1, *img_res) #np.stack convert a list of array into a simple array
            oc_stim = torch.Tensor(oc_stim).to(device)
            oc_stim = rescale(oc_stim, -1,1, pixel_min, pixel_max)
            oc_resp = model(oc_stim)

            ## Save the stimuli and responses
            oc_stims.append(oc_stim.detach().cpu().numpy().squeeze())
            oc_resps.append(oc_resp.detach().cpu().numpy().squeeze())
        
        ## Convert the list of arrays into arrays
        oc_stims = np.stack(oc_stims)
        print(oc_stims.shape)
        oc_resps = np.stack(oc_resps)

    return oc_stims, oc_resps


################################
##### Analysis 1 functions #####
################################

#%%
def get_size_tuning_curves(
    model,
    x_pix,
    y_pix,
    preferred_ori,
    preferred_sf,
    preferred_phase,
    radii = np.logspace(-2,np.log10(2),20) ,
    contrast = 1,
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.35,
    img_res = [93,93],
    neg_val = True
    ):
    
    ''' This function is a size tuning function that makes the size of circular and annular grating images
        vary with their prefered parameters fixed. The difference between the response of the model for each image and the response for a gray image is calculated
        ans saved into an array of the same size of the argument "radii". You can afterwards plot these arrays
        to visualise the effect of the radii on the model's response. If neg_val is set to 'False', the negative values
        are set to 0.


        
        NB : We substract the response to gray image to the response of stimuli because it is necessary for the rest of our analysis, it normalises the values.

        Arguments : 

            - model           : A single cell model of the class 'surroundmodulation.models.SingleCellModel'
            - x_pix           : The x coordinates of the center of the circle, (value in pixels, can be a float)
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

        Outputs : 

            - circular_tuning_curve  : An array containing the responses for every center stimulus minus the response to gray image
            - annular_tuning_curve   : An array containing the responses for every surround stimulus minus the response to gray image
    '''

    ## Select device
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    ## Evaluation mode
    model.eval()

    ## Convert the size to the right shape ( [size_height, size_width] )
    if type(size) is int or type(size) is float: 
        size = [size]*2

    ## To save the responses
    circular_tuning_curve   = torch.zeros(len(radii))
    annular_tuning_curve = torch.zeros(len(radii))


    with torch.no_grad():

        ## Create Gray stimulus to substract to the non gray stimuli
        gray_stim = torch.ones((1, *img_res)).to(device) * ((pixel_min + pixel_max)/2)
        gray_resp = model(gray_stim)

        for i, radius in enumerate(radii) :
            
            ## Make circular stimulus
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

            ## Create the outer edge of the ring
            outer_edge = torch.Tensor(imagen.Disk(
                                smoothing=0.0, 
                                size=(min(size[0],size[1])), #Extend it to the border
                                bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))), 
                                xdensity=img_res[1]/size[1],
                                ydensity=img_res[0]/size[0], 
                                x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                                y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())
            grating_ring = grating_background * outer_edge
            

            ## Create the inner edge of the ring
            inner_edge = (torch.Tensor(imagen.Disk(
                                smoothing=0.0, 
                                size=(radius)*2.0,
                                bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))), 
                                xdensity=img_res[1]/size[1],
                                ydensity=img_res[0]/size[0], 
                                x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                                y = -get_offset_in_degr(y_pix, img_res[0], size[0]))()) * -1) + 1
            grating_ring = grating_ring * inner_edge

            ## Change the contrast 
            grating_center *= contrast
            grating_ring   *= contrast
            
            ## Convert to the right shape for the model
            grating_center = rescale(grating_center,-1,1,pixel_min,pixel_max).reshape(1,*img_res).to(device)
            grating_ring = rescale(grating_ring,-1,1,pixel_min,pixel_max).reshape(1,*img_res).to(device)


            ## Save responses for this radius, substract the gray response
            if neg_val == False :
                ## Avoid negative values
                circular_tuning_curve[i] = torch.maximum(model(grating_center) - gray_resp, torch.tensor(0))
                annular_tuning_curve[i]  = torch.maximum(model(grating_ring) - gray_resp,torch.tensor(0))   

            else : 
                ## Allow negative values
                circular_tuning_curve[i] = model(grating_center) - gray_resp
                annular_tuning_curve[i]  = model(grating_ring) - gray_resp

    return circular_tuning_curve, annular_tuning_curve

def get_GSF_surround_AMRF(
    radii,
    circular_tuning_curve,
    annular_tuning_curve
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
            

        Arguments : 

            - radii                  : The array that contains the radii that were used to obtain circular_tuning_curve and annular_tuning_curve
            - circular_tuning_curve    : An array, the center stimulus size tuning curve
            - annular_tuning_curve  : An array, the surround stimulus size tuning curve

        Outputs : 

            - GSF             : Optimal circular radius that elicits approximately the maximal response
            - surround_extent : Optimal circular radius that elicits approximately the asymptotic response (suppression)
            - AMRF            : Optimal annular radius that elicits approximately the lowest response
            - Ropt            : The response of the model at the GSF for the circular stimuli
            - Rsupp           : The asymptotic response of the model for the circular stimuli

            NB : outputs are radii, not diameters
    '''
   

    GSF             = 0
    surround_extent = 0
    AMRF            = 0
    Ropt            = 0
    Rsupp           = circular_tuning_curve[-1] #The last response = asymptotic value

    ## Select the thresholds 
    GSF_thresh   = (95 * torch.max(circular_tuning_curve)) / 100
    surround_thresh_min = Rsupp - ((5 * Rsupp) /100)
    surround_thresh_max = Rsupp + ((5 * Rsupp) /100)
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
            if resp >= surround_thresh_min and resp <= surround_thresh_max:
                surround_extent = radii[i]
                break

    ## For the AMRF
    for i, resp in enumerate(annular_tuning_curve):
        if resp < AMRF_thresh:
            AMRF = radii[i]
            break
    
    ## Computes the suppression index
    #TODO check if 
    
    SI = ((Ropt - Rsupp) / Ropt).item()
    
    return GSF, surround_extent, AMRF, SI, Ropt, Rsupp

def plot_size_tuning_curve(
    all_neurons_model,
    neuron_id,
    pref_params_df,
    radii = np.logspace(-2,np.log10(2),20),
    contrast = 1,
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.35,
    img_res = [93,93],
    neg_val = True,
    use_full_field_pref_phase = False    

):
    ''' Plot the tuning curve of the neuron corresponding to the index given as an argument.
        The tuning curves are obtained with the "get_size_tuning_curves" function.
        
        NB  : The plot shows diameters, not radii
        NB2 : This function creates a model using the a single cell adaptation of the v1_convnext_ensemble model
    
        Arguments :

            - use_full_field_pref_phase : If set to true, use the 'phase_full_field' column of the parameters instead of the phase computed computed while doing size tuning
            - others : see 'get_size_tuning_curves'
    '''
    ## Get the model for the neuron
    model = SingleCellModel(all_neurons_model, neuron_id )

    ## Get the neuron's preferred parameters 
    max_ori = pref_params_df["orientation"].values[neuron_id]
    max_sf  = pref_params_df["sf"].values[neuron_id]
    x_pix   = pref_params_df["x_pix"].values[neuron_id]
    y_pix   = pref_params_df["y_pix"].values[neuron_id]

    ## Decide if we want to use the preferred phase found full field or not
    if use_full_field_pref_phase : 
        max_phase = pref_params_df["phase_full_field"].values[neuron_id]
    else : 
        max_phase = pref_params_df["phase_GSF"].values[neuron_id]

    ## Perform size tuning to get the curves
    circular_tuning_curve, annular_tuning_curve = get_size_tuning_curves(
            model = model,
            x_pix = x_pix,
            y_pix = y_pix,
            preferred_ori = max_ori,
            preferred_sf = max_sf,
            preferred_phase = max_phase,
            radii = radii,
            contrast = contrast,
            pixel_min = pixel_min,
            pixel_max = pixel_max,
            device = device,
            size = size,
            img_res = img_res,
            neg_val = neg_val
            )
    
    ## Multiply the radii to get the diameters
    plt.plot(radii*2,circular_tuning_curve, label = "circle")
    plt.plot(radii*2, annular_tuning_curve, label = "ring")
    plt.xlabel("diameter (deg)")
    plt.ylabel('response')
    plt.legend()
    plt.title(f"Neuron {neuron_id}")
    plt.show()


    

def analysis_1_as_dataframe(
    all_neurons_model,
    neuron_ids,   
    pref_params_df, 
    radii = np.logspace(-2,np.log10(2),20),
    contrast = 1,
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.35,
    img_res = [93,93],
    use_full_field_pref_phase = False    
    ):

    ''' This function performs size tuning eperiment to data where the preferred phase is known
        (that means that the 'analysis_1_as_dataframe' function has already computed the size tuning and saved the preferred phase)
        (this function is mainly used to reproduce the experiments while using pre-computed parameters)  )

        This function uses a list of neuron index, and a dataframe containing the preferred parameters 
        of thoses neurons and perform the following analyses for each neuron :

            Step 1) Computes the circular_tuning_curve and the annular_tuning_curve with the "get_size_tuning_curves" function

            Step 2) Computes the GSF, surround_extent, AMRF, SI, Ropt, Rsupp with the "get_GSF_surround_AMRF" function

            Step 3) Puts all of those results into a dataframe :

                                 |  index  |   GSF   | surround_extant |   AMRF   |    SI    | 
                                 _____________________________________________________________
                        Neuron 0 |    0    |         |                 |          |          |
                        Neuron 1 |    1    |         |                 |          |          |
                        Neuron 2 |    2    |         |                 |          |          |
                          . . .  |    .    |         |                 |          |          |
                        Neuron n |    n    |         |                 |          |          | 

        NB  : The Value of GSF, surround_extant and AMRF are radii size (in deg), not diameters
        NB2 : This function assumes that pref_params_df is sorted correctly (row 0 = neuron 0, ... row n = neuron n)
        
        Arguments :

            - pref_params_df            : Dataframe containing the columns 'orientation', 'sf', 'phase_GSF', and/or 'phase_full_field' 
            - use_full_field_pref_phase : If set to true, use the 'phase_full_field' column of the parameters instead of the phase computed computed while doing size tuning
            - radii                     : An array containing every radius to test
            - others                    : See in other functions
    '''
    
    all_GSF      = []
    all_surr_ext = []
    all_AMRF     = []
    all_SI       = []

    for neuron_id in tqdm(neuron_ids):
        ## Get the model for the neuron
        model = SingleCellModel(all_neurons_model,neuron_id )

        ## Get the neuron's preferred parameters 
        max_ori = pref_params_df["orientation"].values[neuron_id]
        max_sf  = pref_params_df["sf"].values[neuron_id]
        x_pix   = pref_params_df["x_pix"].values[neuron_id]
        y_pix   = pref_params_df["y_pix"].values[neuron_id]
        
        ## Chose the phases
        if use_full_field_pref_phase : 
            max_phase = pref_params_df["phase_full_field"].values[neuron_id]
        else :
            max_phase = pref_params_df["phase_GSF"].values[neuron_id]

        ## Step 1
        circular_tuning_curve, annular_tuning_curve = get_size_tuning_curves(
            model = model,
            x_pix = x_pix,
            y_pix = y_pix,
            preferred_ori = max_ori,
            preferred_sf = max_sf,
            preferred_phase = max_phase,
            radii = radii,
            contrast = contrast,
            pixel_min = pixel_min,
            pixel_max = pixel_max,
            device = device,
            size = size,
            img_res = img_res
            )
        
        ## Step 2
        GSF, surround_extent, AMRF, SI, Ropt, Rsupp = get_GSF_surround_AMRF(
            radii = radii,
            circular_tuning_curve = circular_tuning_curve,
            annular_tuning_curve = annular_tuning_curve
            )
        
        ## Step 3
        all_GSF.append(GSF)
        all_surr_ext.append(surround_extent)
        all_AMRF.append(AMRF)
        all_SI.append(SI)

    
    df = pd.DataFrame({
        "index"           : neuron_ids,
        "GSF"             : all_GSF,
        "surround_extent" : all_surr_ext,
        "AMRF"            : all_AMRF,
        "SI"              : all_SI},
        index = neuron_ids)
    
    return df

def analysis_1_as_dataframe_all_phases(
    all_neurons_model,
    neuron_ids,   
    pref_params_df, 
    radii = np.logspace(-2,np.log10(2),20),
    phases = np.linspace(0, 2*np.pi, 37)[:-1],
    contrast = 1,
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.35,
    img_res = [93,93]
    ):
    ''' This function is the same as 'analysis_1_as_dataframe' but the analysis is done for every phase instead
        of a fixed phase.
        It returns the values from which the GSF elicited the greatest response

        NB : This function assumes that pref_params_df is sorted correctly (row 0 = neuron 0, ... row n = neuron n)
    '''
    all_phases   = []
    all_GSF      = []
    all_surr_ext = []
    all_AMRF     = []
    all_SI       = []

    ## For every
    for neuron_id in tqdm(neuron_ids):
        ## Get the model for the neuron
        model = SingleCellModel(all_neurons_model,neuron_id )

        ## Get the neuron's preferred parameters 
        max_ori = pref_params_df["orientation"].values[neuron_id]
        max_sf  = pref_params_df["sf"].values[neuron_id]
        max_x   = pref_params_df["x_pix"].values[neuron_id]
        max_y   = pref_params_df["y_pix"].values[neuron_id]

        ## Initiate the parameters to find
        GSF_preferred_phase = 0
        surround_extent_preferred_phase = 0
        AMRF_preferred_phase = 0
        SI_preferred_phase = 0

        Rmax = 0 

        for phase in phases :
            ## Get the tuning curve for the current phase
            circular_tuning_curve, annular_tuning_curve = get_size_tuning_curves(
                model = model,
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
                img_res = img_res
                )
            
            ## Perform some analysis
            GSF, surround_extent, AMRF, SI, Ropt, _ = get_GSF_surround_AMRF(
                radii = radii,
                circular_tuning_curve = circular_tuning_curve,
                annular_tuning_curve = annular_tuning_curve
                )
            
            if Ropt>Rmax :

                preferred_phase = phase
                GSF_preferred_phase = GSF
                surround_extent_preferred_phase = surround_extent
                AMRF_preferred_phase = AMRF
                SI_preferred_phase = SI
                Rmax = Ropt
        
        all_phases.append(preferred_phase)
        all_GSF.append(GSF_preferred_phase)
        all_surr_ext.append(surround_extent_preferred_phase)
        all_AMRF.append(AMRF_preferred_phase)
        all_SI.append(SI_preferred_phase)
        
        
    
    df = pd.DataFrame({
        "index"           : neuron_ids,
        "GSF"             : all_GSF,
        "surround_extent" : all_surr_ext,
        "AMRF"            : all_AMRF,
        "SI"              : all_SI,
        "phase_GSF"  : all_phases},
        index = neuron_ids)
    
    #TODO delete print :
    print("VOICI LA VALEUR DU PREMIER AMRF (devrait etre 0.674)")
    print("Si c'est 0.77 je comprends pas")
    print(all_AMRF[0])
    
    return df

def exclude_low_supp_neurons(
    analysis_1_df,
    supp_thresh = 0.1
    ):

    ''' The goal of this function is to select only the neurons with a surround suppression.
        This function takes a dataframe with a "SI" column corresponding to the Suppression Index of the neurons, it then removes the rows where
        SI < supp_thresh (the neurons with very little surround suppression)

        Arguments :

            - analysis_1_df    : Output of the function 'analysis_1_as_dataframe'
            - supp_thresh      : Threshold, neurons with a suppression index below that value will be excluded

        Outputs :

            - analysis_1_df_filtered : A dataframe with only the neurons with a SI > supp_thresh

    '''

    condition = analysis_1_df["SI"] >= supp_thresh
    analysis_1_df_filtered = analysis_1_df[condition]


    return analysis_1_df_filtered


def remove_bad_neurons(
        analysis_1_df
    ):
    ''' This function removes unwanted neurons from our data.
        The unwanted neurons are the same as in the article :
        neurons whith no suppression nor response saturation.
        We basically remove the neurons with a negative Suppression Index,
        We also remove neurons with no surround extent (surround_extent == 0)
    '''
    ## Remove negative SI
    analysis_1_df_filtered = exclude_low_supp_neurons(analysis_1_df,supp_thresh = 0)

    ## Remove cells with surround_extent == 0 
    analysis_1_df_filtered = analysis_1_df_filtered[analysis_1_df_filtered["surround_extent"] > 0.001]

    return analysis_1_df_filtered

def get_contrast_response(
    model,
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
    size = 2.35,
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
    model.to(device)

    ## Evaluation mode
    model.eval()

    ## Convert the size to the right shape ( [size_height, size_width] )
    if type(size) is int or type(size) is float: 
        size = [size]*2
    
    ## Avoid surround and center to overlap
    if surround_radius < center_radius :
        surround_radius = center_radius

    

    contrast_resps_mat = torch.zeros((len(surround_contrasts), len(center_contrasts))).to(device)

    with torch.no_grad():
        ## Get the response of the model to gray stimulus
        gray_stim = torch.ones((1,93,93)).cuda() * ((pixel_min + pixel_max)/2)
        gray_resp = model(gray_stim)

        ## For every combination of contrasts
        for i, sc in enumerate(surround_contrasts):
            for j, cc in enumerate(center_contrasts) : 

                ## Make images 
                ## Make Center stimulus
                grating_center = torch.Tensor(imagen.SineGrating(mask_shape=imagen.Disk(smoothing=0.0, size=center_radius*2.0),
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
                                    size=center_radius*2.0, 
                                    bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                                    xdensity=img_res[1]/size[1],
                                    ydensity=img_res[0]/size[0], 
                                    x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                                    y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())
                
                ## Add the gray background to the circular stimulus
                grating_center = grating_center * center_mask


                ## Make the surround stimulus
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

                ## Create the outer edge of the ring
                outer_edge = torch.Tensor(imagen.Disk(
                                    smoothing=0.0, 
                                    size=(min(size[0],size[1])), #Extend it to the border
                                    bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))), 
                                    xdensity=img_res[1]/size[1],
                                    ydensity=img_res[0]/size[0], 
                                    x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                                    y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())
                grating_ring = grating_background * outer_edge
                

                ## Create the inner edge of the ring
                inner_edge = (torch.Tensor(imagen.Disk(
                                    smoothing=0.0, 
                                    size=(surround_radius)*2.0,
                                    bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))), 
                                    xdensity=img_res[1]/size[1],
                                    ydensity=img_res[0]/size[0], 
                                    x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                                    y = -get_offset_in_degr(y_pix, img_res[0], size[0]))()) * -1) + 1
                grating_ring = grating_ring * inner_edge


                ## Change the contrast 
                grating_center *= cc
                grating_ring   *= sc

                ## Merge the center and the surround
                image = grating_center + grating_ring
  
                ## Convert to the right shape for the model
                image = rescale(image,-1,1,pixel_min,pixel_max).reshape(1,*img_res).to(device)

                ## Get the model's response 
                resp = model(image)

                ## Normalise the response
                resp_norm = resp - gray_resp

                ## Save the response
                contrast_resps_mat[i,j] = resp_norm

    ## Optional : replace negative values by zero
    if neg_val == False :     
        contrast_resps_mat = torch.maximum(contrast_resps_mat, torch.zeros(contrast_resps_mat.shape).to(device))

    return contrast_resps_mat

def plot_contrast_response(
    neuron_id,
    all_neurons_model,
    analysis_1_df,
    pref_params_df,
    center_contrasts = np.logspace(-2,np.log10(1),18),
    surround_contrasts = np.logspace(-2,np.log10(1),18),
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.35,
    img_res = [93,93],
    neg_val = True
    ):

    ''' This function uses the function 'get_contrast_response' to compute the response matrix 
        and then plots the different curves of response for each contrast combination.

        NB : yaxis  = Response of the single neuron modele 'model', 
             xaxis  = Contrast of the center
             curves = One curve for each surround contrast

        Arguments :

            -neg_val : if set to False, then put every negative value to zero
    '''

    ## Get a single neuron model
    model = SingleCellModel(all_neurons_model,neuron_id )

    ## Select device
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    ## Get the preferred parameters of the neuron
    GSF  = analysis_1_df[analysis_1_df["index"] == neuron_id]["GSF"].values[0]
    AMRF = analysis_1_df[analysis_1_df["index"] == neuron_id]["AMRF"].values[0]
    preferred_ori = pref_params_df[pref_params_df["index"] == neuron_id]["orientation"].values[0]
    preferred_sf = pref_params_df[pref_params_df["index"] == neuron_id]["sf"].values[0]
    preferred_phase = pref_params_df[pref_params_df["index"] == neuron_id]["phase_GSF"].values[0]
    if GSF>AMRF :
        AMRF = GSF
    x_pix = pref_params_df[pref_params_df["index"] == neuron_id]["x_pix"].values[0]
    y_pix = pref_params_df[pref_params_df["index"] == neuron_id]["y_pix"].values[0]

    ## Better to take those values
    center_radius = GSF
    surround_radius = AMRF

    ## Get the contrast tuning curves (matrix)
    contrast_tuning_mat = get_contrast_response(
        model = model,
        x_pix = x_pix,
        y_pix = y_pix,
        center_radius = center_radius,
        surround_radius = surround_radius,
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
    
    print(torch.min(contrast_tuning_mat))
    print(torch.mean(torch.std(contrast_tuning_mat, axis = 0)).cpu())
    
    ## Put the matrix to the cpu for plotting
    contrast_tuning_mat = contrast_tuning_mat.cpu()


    ## plot each curve
    for i in range(len(contrast_tuning_mat)) :
        plt.plot(center_contrasts,contrast_tuning_mat[i], label = f"{round(surround_contrasts[i],5)}")


    plt.legend()
    plt.title(f"Response contrast of the neuron {neuron_id}")
    plt.xlabel("Center contrast")
    plt.ylabel("Response")
    plt.xscale('log')
    plt.show()



def analysis_2_as_dataframe(
    all_neurons_model,
    analysis_1_df,
    pref_params_df,
    center_contrasts = np.logspace(-2,np.log10(1),18),
    surround_contrasts = np.logspace(-2,np.log10(1),18),
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.35,
    img_res = [93,93],
    neg_val = True
    ):
    ''' This function takes a Dataframe outputed from the 'analysis_1_as_dataframe' function and perform for every line (neuron) :

        1) Perform 'get_contrast_response' function to get the response of each neuron to different center and surround contrasts.

        2) Perform one analyse for each results : get the last point of every curves and calculate the difference between the highest and
           the lowest value, also computes the mean variance accross all points
        
        3) Add a column to the dataframe given in argument

        NB : This function uses the SingleNeuronModel class and the v1_convnext_ensemble

    Arguments :

        -analysis_1_df : The output of the 'analysis_1_as_dataframe' containing the indices of the neurons to test and some results of former experiments (such as GSF, AMRF...)
        -pref_params_df : The Dataframe containing the preferred orientation, spatial frequency qnd the phase of every neuron
        TODO finish args and outputs
        TODO implement use fullfieldphase
    '''
        
    ## Extract every neuron id
    neuron_ids = analysis_1_df["index"].values

    all_std           = []
    all_maxmin_ratios = []

    for neuron_id in tqdm(neuron_ids) :
        
        ## Get the model of the neuron
        model = SingleCellModel(all_neurons_model, neuron_id)

        ## Get the preferred parameters of the neuron
        GSF  = analysis_1_df[analysis_1_df["index"] == neuron_id]["GSF"].values[0]
        AMRF = analysis_1_df[analysis_1_df["index"] == neuron_id]["AMRF"].values[0]
        preferred_ori = pref_params_df[pref_params_df["index"] == neuron_id]["orientation"].values[0]
        preferred_sf = pref_params_df[pref_params_df["index"] == neuron_id]["sf"].values[0]
        preferred_phase = pref_params_df[pref_params_df["index"] == neuron_id]["phase_GSF"].values[0]
        if GSF>AMRF :
            AMRF = GSF

        x_pix = pref_params_df[pref_params_df["index"] == neuron_id]["x_pix"].values[0]
        y_pix = pref_params_df[pref_params_df["index"] == neuron_id]["y_pix"].values[0]

        ## Get the contrast tuning curves (matrix)
        contrast_tuning_mat = get_contrast_response(
            model = model,
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


        last_points = contrast_tuning_mat[:,-1]
        min = torch.min(last_points)
        max = torch.max(last_points)
        ratio = (max/min).cpu().item()

        
        mean_std = torch.mean(torch.std(contrast_tuning_mat, axis = 0)).cpu().item()
        
        all_std.append(mean_std)
        all_maxmin_ratios.append(ratio)

    ## Create the dataframe
    analysis_2_df = analysis_1_df.copy()
    analysis_2_df["mean_std"] = all_std
    analysis_2_df["max/min"] = all_maxmin_ratios

    return analysis_2_df

def sort_by_spread(
    analysis_2_df,
    sort_by_std = False
):
    ''' This function sorts the neurons from the ones with less spread curves during contrast tuning
        to the more spread ones. 
        Spread is assessed thanks to the mean_std or maxmin_ratio values.

        Arguments :

            - analysis_2_df : A dataframe obtained thanks to the 'analysis_2_as_dataframe' function (optionnally filtered). 
            - sort_by_std   : If set to true, sort by the mean_std column, if set to false, sort by max/min column.
    '''

    analysis_2_df_sorted = analysis_2_df.copy()


    if sort_by_std :
        ## Sort accross mean_std
        all_std = analysis_2_df_sorted["mean_std"].values
        order = np.argsort(all_std)

    else :
        ## Sort accross min/max
        all_maxmin = analysis_2_df_sorted["max/min"].values
        order = np.argsort(all_maxmin)

    ## Order the rows
    analysis_2_df_sorted = analysis_2_df_sorted.reindex(order, axis = 0)

    return analysis_2_df_sorted

def get_contrast_size_tuning_curve_all_phase(
    model,
    x_pix,
    y_pix,
    preferred_ori,
    preferred_sf,
    phases = np.linspace(0, 2*np.pi, 37)[:-1],
    contrasts = np.linspace(0,1,6),
    radii = np.logspace(-2,np.log10(2),20) ,
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.35,
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

    #TODO : check which device is circular_curve on
    all_circular_curves = torch.zeros((len(phases),len(contrasts), len(radii)))
    all_annular_curves  = torch.zeros((len(phases),len(contrasts), len(radii)))

    
    ## For every phase :
    for num_phase in range(len(phases)) :
        
        ## For every contrast (every row of the matrices)
        for i,contrast in enumerate(contrasts) : 
            
            ## Get the size tuning curve
            circular_curve, annular_curve = get_size_tuning_curves(
                model = model,
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
                neg_val = neg_val
                )


            ## DELETE AFTER, only here to visualise if there is a shift
            ## For low contrast
            if i == 2 :
                GSF_low,_,_,_,_,_ = get_GSF_surround_AMRF(radii = radii,circular_tuning_curve =circular_curve,annular_tuning_curve = annular_curve)
            ## For contrast =1
            if i == 5 : 
                GSF_high,_,_,_,_,_ = get_GSF_surround_AMRF(radii = radii,circular_tuning_curve =circular_curve,annular_tuning_curve = annular_curve)

            ## Fill the corresponding rows of the matrices
            all_circular_curves[num_phase ,i ,:] = circular_curve
            all_annular_curves[num_phase ,i ,:]  = annular_curve

    ##TODO DELETE
    shift_GSF = GSF_high - GSF_low

    all_circular_curves = torch.max(all_circular_curves, dim=0).values
    all_annular_curves = torch.max(all_annular_curves, dim=0).values
    

    
    return all_circular_curves, all_annular_curves, shift_GSF



def plot_contrast_size_tuning_curve_all_phase(
    all_neurons_model,
    neuron_id,
    pref_params_df,
    phases = np.linspace(0, 2*np.pi, 37)[:-1],
    contrasts = np.logspace(-2,np.log10(1),6),
    radii = np.logspace(-2,np.log10(2),20) ,
    pixel_min = -1.7876, #e.g:  (the lowest value encountered in the data we worked with)
    pixel_max =  2.1919, #e.g:  (the highest value encountered in the data we worked with)
    device = None,
    size = 2.35,
    img_res = [93,93],
    neg_val = True,
    plot_annular = False
    ):

    ''' Plot the size tuning curves of the neuron for multiple contrasts values.
        The neuron is given as an index
        The tuning curves are obtained with the "get_contrast_size_tuning_curves" function.
        
        NB  : The plot shows diameters, not radii
        NB2 : This function creates a model using the a single cell adaptation of the v1_convnext_ensemble model

        Arguments :

            - plot_annular : if set to 'True', also plot the contrast size tuning curves of the annular stimuli
            - others       : See 'get_contrast_size_tuning_curve'
    '''
    
    ## Create the model for the neuron
    model = SingleCellModel(all_neurons_model,neuron_id)


    ## Get the preferred parameters of the neuron
    preferred_ori = pref_params_df[pref_params_df["index"] == neuron_id]["orientation"].values[0]
    preferred_sf = pref_params_df[pref_params_df["index"] == neuron_id]["sf"].values[0]
    x_pix = pref_params_df[pref_params_df["index"] == neuron_id]["x_pix"].values[0]
    y_pix = pref_params_df[pref_params_df["index"] == neuron_id]["y_pix"].values[0]

    ## Get the size tuning curves for different contrasts
    all_circular_curves, all_annular_curves, shift_GSF = get_contrast_size_tuning_curve_all_phase( #TODO remove shift_GSF
        model=model,
        x_pix=x_pix,
        y_pix=y_pix,
        preferred_ori=preferred_ori,
        preferred_sf=preferred_sf,
        phases=phases,
        contrasts = contrasts,
        radii = radii,
        pixel_min = pixel_min,
        pixel_max = pixel_max,
        device = device,
        size = size,
        img_res = img_res,
        neg_val = neg_val
        )
    

    for i, circular_tuning_curve in enumerate(all_circular_curves) :
        plt.plot(radii*2,circular_tuning_curve, label = f"{round(contrasts[i],5)}" )
        plt.xlabel("diameter (deg)")
        plt.ylabel('response')
        plt.legend()
        plt.title(f"Neuron {neuron_id}, circular stimulus. Shift :{shift_GSF}") #TODO remove shift_GSF
    plt.show()

    if plot_annular :
        for i, annular_tuning_curve in enumerate(all_annular_curves) :
            plt.plot(radii*2,annular_tuning_curve, label = f"{round(contrasts[i],5)}" )
            plt.xlabel("diameter (deg)")
            plt.ylabel('response')
            plt.legend()
            plt.title(f"Neuron {neuron_id}, annular stimulus")
        plt.show()

def find_center_phase(pref_params_df,
    phases = np.linspace(0, 2*np.pi, 37)[:-1],
    radii = np.logspace(-2,np.log10(2),40)):

    ''' This function will do a size tuning function on every neuron contained on the dataframe "pref_params_df".
        It will make the phase vary for every

        #TODO maybe remove this function
    '''
    