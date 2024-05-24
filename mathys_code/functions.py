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
from surroundmodulation.utils.plot_utils import plot_img
from scipy.optimize import curve_fit
## Import models 
from nnvision.models.trained_models.v1_task_fine_tuned import v1_convnext_ensemble #TODO remove this line since useful only in main
from surroundmodulation.models import SingleCellModel
## Data storage
import h5py


####################################################################################
#####    THIS CODE CONTAINS EVERY FUNCTION USED TO PERFORM THE EXPERIMENTS.    #####
#####           ----------------------------------------------------           #####
##### Those function works with a HDF5 file where they all store and get data  #####
####################################################################################

####################
## PREREQUISITE : ##
####################

'''
1)  It is necessary to have a model that works as so :
    it's imput is an image -> it outputs an array of neuron activities

2)

'''

'''
Things to know about the functions and the HDF5 file :

1) If the file doesn't exist, it creates it. Same for every group of the file

2) If some group exists in the file, you have the choice of overwriting them or appending in them 

3) If you chose to add some data to already existing data, the functions will check 
   if the arguments are the same in both cases.

4) If the function requires the execution of other functions it will :

    - check if those analyses were performed
    - check if the arguments are matching between all of those analyses
    - check if the requested neurons are present in all of thoses analyses

'''

#############################################################
##### PART 0 : Functions that help manage the HDF5 file #####
#############################################################


def clear_group(h5_file, group_path):
    ''' This function removes the specified group in the HDF5 file and every subgroup

    Arguments :

        - h5_file      : Path to the HDF5 file
        - group_path   : Path to the group in the file
    '''

    try :
        ## Open the file in r+ so it is possible to remove specific parts
        with h5py.File(h5_file, "r+") as file:

            ## Check if the group exists
            if group_path in file:

                del file[group_path]

                print(f"Group '{group_path}' cleared successfully.")

            else:
                print(f"Group '{group_path}' did not exist in the HDF5 file.")
    
    ## Just in case the file doesn't exist yet
    except FileNotFoundError :
        pass


def check_group_exists(h5_file, group_path):
    ''' This function returns True if the group exists and False if it does not
    '''
    
    try :
        with h5py.File(h5_file, 'r') as file:
            # Try to access the group, if it doesn't exist, it will raise a KeyError
            _ = file[group_path]

            return True
        
    except KeyError:

        return False
        
def assert_group_exists(h5_file, group_path):
    #TODO remove this function
    ''' This function checks if the group exist, and if it doesn't, it creates it
    '''
    with h5py.File(h5_file, 'r') as file:

        try:
            # Try to access the group, if it doesn't exist, it will raise a KeyError
            _ = file[group_path]
            
        except KeyError:
            file.create_group(group_path)


def get_arguments_from_str(
    args_str
):
    
    ''' This function takes a formated string and creates a dictionnary of the arguments values from it.    

        Arguments :

            - args_str : string of the following format : "argument1=argument1_value/argument2=argument2_value/..."
        TODO ENLEVER CES PTN DE \n
    '''
    ## Remove the unwanted '\n' 
    args = args_str.replace("\n", "")

    ## Separate every parameter
    split = args.split("/")

    ## Create the dictionary  
    dico_args = {}

    ## For every splitted part, extract the name of the parameter and its value (keep everything in string type)
    for part in split :
        name  = part.split('=')[0]
        value = part.split('=')[1]
        dico_args[name] = value

    return dico_args

def get_arguments_from_attributes(
    h5_file,
    group_path
):
    
    ''' This functions goes in the attributes of a group and searches for the "arguments" attribute.
        It then extracts the values of every parameters and makes it a dictionnary
    '''

    ## Open the file
    with h5py.File(h5_file, 'r') as file :
        group = file[group_path]
        args_str = group.attrs["arguments"]

        dico_args = get_arguments_from_str(args_str)

    return dico_args

def get_list_from_str(
    list_str
):
    ''' This function converts a string of a list that have values separated by spaces to a numpy.array
        e.g. : "[1 2 3 4 5]" -> [1,2,3,4,5]
    ''' 

    ## Remove the square brackets
    list_str = list_str[1:-1]

    ## Remove the commas if there are
    list_str = list_str.replace(",", "")

    ## Get the isolated characters
    split = list_str.split(" ")

    
    
    ## Convert to floats and put everything in a list
    list_list = []
    for str_num in split :
        if str_num != '' :
            list_list.append(np.float32(str_num))

    return np.array(list_list)


def group_init(
        h5_file,
        group_path,
        group_args_str
):
    ''' This function aims to initialize a group or to raise an error if there is a problem. 
    What it does is the following :

    1) Check if the group already exists

    2) If it do not exist, creates it and fill its arguments (group_args_str) as attributes

    3) If it does exist, check if the arguments are the same, if not, raise an error

    Arguments :

        - group_args_str : the value of the arguments used for the creation of the data in the group
    it is formated as so : "argument1=argument1_value/argument2=argument2_value/..."

    '''

    with h5py.File(h5_file, 'a') as file :

        ## 1) Check if the group exists
        if not check_group_exists(h5_file,group_path):

            ## 2) If it does not, create it
            file.create_group(group_path)

            ## 2) Add its arguments as attributes
            file[group_path].attrs["arguments"] = group_args_str

        ## 3) If it already exists 
        else :

            ## 3) Get the arguments in the existing group
            existing_group_args_str = file[group_path].attrs["arguments"]

            ## 3) If the arguments don't match
            if group_args_str != existing_group_args_str :
                
                ## 3) Raise an Error
                print('Chosen arguments :')
                print(group_args_str)
                print("Arguments already existing :")
                print(existing_group_args_str)
                print("\n")
                raise ValueError("An error occurred: The chosen arguments do not match the already existing ones for this function, please make sure to select matching ones or to empty the file by setting 'overwrite' to 'true'.")



def check_arguments(
    h5_file, 
    list_group_path,
    group_args_str = None
):

    ''' This function verifies if all the groups where computed using the same arguments.
        It returns True if the groups are compatible and False if they are not.
        It is also possible to add a string of arguments, the function will compare the arguments to the other groups. (this is useful if you want to check only some arguments, or arguments of a group not created yet)

        NB: If one argument was a list and the other an array it will say that they are not the same, even if they have the same values

        Arguments :
            
            - h5_file        : Path to the HDF5 file
            - list_group     : List of the groups' path
            - group_args_str : (optional) String of arguments to compare with the other groups

        Outputs :

            - are_compatible : returns True if they are compatible
    '''

    same_arguments = True
    
    #TODO if too long change this method

    ## Get every pair of group
    pairs = list(combinations(list_group_path,2))

    ## For every pair 
    for pair in pairs :

        ## Get the paths
        group1_path = pair[0]
        group2_path = pair[1]

        ## Get the arguments of both groups 
        dico_args1 = get_arguments_from_attributes(h5_file, group1_path)
        dico_args2 = get_arguments_from_attributes(h5_file, group2_path)

        ## Get the arguments shared between both groups
        shared_args = np.intersect1d(np.array(list(dico_args1.keys())), np.array(list(dico_args2.keys())))

        ## Check if the values are the same
        for argument in shared_args :
            if dico_args1[argument] != dico_args2[argument] :
                print(f"'{argument}' differs between the groups : {dico_args1[argument]} vs {dico_args2[argument]}")
                same_arguments = False


    if group_args_str != None :

        ## Get the arguments of from the string
        group1_path = pair[0]
        dico_args1 = get_arguments_from_str(group_args_str)

        for group_path in list_group_path :

            ## Get the arguments of the second group
            dico_args2 = get_arguments_from_attributes(h5_file, group_path)

            ## Get the arguments shared between both groups
            shared_args = np.intersect1d(np.array(list(dico_args1.keys())), np.array(list(dico_args2.keys())))

            ## Check if the values are the same
            for argument in shared_args :

                if dico_args1[argument] != dico_args2[argument] :
                    print(f"'{argument}' differs : {dico_args1[argument]} vs {dico_args2[argument]}")
                    same_arguments = False


    return same_arguments


def check_neurons_presence(
    h5_file,
    list_group_path,
    neuron_ids
):
    ''' This function uses a list of groups in a HDF5 file and checks if the requested neurons are presents in those groups

    Arguments :

        - h5_file      : Path to the HDF5 file
        - list_group   : List of the groups' path, can be a list containing only one value
        - neuron_ids   : List of the requested neurons 
    '''

    ## Open the file
    with h5py.File(h5_file, 'r') as file :

        for group_path in list_group_path :

            ## Access each group
            group = file[group_path]
            
            ## Get a list of neuron indices in the group
            all_neurons = [int(neuron.split("_")[1]) for neuron in list(group.keys())]

            ## Check if the requested neurons are present 
            if len(np.intersect1d(all_neurons, neuron_ids)) != len(neuron_ids) :
                return False
    
    return True
            

def check_compatibility(
    h5_file,
    list_group_path,
    neuron_ids,
    group_args_str = None
):
    ''' This function makes multiple verifications to see if multiple groupss can work together.

        1) It will check if all of the group exists

        2) It will check if their arguments are matching

        3) It will check if the requested neurons are present in every group
    
        Arguments :

            - see 'check_arguments' and 'check_neurons_presence'

        Outputs :

            no output, only raises Errors if there are issues
    '''

    ## 1)
    for group_path in list_group_path :
        if not check_group_exists(h5_file, group_path) :
            raise KeyError(f"An error occurred: The group {group_path} is absent, please execute the required functions before executing this one (check function description)")
            
    ## 2) 
    if not check_arguments(h5_file, list_group_path,group_args_str) :
        raise ValueError("An error occurred: The arguments are not matching accross every group, please make sure that every experiment was run with the same arguments")
    
    ## 3) 
    if not check_neurons_presence(h5_file, list_group_path, neuron_ids,) : 
        raise ValueError("An error occurred: The chosen neurons are not present accross every group, please make sure that every experiment was run for this neuron set")

def check_group_exists_error(
        h5_file, 
        group_path
):
    ''' Same as 'check_group_exists' but returns an error if the group does not exist
    '''

    if not check_group_exists(h5_file, group_path) :
        raise KeyError(f"An error occurred: The group {group_path} is absent, please execute the required functions before executing this one (check function description)")
            
def check_arguments_error(
        h5_file, 
        list_group_path,
        group_args_str = None
):
    ''' Same as 'check_arguments' but returns an error if the groups have unmatching arguments
    '''

    if not check_arguments(h5_file, list_group_path,group_args_str) :
        raise ValueError("An error occurred: The arguments are not matching accross every group, please make sure that every experiment was run with the same arguments")
        
def check_neurons_presence_error(
    h5_file,
    list_group_path,
    neuron_ids
):
    ''' Same as 'check_neurons_presence' but returns an error if the neurons are not found
    '''

    if not check_neurons_presence(h5_file, list_group_path, neuron_ids,) : 
        raise ValueError("An error occurred: The chosen neurons are not present accross every group, please make sure that every experiment was run for this neuron set")


########################################################
##### PART I : Functions that perform pre-analyses #####
########################################################
def get_center_surround_stimulus(
    center_radius,
    center_ori,
    center_sf,
    center_phase,
    center_contrast,
    surround_radius,
    surround_ori,
    surround_sf,
    surround_phase,
    surround_contrast,
    x_pix,
    y_pix,
    pixel_min = -1.7876, 
    pixel_max =  2.1919, 
    size = 2.67,
    img_res = [93,93],
    device = None
):
    ''' This function aims to create an image containing a center circular grating patch
        and a surround annular patch.
    '''

    ## Select device
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ## Convert the size to the right shape ( [size_height, size_width] )
    if type(size) is int or type(size) is float: 
        size = [size]*2
        
    ## Make images 
    ## Make Center stimulus
    grating_center = torch.Tensor(imagen.SineGrating(mask_shape=imagen.Disk(smoothing=0.0, size=center_radius*2.0),
                    orientation=center_ori,
                    frequency=center_sf,
                    phase=center_phase,
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
                        orientation=surround_ori,
                        frequency=surround_sf,
                        phase=surround_phase,
                        bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))),
                        offset = -1,
                        scale=2,   
                        xdensity=img_res[1]/size[1],
                        ydensity=img_res[0]/size[0], 
                        x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                        y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())

    # TODO delete
    # ## Create the outer edge of the ring
    # outer_edge = torch.Tensor(imagen.Disk(
    #                     smoothing=0.0, 
    #                     size=(min(size[0],size[1])), #Extend it to the border
    #                     bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))), 
    #                     xdensity=img_res[1]/size[1],
    #                     ydensity=img_res[0]/size[0], 
    #                     x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
    #                     y = -get_offset_in_degr(y_pix, img_res[0], size[0]))())
    # grating_ring = grating_background * outer_edge
    
    ## Create the inner edge of the ring
    inner_edge = (torch.Tensor(imagen.Disk(
                        smoothing=0.0, 
                        size=(surround_radius)*2.0,
                        bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))), 
                        xdensity=img_res[1]/size[1],
                        ydensity=img_res[0]/size[0], 
                        x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                        y = -get_offset_in_degr(y_pix, img_res[0], size[0]))()) * -1) + 1
    # grating_ring = grating_ring * inner_edge

    grating_ring = grating_background * inner_edge

    ## Change the contrast 
    grating_center *= center_contrast
    grating_ring   *= surround_contrast

    ## Merge the center and the surround
    image = grating_center + grating_ring

    ## Convert to the right shape for the model
    image = rescale(image,-1,1,pixel_min,pixel_max).reshape(1,*img_res).to(device)

    return image


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


def get_all_grating_parameters_save(
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
    and write the preferred parameters of the selected neurons in the HDF5 file

    NB : The 'phase' that is saved is not the phase to use in other experiments since it is the phase for full fields gratings.
         The phase needs to adapt the shape of the stimulus, so another phase will be computed later with the size tuning experiment, it will be the phase working for the stimulus' radius
      

    Arguments :

        - h5_file           : The HDF5 file containing the data
        - overwrite         : If set to True, will erase the data in the group "full_field_params" and then fill it again. 
                              If set to False, the function will conserve the existing data and only add the one not already present
        - all_neurons_model : The full model containing every neurons (In our analysis it correspond to the v1_convnext_ensemble)
        - neuron_ids        : The list (or array) of the neurons we wish to perform the analysis on
        - others            : Explained in 'find_preferred_grating_parameters_full_field'
    '''

    ## Create the group path
    group_path = "/full_field_params"

    ## Clear the group if requested
    if overwrite : 
        clear_group(h5_file,group_path)

    ## This will serve to create and verify the arguments of the group
    args_str = f"orientations={orientations}/spatial_frequencies={spatial_frequencies}/phases={phases}/contrast={contrast}/img_res={img_res}/pixel_min={pixel_min}/pixel_max={pixel_max}/size={size}"

    ## Create the group if it doesn't exist, and, if it already exists, check if the arguments are matching
    group_init(h5_file, group_path, args_str)
 
    with h5py.File(h5_file, 'a') as file :
        ## Select the group
        group = file[group_path]

        ## Add description
        group.attrs["description"] = "This group contains the preferred parameters for every neuron (one dataset per neuron). datasets = [orientation, sf, phase_full_field] "

        ## For every selected neuron
        for id in tqdm(neuron_ids) :

            ## Get the single cell model of this neuron
            single_model = SingleCellModel(all_neurons_model,id)

            ## Find its preferred parameters
            max_ori, max_sf, max_phase, _, _ = find_preferred_grating_parameters_full_field(
                single_model = single_model, 
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
            
            ## Put everything into a list
            data = [max_ori, max_sf, max_phase]
            
            ## Create a dataset for the neuron if it doen't already exists
            try :
                group.create_dataset(f"neuron_{id}", data=data)

            except ValueError :
                pass


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
        TODO : maybe useless to save the preferred phase, check if we use it eventually
        TODO : renommer, retirer le "Fast" et retirer la deuxième version

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
            group.attrs["description"] = "This group contains the preferred parameters for every neuron (one dataset per neuron). datasets = [orientation, sf, phase_full_field] "

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
        
        ## Convert the list of tensors to a unique tensor
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
        group.attrs["description"] = "This group contains the preferred position for every neuron (one dataset per neuron). datasets = [x_pix, y_pix, error] "     

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


def size_tuning_all_phases(
    single_model, 
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
    size = 2.67,
    img_res = [93,93]
    ):
    ''' 
    TODO : DELETE THIS FUNCTION
    
        This function tries to find the radius size and grating phase of a circular
        grating patch that maximises the response of a single neuron model. 
        Other parameters such as grating orientaion, contrast, and spatial frequency are fixed.
        
        This function is an annotated copy of  "analyses.size_tuning_all_phases".

        Arguments : 

            - single_model        : A single cell model of the class 'surroundmodulation.models.SingleCellModel'
            - x_pix               : The x coordinates of the center of the neuron's receptive field (in pixel, can be a float)
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
    single_model.to(device)

    ## Evaluation mode
    single_model.eval()

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
                resp[i, j] = single_model(input)
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
    ''' This function converts the distance in pixels to a distance in degrees along one axis.

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
    TODO : DELETE THIS FUNCTION

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

    return stim


def orientation_contrast(
    single_model, 
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
    size = 2.67
    ):

    ''' 
        TODO : DELETE THIS FUNCTION

        This function takes a Single Neuron Model and returns an array of the stimulations
        and the model's response for every combination of phase and surround orientation.
        This function uses the "get_orientation_contrast_stim" function in order to create the stimuli.

        Arguments :
            - single_model      : A single cell model of the class 'surroundmodulation.models.SingleCellModel'
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
    single_model.to(device)

    ## Evaluation mode
    single_model.eval()

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
            oc_resp = single_model(oc_stim)

            ## Save the stimuli and responses
            oc_stims.append(oc_stim.detach().cpu().numpy().squeeze())
            oc_resps.append(oc_resp.detach().cpu().numpy().squeeze())
        
        ## Convert the list of arrays into arrays
        oc_stims = np.stack(oc_stims)
        print(oc_stims.shape)
        oc_resps = np.stack(oc_resps)

    return oc_stims, oc_resps


#########################################
##### DATA ACQUISITION AND ANALYSES #####
#########################################


#%%
def get_size_tuning_curves_save(
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
    neg_val = True,     #TODO remove all neg_val maybe useless
    compute_annular = True
    ):
    
    ''' This function is a size tuning function that makes the size of circular and annular grating images vary 
        with their prefered parameters fixed. The difference between the response of the model for each image and the response for a gray image 
        is calculated and saved into an array of the same size of the argument "radii". You can afterwards plot these arrays
        to visualise the effect of the radii on the model's response. If neg_val is set to 'False', the negative values
        are set to 0.
        
        NB : We substract the response to gray image to the response of stimuli because it is necessary for the rest of our analysis, it normalises the values.

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
    neg_val = True,     #TODO remove all neg_val maybe useless
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

        ## Add the description
        group_size_tuning.attrs["description"] = "This group contains different subgroups : curves, results"

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
    center_contrasts = np.logspace(-2,np.log10(1),18),  #TODO change this value to np.logspace(np.log10(0.06),np.log10(1),18) ?
    surround_contrasts = np.logspace(-2,np.log10(1),18), #TODO np.logspace(np.log10(0.06),np.log10(1),6)
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
                plot_img(image, pixel_min, pixel_max)

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

        ## Add the description
        group_cr.attrs["description"] = "This group contains different subgroups : curves, results"

        
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

    #TODO : check which device is circular_curve on
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

        ## Add the description
        #TODO mqybe get rid of descriptions
        group_cst.attrs["description"] = "This group contains different subgroups : curves, results"

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


########################################
##### VISUALISATION OF THE RESULTS #####
########################################


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
    plt.xlabel("diameter (deg)")
    plt.ylabel('response')
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
    minedge = max(0.06,xymin - 0.5) #TODO 0.06 works ? or 0.08 had no pb
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    maxedge = xymax +1
    ax.set_xlim(minedge, maxedge)
    ax.set_ylim(minedge, maxedge)

    # Plot diagonal line
    ax.plot([minedge, maxedge], [minedge, maxedge], 'k--')

    ## The histograms
    nbins = 15
    if log_axes :
        ax_histx.hist(x, density = True, bins = np.logspace(np.log10(minedge), np.log10(maxedge),nbins))
        ax_histy.hist(y, density = True, orientation='horizontal', bins = np.logspace(np.log10(minedge), np.log10(maxedge),nbins))
    else :
        ax_histx.hist(x, density = True, bins = np.linspace(minedge, maxedge,nbins))
        ax_histy.hist(y, density = True, orientation='horizontal', bins = np.linspace(minedge, maxedge,nbins))
    
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    
    plt.show()


def size_tuning_results_1(  #plot_scatter_GSF_surround_extent
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

    plt.hist(all_SI, bins = bins,edgecolor='black', density=True)
    plt.xlabel("Suppression Index (SI)")
    plt.ylabel("Density")
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
                GSF_low,_,_,_,_ = get_GSF_surround_AMRF(radii = radii,circular_tuning_curve=low_contrast_curve, annular_tuning_curve = None)
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

# %%


