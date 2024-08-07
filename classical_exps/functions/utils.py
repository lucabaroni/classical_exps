#####################################################################
#####    THIS CODE CONTAINS EVERY ADDITIONAL USEFUL FUNCTION    #####
#####    ---------------------------------------------------    #####
#####################################################################


############################
##### PART 0 : Imports #####
############################

## Useful
import numpy as np
import torch
from itertools import combinations
import pickle
## Image generation
import imagen
from imagen.image import BoundingBox
## Data storage
import h5py

#############################################################
##### PART I : Functions that help manage the HDF5 file #####
#############################################################


# As the project works with a HDF5 file, a lot of functions are making
# some verifications in order to make sure everything works properly 
# and stays consistant in that file. 
# Exemple : verify parameters, neurons presence, required analyses...


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
    

def get_arguments_from_str(
    args_str
):
    
    ''' This function takes a formated string and creates a dictionnary of the arguments values from it.    

        Arguments :

            - args_str : string of the following format : "argument1=argument1_value/argument2=argument2_value/..."
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


###########################################################################################
##### PART II : Functions facilitating the stimulus generation (coded by Luca Baroni) #####
###########################################################################################

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


def rescale(x, in_min, in_max, out_min, out_max):
    in_mean = (in_min + in_max) / 2
    out_mean = (out_min + out_max) / 2
    in_ext = in_max - in_min
    out_ext = out_max - out_min
    gain = out_ext / in_ext
    x_rescaled = (x - in_mean) * gain + out_mean
    
    return x_rescaled


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

   
    ## Create the inner edge of the ring
    inner_edge = (torch.Tensor(imagen.Disk(
                        smoothing=0.0, 
                        size=(surround_radius)*2.0,
                        bounds=BoundingBox(points = ((-size[1]/2, -size[0]/2), (size[1]/2, size[0]/2))), 
                        xdensity=img_res[1]/size[1],
                        ydensity=img_res[0]/size[0], 
                        x = get_offset_in_degr(x_pix, img_res[1], size[1]), 
                        y = -get_offset_in_degr(y_pix, img_res[0], size[0]))()) * -1) + 1

    grating_ring = grating_background * inner_edge

    ## Change the contrast 
    grating_center *= center_contrast
    grating_ring   *= surround_contrast

    ## Merge the center and the surround
    image = grating_center + grating_ring

    ## Convert to the right shape for the model
    image = rescale(image,-1,1,pixel_min,pixel_max).reshape(1,*img_res).to(device)

    return image

def plot_img(img, pixel_min, pixel_max, title = None, name = None, showfig=True):
    ''' This function displays an image
    '''
    if type(img) != np.ndarray:
        img = img.cpu().detach().squeeze().numpy()
    plt.imshow(img, cmap="Greys_r", vmax=pixel_max, vmin=pixel_min)
    if title != None:
        plt.title(title)
    plt.colorbar()
    if name != None:
        plt.savefig(name)
    if showfig==True:
        plt.show()

#####################################################################
##### PART III : Random useful functions (coded by Luca Baroni) #####
#####################################################################

def pickleread(path):

    with open(path, 'rb') as f:
        x = pickle.load(f)

    return x

def execute_function(func_name, params):
    # Get the function object by name
    func = globals().get(func_name)
    
    # Check if the function exists
    if func is None or not callable(func):
        raise ValueError(f"Function '{func_name}' not found or not callable.")
    
    # Call the function with the parameters
    func(**params)