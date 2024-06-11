#%%
import numpy as np
import torch
from tqdm import tqdm
import os
import math
## Plots
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
from functions_2 import *


# %%

#############################################################################################
## Functions for the ARTICLES IV :                                                         ##
##                                                                                         ##
## A functional and perceptual signature of the second visual area in primates             ##
## Jeremy Freeman, Corey M Ziemba, David J Heeger, Eero P Simoncelli1 & J Anthony Movshon  ##
## DOI : https://doi.org/10.1038/nn.3402.                                                  ##
##                                                                                         ##
#############################################################################################

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
    contrast = 1,  #TODO : maybe change it, for now doesnt interacts with other but it may at one moment, or maybe osef bc it doesnt interact ?
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
            - 5 : '.png' image should be in png TODO am I sure it wont work with other formats ?
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
            - TODO : add random seed

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
    list_imgs_names = [f for f in list_imgs_names if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))] #TODO, only png ?
    
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
            # TODO uncomment
            tex_img = rescale(tex_img, min_val, max_val, -1, 1)*contrast
            tex_img = rescale(tex_img, -1, 1, pixel_min, pixel_max)
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
    contrast = 1,  #TODO : maybe change it, for now doesnt interacts with other but it may at one moment, or maybe osef bc it doesnt interact ?
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
    args_str = f"contrast={contrast}/pixel_min={pixel_min}/pixel_max{pixel_min}/num_samples={num_samples}/img_res={img_res}" #TODO change ?
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



def texture_noise_response_results_1(
    h5_file, 
    neuron_ids,
    wanted_fam_order = None
):  
    ''' This function aims to visualise if the neurons product the same response to texture or noise images
    
        Prerequisite :
            
            - function 'texture_noise_response_experiment' executed for the required neurons

        Filtering :
            
            TODO ???
        
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
            
            TODO ???
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
            
            TODO ???
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


import numpy as np
## Models
from nnvision.models.trained_models.v1_task_fine_tuned import v1_convnext_ensemble
## Functions
from functions   import *
from functions_2 import *
from functions_3 import *
## To select the well predicted neurons 
from surroundmodulation.utils.misc import pickleread
from surroundmodulation.utils.plot_utils import plot_img
h5_file='/project/mathys_code/article1.h5'
n = 458
neuron_ids = np.arange(n) ## Because 458 outputs in our model
corrs = pickleread("/project/mathys_code/avg_corr.pkl") ## The correlation score of the neurons
neuron_ids = neuron_ids[corrs>0.75]


texture_noise_response_results_1(
    h5_file, 
    neuron_ids
)

#%%
