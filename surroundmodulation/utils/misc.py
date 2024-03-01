import numpy as np
import torch
import wandb
import pickle

def sort_and_save_indices_descending(numbers):
    # Sorting the numbers in decreasing order and saving their original indices
    sorted_numbers_with_indices = sorted(enumerate(numbers), key=lambda x: x[1], reverse=True)
    
    # Extracting the sorted numbers and their indices
    sorted_numbers = [item[1] for item in sorted_numbers_with_indices]
    sorted_indices = [item[0] for item in sorted_numbers_with_indices]
    
    return sorted_numbers, sorted_indices

def pickleread(path):
    with open(path, 'rb') as f:
        x = pickle.load(f)
    return x

def picklesave(path, file):
    with open(path, 'wb') as f:
        pickle.dump(file, f)

def rescale(x, in_min, in_max, out_min, out_max):
    in_mean = (in_min + in_max) / 2
    out_mean = (out_min + out_max) / 2
    in_ext = in_max - in_min
    out_ext = out_max - out_min
    gain = out_ext / in_ext
    x_rescaled = (x - in_mean) * gain + out_mean
    return x_rescaled


# def prepare_video(img, pixel_min=None, pixel_max=None, fps=30):
#     """Prepare video in Wandb format, so that it can be logged"""
#     imgs_np = img.cpu().detach().numpy()
#     imgs_np = np.tile(imgs_np, (1, 3, 1, 1))
#     if pixel_min == None:
#         pixel_min = -np.max(np.abs(imgs_np))
#     if pixel_max == None:
#         pixel_max = np.max(np.abs(imgs_np))
#     imgs_video_in_scale = rescale(imgs_np, pixel_min, pixel_max, 0, 255)
#     return wandb.Video(np.uint8(imgs_video_in_scale), fps=fps)


# def prepare_images_for_table(img, pixel_min=None, pixel_max=None):
#     imgs_np = img.cpu().detach().numpy()
#     imgs_np = np.tile(imgs_np, (1, 3, 1, 1))
#     if pixel_min == None:
#         pixel_min = -np.max(np.abs(imgs_np))
#     if pixel_max == None:
#         pixel_max = np.max(np.abs(imgs_np))
#     imgs_in_scale = rescale(imgs_np, pixel_min, pixel_max, 0, 255)
#     return [wandb.Image(np.moveaxis(np.uint8(img), 0, -1)) for img in imgs_in_scale]


def prepare_image(img, pixel_min=None, pixel_max=None):
    img_np = img.cpu().detach().numpy()
    img_np = np.tile(img_np, (3, 1, 1))
    if pixel_min == None:
        pixel_min = -np.max(np.abs(img_np))
    if pixel_max == None:
        pixel_max = np.max(np.abs(img_np))
    img_in_scale = rescale(img_np, pixel_min, pixel_max, 0, 255)
    return wandb.Image(np.moveaxis(np.uint8(img_in_scale), 0, -1))


# def flatten_list(t):
#     return [item for sublist in t for item in sublist]


# def rescale(x, in_min, in_max, out_min, out_max):
#     in_mean = (in_min + in_max) / 2
#     out_mean = (out_min + out_max) / 2
#     in_ext = in_max - in_min
#     out_ext = out_max - out_min
#     gain = out_ext / in_ext
#     x_rescaled = (x - in_mean) * gain + out_mean
#     return x_rescaled


# def standardize(x, dim=(1, 2, 3), return_shift_gain=False):
#     shift = -x.mean(dim=dim, keepdim=True)
#     gain = 1 / x.std(dim=dim, keepdim=True)
#     x_standardized = (x + shift) * gain
#     if return_shift_gain == True:
#         return x_standardized, shift, gain
#     else:
#         return x_standardized


# def normalize(x, dim=(1, 2, 3), return_shift_gain=False):
#     shift = -x.mean(dim=dim, keepdim=True)
#     x_shifted = x + shift
#     gain = 1 / torch.linalg.norm(x_shifted, dim=dim, keepdim=True)
#     x_normalized = x_shifted * gain
#     if return_shift_gain == True:
#         return x_normalized, shift, gain
#     else:
#         return x_normalized


# def rescale_back(x, shift, gain):
#     return (x / gain) - shift


# ##################
# from staticnet.base import CorePlusReadout2d, Elu1
# from staticnet.cores import GaussianLaplaceCore
# from staticnet.readouts import SpatialTransformerPyramid2dReadout
# from staticnet.shifters import MLPShifter
# from staticnet.modulators import MLPModulator
# from featurevis import models


# def build_network(configs):
#     Core = GaussianLaplaceCore
#     Readout = SpatialTransformerPyramid2dReadout
#     Shifter = MLPShifter
#     Modulator = MLPModulator

#     core = Core(input_channels=configs["img_shape"][1], **configs["core_key"])
#     ro_in_shape = CorePlusReadout2d.get_readout_in_shape(core, configs["img_shape"])
#     readout = Readout(ro_in_shape, configs["n_neurons"], **configs["ro_key"])
#     shifter = Shifter(configs["n_neurons"], 2, **configs["shift_key"])
#     modulator = Modulator(configs["n_neurons"], 3, **configs["mod_key"])
#     model = CorePlusReadout2d(
#         core, readout, nonlinearity=Elu1(), shifter=shifter, modulator=modulator
#     )
#     return model


# def load_network(model, state_dict):
#     try:
#         state_dict = {
#             k: torch.as_tensor(state_dict[k][0].copy()) for k in state_dict.dtype.names
#         }
#     except AttributeError:
#         state_dict = {
#             k: torch.as_tensor(state_dict[k].copy()) for k in state_dict.keys()
#         }
#     mod_state_dict = model.state_dict()
#     for k in set(mod_state_dict) - set(state_dict):
#         log.warning(
#             "Could not find paramater {} setting to initialization value".format(
#                 repr(k)
#             )
#         )
#         state_dict[k] = mod_state_dict[k]
#     model.load_state_dict(state_dict)
#     return model


# def forward(
#     grid, cppn, img_transf, encoding_model, resolution_increase_factor=1.0, gb=None
# ):
#     """forward pass throught the pipeline"""
#     img_pre = cppn(grid)
#     img_post = img_transf(img_pre)
#     if gb != None:
#         img_post.register_hook(gb)
#     acts = encoding_model(img_post)

#     if resolution_increase_factor > 1:
#         with torch.no_grad():
#             img_pre_hres = cppn(
#                 grid, img_res=[r * resolution_increase_factor for r in cppn.img_res]
#             )
#             img_post_hres = img_transf(img_pre_hres)
#     else:
#         img_post_hres = img_post

#     return img_pre, img_post, acts, img_post_hres


# def check_activity_requirements(acts, requirements):
#     passed = False
#     if (
#         acts.mean() > requirements["avg"]
#         and acts.std() < requirements["std"]
#         and acts.min() > requirements["necessary_min"]
#     ):
#         passed = True
#     elif acts.min() > requirements["sufficient_min"]:
#         passed = True
#     return passed


# def get_wandb_act_line_plot(act, grid_v, imgs):
#     data = [[x, y, img] for (x, y, img) in zip(grid_v, act, imgs)]
#     table = wandb.Table(data=data, columns=["latent_input", "activation", "img"])
#     return wandb.plot.line(
#         table,
#         "latent_input",
#         "activation",
#         title=f"activation vs latent input ",
#     )


# def get_ensemble_model(configs, state_dict_ls, neuron_id):
#     # Build a specified model
#     model = build_network(configs)

#     # Load model with trained state_dict from 4 different initialization seeds
#     all_models = [
#         load_network(model, state_dict_ls[i]["model"])
#         for i in range(len(state_dict_ls))
#     ]
#     all_models = [load_network(model, state_dict_ls[i]["model"]) for i in range(3, 4)]

#     # Specify mean eye position
#     mean_eyepos = [0, 0]

#     # Create model ensemble
#     mean_eyepos = torch.tensor(
#         mean_eyepos, dtype=torch.float32, device="cuda"
#     ).unsqueeze(0)
#     model_ensemble = models.Ensemble(
#         all_models,
#         configs["key"]["readout_key"],
#         eye_pos=mean_eyepos,
#         neuron_idx=neuron_id,
#         average_batch=False,
#         device="cuda",
#     )
#     return model_ensemble


# def standard_log(
#     cppn,
#     grid_v,
#     img_transforms,
#     model_ensemble,
#     MEI_activation,
#     pixel_min,
#     pixel_max,
# ):
#     with torch.no_grad():
#         cppn.eval()
#         _, img_post, _acts, _ = forward(
#             grid_v,
#             cppn,
#             img_transforms,
#             model_ensemble,
#         )
#         acts = _acts / MEI_activation
#         video = prepare_video(img_post, pixel_min=pixel_min, pixel_max=pixel_max)
#         logging_dict = {
#             "MEIs": video,
#             "act": acts,
#             "act_mean": acts.mean(),
#             "act_std": acts.std(),
#         }
#         cppn.train()
#     return logging_dict


# def satisfied_requirements_log(
#     video, acts, grid_v, img_post, pixel_min, pixel_max, name, **kwargs
# ):
#     print(name + " requirements_satisfied")
#     logging_dict = {}
#     print(acts.shape, grid_v.shape)
#     acts_line_plot = get_wandb_act_line_plot(
#         acts,
#         grid_v.squeeze(),
#         prepare_images_for_table(img_post, pixel_min, pixel_max),
#     )
#     logging_dict.update(
#         {
#             name + "_req/MEIs": video,
#             name + "_req/acts": acts_line_plot,
#         }
#     )
#     return logging_dict
