#%%
import os 
import matplotlib.pyplot as plt

import numpy as np
from surroundmodulation.models import SingleCellModel

from tqdm import tqdm
from nnvision.models.trained_models.v1_task_fine_tuned import v1_convnext_ensemble
from imagen.image import BoundingBox
from surroundmodulation.utils.misc import pickleread, picklesave
from surroundmodulation.analyses import *

#%%
std = 0.05
mean = 0
orientations = np.linspace(0, np.pi, 37)[:-1]
spatial_frequencies = np.linspace(1, 7, 25)
radii = np.linspace(0.05, 2, 40)#this 
img_res = [93, 93]
size= [2.35,2.35]
gap = 0.2

# idxs = np.array(pickleread('/project/experiment_data/convnext/gabor_idx.pickle'))
idxs = [0]
device = f'cuda'

#%%
print(device)
d = {}
# picklesave(f'/project/experiment_data/convnext/data_v4.pickle', d)
for idx in tqdm(idxs):
    model = SingleCellModel(v1_convnext_ensemble, idx)
    mei, mei_act = create_mei(model, gaussianblur=3., device = device)
    mei_mask,  px, py = create_mask_from_mei(mei, zscore_thresh=1.25)
    exc_full_surr, exc_only_surr, exc_full_surr_act, exc_only_surr_act = create_surround(model, mei, mei_mask, objective='max', gaussianblur=3., device = device)
    inh_full_surr, inh_only_surr, inh_full_surr_act, inh_only_surr_act = create_surround(model, mei, mei_mask, objective='min', gaussianblur=3., device = device)
    max_ori, max_sf, max_phase, max_stim, max_resp = find_preferred_grating_parameters(model, mei_mask, device = device, contrast=0.2, spatial_frequencies=spatial_frequencies)
    top_radius, top_phase, top_grating, top_resp, st_resp, st_gratings = size_tuning_all_phases(model, px, py, max_ori, max_sf, return_all=True, device = device, contrast =0.2, radii = radii)
    oc_stims, oc_resps = orientation_contrast(
        model, 
        x_pix = px, 
        y_pix= py,
        preferred_ori=max_ori, 
        preferred_sf = max_sf,
        center_radius=top_radius,
        orientation_diffs = np.linspace(0, np.pi, 37)[:-1], 
        phases = top_phase + np.linspace(0, 2*np.pi, 19)[:-1], 
        gap=gap, 
        contrast = 0.2
    )
    d[idx] = {
        'mei': mei, 
        'mei_act': mei_act, 
        'mask_mei': mei_mask,
        'center_mask_mei': [px, py],
        'exc_full_surr' : exc_full_surr, 
        'exc_only_surr' : exc_only_surr, 
        'exc_full_surr_act' : exc_full_surr_act, 
        'exc_only_surr_act' : exc_only_surr_act, 
        'inh_full_surr' : inh_full_surr, 
        'inh_only_surr' : inh_only_surr, 
        'inh_full_surr_act' : inh_full_surr_act, 
        'inh_only_surr_act' : inh_only_surr_act, 
        'masked_grating_max_ori' : max_ori,
        'masked_grating_max_sf' : max_sf,
        'max_phase_max_phase' : max_phase, 
        'masked_grating_max_stim' : max_stim,
        'masked_grating_max_resp' : max_resp, 
        'size_tuning_top_radius' : top_radius,
        'size_tuning_top_phase' : top_phase,
        'size_tuning_top_grating' : top_grating,
        'size_tuning_resp' : st_resp,
        'oc_resps': oc_resps,
        'oc_stims': oc_stims,
        }
    picklesave(f'/project/experiment_data/convnext/data_vtest.pickle', d)

# %%
d= pickleread('/project/experiment_data/convnext/data_v2.pickle')

#%% estimate size of neurons with size tuning:
def diameter(x1s, y1s, x2s, y2s):
    x1 = np.tile(x1s, (len(x2s), 1))
    y1 = np.tile(y1s, (len(y2s), 1))
    diameters = np.sqrt((x2s[:, None]-x1)**2 + (y2s[:, None]-y1)**2)
#     print(x1s.shape, x2s.shape, diameters.shape)
    return np.max(diameters)

all_masks = [d[idx]['mask_mei'] for idx in idxs]

# compute MEI mask size
ratio = 2.35/all_masks[0].shape[1]
y, x = np.indices(all_masks[0].shape)
threshold = 0.5
threshold_mask = np.array([mask>threshold for mask in all_masks])
threshold_indices = np.array([[y[mask], x[mask]] for mask in threshold_mask])
# mei_diameter = np.array([diameter(*indices1, *indices2) for indices1, indices2 in zip(threshold_indices[:-1], threshold_indices[1:])]) * ratio

mei_diameter = np.array([diameter(*indices1, *indices2) for indices1, indices2 in zip(threshold_indices, threshold_indices)]) * ratio
print(mei_diameter.mean())
#%% reimplementation
dists = []
for i in range(len(threshold_indices)):
    dist = torch.cdist(torch.Tensor(np.stack(threshold_indices[i]).T),torch.Tensor(np.stack(threshold_indices[i]).T)).max()*ratio
    dists.append(dist)
print(np.mean(dists))
#%% grating summation field (GSF)
top_radii = np.array([d[idx]['size_tuning_top_radius'] for idx in idxs])
print(np.mean(top_radii)*2)

#%%
plt.hist(mei_diameter-(top_radii*2))
plt.vlines((mei_diameter-(top_radii*2)).mean(), 0, 25, 'k', label='mean')
plt.title(f'mei_diameter (mean={mei_diameter.mean():.2f}) - GSF_diameter (mean={top_radii.mean()*2:.2f})')
plt.legend()
plt.show()
#%%
# plot_img(d[idxs[0]]['mask_mei'], -0.4, 0.4)
# plot_img(d[idxs[0]]['mei'], -0.4, 0.4)
# plot_img(d[idxs[0]]['masked_grating_max_stim'], -0.4, 0.4)
# plot_img(d[idxs[0]]['size_tuning_top_grating'], -0.4, 0.4)
# plot_img(d[idxs[0]]['oc_stims'][0,0], -0.4, 0.4)

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
for j in idxs:
    fig, axes = plt.subplots(2,4, figsize=(10,6))
    fig.suptitle(str(j), fontsize=16)
    for i, ax in enumerate(axes.flat): 
        if i!=3:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0,0].imshow(d[j]['mei'], cmap='Greys_r', vmin = -.9, vmax=.9)
    axes[0,0].set_title('MEI')
    axes[0,1].imshow(d[j]['masked_grating_max_stim'].cpu().squeeze(), cmap='Greys_r', vmin = -.9, vmax=.9)
    axes[0,1].set_title('best grating within mask')
    axes[0,2].imshow(d[j]['size_tuning_top_grating'], cmap='Greys_r', vmin = -.9, vmax=.9)
    axes[0,2].set_title('grating summation field')
    axes[1,0].imshow(d[j]['exc_full_surr'], cmap='Greys_r', vmin = -.9, vmax=.9)
    axes[1,0].set_title('MEI + exc surr')
    axes[1,1].imshow(d[j]['inh_full_surr'], cmap='Greys_r', vmin = -.9, vmax=.9)
    axes[1,1].set_title('MEI + inh surr')
    axes[1,2].imshow(d[j]['oc_stims'][0,0], cmap='Greys_r', vmin = -.9, vmax=.9)
    axes[1,2].set_title('center + collinear surr')
    axes[1,3].imshow(d[j]['oc_stims'][18,0], cmap='Greys_r', vmin = -.9, vmax=.9)
    axes[1,3].set_title('center + orthogonal surr')
    axes[0,3].plot(radii, d[j]['size_tuning_resp'].mean(axis=-1), )
    axes[0,3].set_xlabel('radius')
    axes[0,3].set_xlabel('response')
    axes[0,3].set_title('size tuning')
    axes[0,3].set_box_aspect(1)
     # Create a ScalarMappable with the colormap and data range used in your images
    # Create a ScalarMappable with the colormap and data range used in your images
    norm = Normalize(vmin=rescale(-.9, -1.7876, 2.1919, 0, 255), vmax=rescale(.9, -1.7876, 2.1919, 0, 255))
    smappable = ScalarMappable(cmap='Greys_r', norm=norm)
    smappable.set_array([])  # Set an empty array for the colorbar
    
    # Add a colorbar to the figure, adjusting pad and fraction for position and width
    cbar = fig.colorbar(smappable, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.03, pad=0.0)
    # cbar.set_label('Intensity')  # Customize this label as needed
    
    plt.tight_layout(rect=[0, 0, 0.86, 1])  # Adjust the rect parameter to make space for the colorbar
    plt.savefig(f'/project/experiments_convnext/figures/summary_{j}.png')
    plt.show()



    

    # for i, ax in enumerate(axes.flat):
    #     img = (d[j][what[i]] if type(d[j][what[i]]) == np.ndarray else d[j][what[i]].cpu().squeeze())
    #     if i == 3:
    #         img = img[0,0]
    #     ax.imshow(img, cmap='Greys_r', vmin = -.4, vmax=.4)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_title(j)
    # plt.show()

#%%
what = ['mei', 'masked_grating_max_stim', 'size_tuning_top_grating', 'oc_stims']
# for j in range(94):
#     fig, axes = plt.subplots(2,2)
#     for i, ax in enumerate(axes.flat):
#         img = (d[idxs[j]][what[i]] if type(d[idxs[j]][what[i]]) == np.ndarray else d[idxs[j]][what[i]].cpu().squeeze())
#         if i == 3:
#             img = img[0,0]
#         ax.imshow(img, cmap='Greys_r', vmin = -.4, vmax=.4)
#         ax.set_xticks([])
#         ax.set_yticks([])
#     plt.show()

# for i in range(94):
#     plt.plot(radii, d[idxs[i]]['size_tuning_resp'].mean(axis=1))
#     plt.title(str(idxs[i]))
#     plt.show()

# for i in range(94):
#     plt.plot(np.linspace(0, np.pi, 37)[:-1], d[idxs[i]]['oc_resps'].mean(axis=1))
#     plt.title(str(i))
#     plt.show()


for j in idxs:
    fig, axes = plt.subplots(2,3)
    for i, ax in enumerate(axes.flat):
        img = (d[j][what[i]] if type(d[j][what[i]]) == np.ndarray else d[j][what[i]].cpu().squeeze())
        if i == 3:
            img = img[0,0]
        ax.imshow(img, cmap='Greys_r', vmin = -.4, vmax=.4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(j)
    plt.show()

for j in idxs:
    fig, axes = plt.subplots(2,2)
    for i, ax in enumerate(axes.flat):
        img = (d[j][what[i]] if type(d[j][what[i]]) == np.ndarray else d[j][what[i]].cpu().squeeze())
        if i == 3:
            img = img[0,0]
        ax.imshow(img, cmap='Greys_r', vmin = -.4, vmax=.4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(j)
    plt.show()

#%%
size_tuning_top_resp = np.array([d[idx]['size_tuning_resp'].mean(axis=-1).max() for idx in idxs])
oc_resp_coll = np.array([d[idx]['oc_resps'][0].mean() for idx in idxs])/size_tuning_top_resp
oc_resp_orth = np.array([d[idx]['oc_resps'][18].mean() for idx in idxs])/size_tuning_top_resp

plt.scatter(oc_resp_coll,oc_resp_orth)
plt.vlines(1, 0, 1.25, 'r', linestyles='dotted')
plt.hlines(1, 0, 1.25, 'r', linestyles='dotted')
plt.plot([0,1.25], [0,1.25], 'r--')
plt.xlabel('collinear surround (normalized response)')
plt.ylabel('orthogonal surround (normalized response)')
plt.show()
# %%
