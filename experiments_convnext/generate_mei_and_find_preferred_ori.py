
#%%
from nnvision.models.trained_models.v1_task_fine_tuned import v1_convnext, v1_convnext_ensemble
import torch 
import numpy as np
import datajoint as dj 
import os 
dj.config["enable_python_native_blobs"] = True
dj.config["database.host"] = os.environ["DJ_HOST"]
dj.config["database.user"] = os.environ["DJ_USER"]
dj.config["database.password"] = os.environ["DJ_PASSWORD"]
from surroundmodulation.models import SingleCellModel
from surroundmodulation.utils.plot_utils import plot_img
import matplotlib.pyplot as plt
from surroundmodulation.analyses import create_mei, create_mask_from_mei, find_mask_center
from surroundmodulation.utils.misc import pickleread, picklesave, sort_and_save_indices_descending
from tqdm import tqdm

#%%
data_d = {}

for idx in tqdm(range(458)):
    model = SingleCellModel(v1_convnext_ensemble, idx)
    model.eval()
    mei, mei_act = create_mei(model, gaussianblur=3.)
    mask_mei,  px, py = create_mask_from_mei(mei, zscore_thresh=1.)

    data_d[idx] = {
        'mei': mei, 
        'mei_act': mei_act, 
        'mask_mei': mask_mei,
        'center_mask_mei': [px, py]
    }

picklesave('/project/experiment_data/convnext/data_all_mei.pickle', data_d)

#%% find preferred ori
import imagen
from imagen.image import BoundingBox
from surroundmodulation.utils.misc import rescale
from surroundmodulation.utils.plot_utils import plot_img

model = v1_convnext_ensemble
orientation = 0
frequency = 1.5
phase = 0
vmin = -1.7876
vmax = 2.1919
model.cuda()
orientations = np.linspace(0, 2*np.pi, 37)[:-1]
phases = np.linspace(0, 2*np.pi, 37)[:-1]

with torch.no_grad():
    resps = np.zeros((len(orientations), len(phases), 458))
    for i, orientation in enumerate(orientations):
        for j, phase in enumerate(phases):
            a = rescale(rescale(imagen.SineGrating(orientation= orientation,
                                    frequency= frequency,
                                    phase= phase,
                                    bounds=BoundingBox(radius=6.67/2),
                                    offset = 0, 
                                    scale=1 ,
                                    xdensity= 93/6.67,
                                    ydensity= 93/6.67)(), 0, 1,-1, 1)*0.7, -1, 1, vmin, vmax)
            grating = torch.Tensor(a).reshape(1, 1, 93, 93).cuda()
            resps[i, j] = model(grating).squeeze().cpu()
pref_ori = []
for i in range(458):
    pref_ori.append(np.degrees(orientations[resps.max(axis=1)[:, i].argmax()]))

# %%
import imagen
from imagen.image import BoundingBox
from surroundmodulation.utils.misc import rescale
from surroundmodulation.utils.plot_utils import plot_img

model = v1_convnext_ensemble
orientation = 0
frequency = 1.5
phase = 0
vmin = -1.7876
vmax = 2.1919
model.cuda()
orientations = np.linspace(0, np.pi, 37)[:-1]
phases = np.linspace(0, 2*np.pi, 37)[:-1]

with torch.no_grad():
    resps = np.zeros((len(orientations), len(phases), 458))
    for i, orientation in enumerate(orientations):
        for j, phase in enumerate(phases):
            a = rescale(rescale(imagen.SineGrating(orientation= orientation,
                                    frequency= frequency,
                                    phase= phase,
                                    bounds=BoundingBox(radius=6.67/2),
                                    offset = 0, 
                                    scale=1 ,
                                    xdensity= 93/6.67,
                                    ydensity= 93/6.67)(), 0, 1,-1, 1)*0.7, -1, 1, vmin, vmax)
            grating = torch.Tensor(a).reshape(1, 1, 93, 93).cuda()
            resps[i, j] = model(grating).squeeze().cpu()
pref_ori1 =[]      
for i in range(458):
    pref_ori1.append(np.degrees(orientations[resps.max(axis=1)[:, i].argmax()]))
 # %%
plt.hist(np.abs(np.array(pref_ori) - np.array(pref_ori1)), bins=30)
plt.show()
# %%
data_d = pickleread('/project/experiment_data/convnext/data_all_mei.pickle')
for i in range(458):
    data_d[i]['preferred_ori'] = pref_ori[i]
picklesave('/project/experiment_data/convnext/data_all_mei_and_ori.pickle', data_d)
# %%
data_d = pickleread('/project/experiment_data/convnext/data_all_mei_and_ori.pickle')
for i in range(40):
    plot_img(data_d[i]['mei'], -2,2, title=str(i) + ' ' + str(data_d[i]['preferred_ori'])) 

    # plt.title(i)

# %%

# %%
