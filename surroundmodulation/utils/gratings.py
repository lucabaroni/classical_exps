import pickle
import numpy as np
import imagen
from imagen.image import BoundingBox
from surroundmodulation.models import keyless_ensamble_monkey_model
from surroundmodulation.utils.model_utils import single_cell_model
from surroundmodulation.utils.plot_utils import plot_img
from surroundmodulation.utils.misc import rescale
import torch
from tqdm import tqdm
import sys

import logging
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)


# Get the logger for 'param'
param_logger = logging.getLogger('param')

param_logger.setLevel(logging.ERROR)

def present_all_full_field_gratings(model, device, return_max = False):
    model.to(device)
    model.eval()
    d_max = {}
    resp_d = {}
    resp_max = 0
    with torch.no_grad():
        for frequency in tqdm(np.linspace(0.1, 2, 20)):
            for orientation in np.linspace(0, 360, 36):
                input =torch.Tensor(np.stack([imagen.SineGrating(
                    orientation = orientation, 
                    frequency = frequency,
                    phase = phase,
                    bounds = BoundingBox(radius=6.67/2) ,
                    offset = 0,
                    scale = 1,
                    xdensity = 93/6.67,
                    ydensity = 93/6.67,
                )().to(device).reshape(1,93,93) for phase in np.linspace(0, 360, 36)]))

                input = rescale(input, 0, 1, pixel_min, pixel_max)
                resp=model(input)
                resp_d[f'ori={orientation:.2f}_sf={frequency:.2f}_={orientation:.2f}']= resp.detach().cpu().numpy().squeeze()
                if return_max:
                    if resp>resp_max:
                        resp_max = resp

    if return_max: 
        return resp_d, d_max
    else:   
        return resp_d
