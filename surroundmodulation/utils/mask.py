import numpy as np
import torch
from scipy import ndimage, interpolate
from skimage import morphology


def create_mask_from_mei(mei, zscore_thresh = 1.5, closing_iters=2, gaussian_sigma =1):
    params = {
        'zscore_thresh' : zscore_thresh,
        'closing_iters' : closing_iters, 
        'gaussian_sigma' : gaussian_sigma, 
    }   

    if type(mei) == type(torch.ones(1)):
        mei = mei.detach().cpu().numpy().squeeze()
    else: 
        mei = mei.squeeze()

    norm_mei = (mei - mei.mean()) / mei.std()

    thresholded = np.abs(norm_mei) > params['zscore_thresh']
    # Remove small holes in the thresholded image and connect any stranding pixels
    closed = ndimage.binary_closing(thresholded, iterations=params['closing_iters'])
    #%%
    # Remove any remaining small objects
    labeled = morphology.label(closed, connectivity=2)
    most_frequent = np.argmax(np.bincount(labeled.ravel())[1:]) + 1
    oneobject = labeled == most_frequent

    # Create convex hull just to close any remaining holes and so it doesn't look weird
    hull = morphology.convex_hull_image(oneobject)

    # Smooth edges
    smoothed = ndimage.gaussian_filter(hull.astype(np.float32),
                                        sigma=params['gaussian_sigma'])
    mask = smoothed  # final mask

    # Compute mask centroid
    px_y, px_x = (coords.mean() + 0.5 for coords in np.nonzero(hull))
    mask_y, mask_x = px_y - mask.shape[0] / 2, px_x - mask.shape[1] / 2

    return mask, mask_y, mask_x

def find_mask_center(mask):
    # Create coordinate grids
    y, x = np.indices(mask.shape)

    # Calculate weighted centroids
    centroid_x = np.sum(x * mask) / np.sum(mask)
    centroid_y = np.sum(y * mask) / np.sum(mask)

    # print(f"Centroid: x={centroid_x}, y={centroid_y}")
    return (centroid_x, centroid_y)