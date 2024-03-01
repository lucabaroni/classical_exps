
#%%
import numpy as np
from surroundmodulation.utils.misc import rescale, pickleread, picklesave
import matplotlib.pyplot as plt 
x = pickleread('/project/experiment_data/convnext/data.pickle')
import cv2

def plot_grid(images, grid_size=(6,6), masks=None, titles=None, pixel_min = -1.7876, pixel_max = 2.1919, display_min=0, display_max=255, figsize=(13,13)):
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=figsize)
    if masks == None: 
        masks = [None]*grid_size[0]*grid_size[1]
    else:
        print("Can't rescale images due to mask presence: minmax set to (0, 255)")
    if titles ==None: 
        titles = [None]*grid_size[0]*grid_size[1]
    for ax, original_image, mask, title in zip(axes.ravel(), images, masks, titles):
        normalized_image = rescale(original_image, pixel_min, pixel_max, 0, 1)
        if mask is not None:
            # Threshold the mask
            _, binary = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)

            # # Convert binary float mask to uint8 for contour detection
            binary_uint8 = (binary * 255).astype(np.uint8)

            # # Find contours
            contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Sort contours by area and keep the largest one
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            largest_contour = contours[0]
            
            # Create an RGB version of the original image to overlay contour
            overlay_image = np.stack([normalized_image]*3, axis=-1)
            # Draw the largest contour on the original image in red
            cv2.drawContours(overlay_image, [largest_contour], -1, (1, 0, 0), 1)
        else: 
            overlay_image = rescale(normalized_image, 0, 1,0, 255)
        #  Display using matplotlib
            ax.set_title(title)
        ax.imshow(overlay_image, cmap='Greys_r', vmin=display_min, vmax=display_max)
        ax.axis('off')  # Turn off axis numbers and ticks
    plt.tight_layout()
    plt.show()

keys = x.keys()

meis = [x[idx]['mei'] for idx in keys]
plot_grid(
    images = meis,
    titles = [str(k) for k in keys],
    grid_size=(10,20), 
    figsize=(30,15),
    display_min=55,
     display_max=200,
)


# %%
idxs = [255, 434, 61, 62, 183, 453, 188, 222, 252, 263, 327, 246, 30, 320, 89, 187, 224, 198, 431, 306, 368, 79, 291, 240, 95, 456, 340, 285, 129, 203,  67, 206, 18,  294,  357, 356, 204, 447, 166,  169,  100,  234, 54, 178,  276, 413, 452,  394,  257, 400,  156, 247, 313,  348, 59, 99,  280, 435, 103,  126, 440,  284,  442,  381,  153, 275,  342, 182,  328,  332, 233,  201,  392,  355, 213, 133,  243,  331,  362,  136,  192,  316, 199,  25,  432,  229,  8,  2,  0,  329,  412,  150, 367, 303]

picklesave('/project/experiment_data/convnext/gabor_idx.pickle', idxs )
for idx in idxs:
    if idx not in x.keys():
        print(idx)

# # %%
# pickleread('/project/experiment_data/convnext/data.pickle',)
# %%
