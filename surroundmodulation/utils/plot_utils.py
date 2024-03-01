import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision.utils import make_grid
from surroundmodulation.utils.misc import rescale


def plot_f(f, title="", vmin=None, vmax=None, return_plt=False, ticks=True, cmap=None):
    plt.clf()
    if type(f) == torch.Tensor:
        f = f.detach().cpu().numpy().squeeze()
    m = np.max(np.abs(f))
    if vmin is None:
        min = -m
    else:
        min = vmin
    if vmax is None:
        max = m
    else:
        max = vmax
    if cmap == "greys":
        color_map = "Greys_r"
    else:
        color_map = cm.coolwarm
    plt.imshow(f, vmax=max, vmin=min, cmap=color_map)
    plt.title(title)
    plt.colorbar()
    if ticks == False:
        plt.xticks([])
        plt.yticks([])
    if return_plt:
        return plt
    else:
        plt.show()


def plot_img(img, pixel_min, pixel_max, title = None, name = None, showfig=True):
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


def plot_filters(filters, nrow, figsize=None, cmap=None, vmin=None, vmax=None):
    fig, ax = (
        plt.subplots(dpi=200)
        if figsize is None
        else plt.subplots(dpi=200, figsize=figsize)
    )
    image_grid = make_grid(filters, nrow=nrow).mean(0).cpu().data.numpy()
    if vmin == None:
        vmin = -np.abs(image_grid).max()
    if vmax == None:
        vmax = np.abs(image_grid).max()
    ax.imshow(image_grid, vmin=vmin, vmax=vmax, cmap=cmap)
    return fig, ax


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
         # Display using matplotlib
        ax.set_title(title)
        ax.imshow(overlay_image, cmap='Greys_r', vmin=display_min, vmax=display_max)
        ax.axis('off')  # Turn off axis numbers and ticks
    plt.tight_layout()
    plt.show()