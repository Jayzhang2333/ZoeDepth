import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# File paths
original_image_path = '/media/jay/apple/FLSea_latest/archive/red_sea/big_dice_loop/big_dice_loop/imgs/16316007748831346.tiff'
ground_truth_depth_path = original_image_path.replace('.tiff', '_SeaErra_abs_depth.tif').replace('/imgs/', '/depth/')
predicted_image_path = original_image_path.replace('/imgs/', '/depth_pred/')
error_image_path = original_image_path.replace('/imgs/', '/error_map/')

# Load images
original_image = Image.open(original_image_path)
ground_truth_depth = np.array(Image.open(ground_truth_depth_path), dtype=np.float32)

if os.path.exists(predicted_image_path):
    predicted_image = np.array(Image.open(predicted_image_path), dtype=np.float32)
else:
    predicted_image = None
    print(f"Predicted image not found: {predicted_image_path}")

if os.path.exists(error_image_path):
    error_image = np.array(Image.open(error_image_path), dtype=np.float32)
else:
    error_image = None
    print(f"Error image not found: {error_image_path}")

# Plot and save images with 'inferno_r' colormap if available
def save_image_as_png(data, save_path, cmap):
    if data is not None:
        plt.imshow(data, cmap=cmap)
        # plt.colorbar()
        # plt.title(title)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight',pad_inches=0)
        plt.close()
    else:
        print(f"Cannot save image: Data is None")

ground_truth_depth[ground_truth_depth==0] = 10
ground_truth_depth[ground_truth_depth>10] = 10
save_image_as_png(ground_truth_depth, '/home/jay/modified_zoedepth_results/13/ground_truth_depth.png', 'inferno_r')
if predicted_image is not None:
    save_image_as_png(predicted_image, '/home/jay/modified_zoedepth_results/13/predicted_image.png', 'inferno_r')
if error_image is not None:
    save_image_as_png(error_image, '/home/jay/modified_zoedepth_results/13/error_map.png', 'inferno')

