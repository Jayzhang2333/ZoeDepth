import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

# Read the text file
original_image_path = '/media/jay/apple/uw_depth_lizard_data/images/1495.png'  # Replace with your file path
ground_truth_depth_path = original_image_path.replace('.tiff', '_SeaErra_abs_depth.tif')
ground_truth_depth_path = ground_truth_depth_path.replace('/images/', '/depth_tiff/')
ground_truth_depth_path = ground_truth_depth_path.replace('.png', '.tiff')
# Replace the specified directory in the original image path for the predicted image path
predicted_image_path = original_image_path.replace('/images/', '/depth_pred/')
error_image_path = original_image_path.replace('/images/', '/error_map/')

# Load images
original_image = Image.open(original_image_path)
ground_truth_depth = np.array(Image.open(ground_truth_depth_path), dtype=np.float32)
# print(np.shape(ground_truth_depth))

# Check if the predicted image exists, otherwise skip or handle as needed
if os.path.exists(predicted_image_path):
    predicted_image = np.array(Image.open(predicted_image_path), dtype=np.float32)
else:
    print(f"Predicted image not found: {predicted_image_path}")
    

if os.path.exists(error_image_path):
    error_image = np.array(Image.open(error_image_path), dtype=np.float32)
else:
    print(f"Predicted image not found: {error_image_path}")
    
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

ground_truth_depth[ground_truth_depth==0] = 2
ground_truth_depth[ground_truth_depth>2] = 2
save_image_as_png(ground_truth_depth, '/home/jay/modified_zoedepth_results_lizard/5/ground_truth_depth.png', 'viridis')
if predicted_image is not None:
    save_image_as_png(predicted_image, '/home/jay/modified_zoedepth_results_lizard/5/predicted_image.png', 'viridis')
if error_image is not None:
    save_image_as_png(error_image, '/home/jay/modified_zoedepth_results_lizard/5/error_map.png', 'viridis')

