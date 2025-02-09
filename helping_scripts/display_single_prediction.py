import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

# Read the text file
original_image_path = '/media/jay/apple/FLSea_latest/archive/red_sea/sub_pier/sub_pier/imgs/16316026388705614.tiff'  # Replace with your file path
ground_truth_depth_path = original_image_path.replace('.tiff', '_SeaErra_abs_depth.tif')
ground_truth_depth_path = ground_truth_depth_path.replace('/imgs/', '/depth/')
# Replace the specified directory in the original image path for the predicted image path
predicted_image_path = original_image_path.replace('/imgs/', '/depth_pred_masking/')
error_image_path = original_image_path.replace('/imgs/', '/error_map_masking/')

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
    

# Display images in a row
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].imshow(original_image)
ax[0].set_title('Original Image')
ax[0].axis('off')


# # Normalize and invert the depth image
# depth_min = np.min(ground_truth_depth)
# depth_max = np.max(ground_truth_depth)

# # Normalize to range [0, 1]
# ground_truth_depth = (ground_truth_depth - depth_min) / (depth_max - depth_min)

# # Invert the depth values so that far distances become black (0) and close distances become white (1)
# ground_truth_depth = 1 - ground_truth_depth

ground_truth_depth[ground_truth_depth==0] = 10
ground_truth_depth[ground_truth_depth>10] = 10
im1 = ax[1].imshow(np.array(ground_truth_depth), cmap='inferno_r')  # Adjust cmap as needed
ax[1].set_title('Ground Truth Depth')
ax[1].axis('off')
# cbar1 = plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
# cbar1.set_label('Depth')

# # Normalize and invert the depth image
# depth_min = np.min(predicted_image)
# depth_max = np.max(predicted_image)

# # Normalize to range [0, 1]
# predicted_image = (predicted_image - depth_min) / (depth_max - depth_min)

# # Invert the depth values so that far distances become black (0) and close distances become white (1)
# predicted_image = 1 - predicted_image

# Display predicted depth with colorbar



# predicted_image[predicted_image > 5.4] = 10.0


# im2 = ax[2].imshow(np.array(predicted_image), cmap='inferno_r')  # Adjust cmap as needed
# ax[2].set_title('Predicted Depth')
# ax[2].axis('off')
# # cbar2 = plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
# # cbar2.set_label('Depth')

# im3 = ax[3].imshow(np.array(error_image), cmap='inferno')  # Adjust cmap as needed
# ax[3].set_title('Error Map')
# ax[3].axis('off')
# cbar3 = plt.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)
# cbar3.set_label('Depth')

plt.show()