# import matplotlib.pyplot as plt
# from PIL import Image
# import os
# import numpy as np

# # Read the text file
# file_path = './train_test_inputs/landward_path_test_with_matched_features.txt'  # Replace with your file path
# with open(file_path, 'r') as f:
#     lines = f.readlines()

# # Process and display images
# for line in lines:
#     original_image_path, ground_truth_depth_path = line.strip().split()[:2]
    
#     # Replace the specified directory in the original image path for the predicted image path
#     predicted_image_path = original_image_path.replace('/imgs/', '/depth_pred/')
#     error_image_path = original_image_path.replace('/imgs/', '/error_map/')
    
#     # Load images
#     original_image = Image.open(original_image_path)
#     ground_truth_depth = np.array(Image.open(ground_truth_depth_path), dtype=np.float32)
#     # print(np.shape(ground_truth_depth))
    
#     # Check if the predicted image exists, otherwise skip or handle as needed
#     if os.path.exists(predicted_image_path):
#         predicted_image = np.array(Image.open(predicted_image_path), dtype=np.float32)
#     else:
#         print(f"Predicted image not found: {predicted_image_path}")
#         continue

#     if os.path.exists(error_image_path):
#         error_image = np.array(Image.open(error_image_path), dtype=np.float32)
#     else:
#         print(f"Predicted image not found: {error_image_path}")
#         continue
    
#     # Display images in a row
#     fig, ax = plt.subplots(1, 4, figsize=(15, 5))
#     ax[0].imshow(original_image)
#     ax[0].set_title('Original Image')
#     ax[0].axis('off')
    
#     im1 = ax[1].imshow(np.array(ground_truth_depth), cmap='viridis')  # Adjust cmap as needed
#     ax[1].set_title('Ground Truth Depth')
#     ax[1].axis('off')
#     # cbar1 = plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
#     # cbar1.set_label('Depth')
    
#     # Display predicted depth with colorbar
#     im2 = ax[2].imshow(np.array(predicted_image), cmap='viridis')  # Adjust cmap as needed
#     ax[2].set_title('Predicted Depth')
#     ax[2].axis('off')
#     # cbar2 = plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
#     # cbar2.set_label('Depth')

#     im3 = ax[3].imshow(np.array(error_image), cmap='viridis')  # Adjust cmap as needed
#     ax[3].set_title('Error Map')
#     ax[3].axis('off')
#     cbar3 = plt.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)
#     cbar3.set_label('Depth')
    
#     plt.show()
    
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

# Read the text file
file_path = './train_test_inputs/landward_path_test_with_matched_features.txt'  # Replace with your file path
with open(file_path, 'r') as f:
    lines = f.readlines()

# Process and display images in groups of five
batch_size = 5
for i in range(0, len(lines), batch_size):
    fig, axs = plt.subplots(batch_size, 4, figsize=(20, 5 * batch_size))
    
    for j in range(batch_size):
        if i + j >= len(lines):
            break  # Break if there are no more images to process
        
        line = lines[i + j]
        original_image_path, ground_truth_depth_path = line.strip().split()[:2]

        # Replace the specified directory in the original image path for the predicted and error image paths
        predicted_image_path = original_image_path.replace('/imgs/', '/depth_pred/')
        error_image_path = original_image_path.replace('/imgs/', '/error_map/')

        # Load images
        original_image = Image.open(original_image_path)
        ground_truth_depth = np.array(Image.open(ground_truth_depth_path), dtype=np.float32)

        # Load predicted and error images
        if os.path.exists(predicted_image_path):
            predicted_image = np.array(Image.open(predicted_image_path), dtype=np.float32)
        else:
            print(f"Predicted image not found: {predicted_image_path}")
            continue

        if os.path.exists(error_image_path):
            error_image = np.array(Image.open(error_image_path), dtype=np.float32)
        else:
            print(f"Error image not found: {error_image_path}")
            continue

        # Display images in a row
        axs[j, 0].imshow(original_image)
        axs[j, 0].set_title('Original Image')
        axs[j, 0].axis('off')

        im1 = axs[j, 1].imshow(ground_truth_depth, cmap='viridis')  # Adjust cmap as needed
        axs[j, 1].set_title('Ground Truth Depth')
        axs[j, 1].axis('off')

        im2 = axs[j, 2].imshow(predicted_image, cmap='viridis')  # Adjust cmap as needed
        axs[j, 2].set_title('Predicted Depth')
        axs[j, 2].axis('off')

        im3 = axs[j, 3].imshow(error_image, cmap='viridis')  # Adjust cmap as needed
        axs[j, 3].set_title('Error Map')
        axs[j, 3].axis('off')
        cbar3 = plt.colorbar(im3, ax=axs[j, 3], fraction=0.046, pad=0.04)
        cbar3.set_label('Depth')

    plt.tight_layout()
    plt.show()
