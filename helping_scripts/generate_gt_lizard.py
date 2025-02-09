import numpy as np
import os
from PIL import Image

# Define the path to your folder containing .bin files
folder_path = '/media/jay/apple/uw_depth_lizard_data/depth'
output_folder = '/media/jay/apple/uw_depth_lizard_data/depth_tiff'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Image dimensions
width, height = 1355, 1002

# Process each .bin file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.bin'):
        file_path = os.path.join(folder_path, file_name)
        
        # Read the .bin file
        with open(file_path, 'rb') as f:
            # Load data as a 1D array with the specified data type (e.g., float32)
            depth_data = np.fromfile(f, dtype=np.float32)
        
        # Ignore the first 3 values
        depth_data = depth_data[3:]
        
        # Reshape the data to (height, width)
        try:
            depth_map = depth_data.reshape((height, width))
            
            # Convert the depth map to a PIL image and save as TIFF in float32
            output_path = os.path.join(output_folder, file_name.replace('.bin', '.tiff'))
            depth_image = Image.fromarray(depth_map, mode='F')  # Mode 'F' for 32-bit float
            depth_image.save(output_path, format='TIFF')
            
            print(f"Saved {output_path} as TIFF with float32 precision.")
        
        except ValueError:
            print(f"Failed to reshape {file_name}. The data may not match the specified dimensions.")
