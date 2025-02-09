from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def generate_feature_map_for_ga(feature_fp, original_height=480, original_width=640, new_height=384, new_width=512):
    # Check the file extension to determine the delimiter
    file_extension = os.path.splitext(feature_fp)[-1].lower()
    
    if file_extension == '.csv':
        df = pd.read_csv(feature_fp)
    elif file_extension == '.txt':
        df = pd.read_csv(feature_fp, delimiter=' ', header=0, names=['row', 'column', 'depth'])
    else:
        raise ValueError("Unsupported file format. Only CSV and TXT files are supported.")
    
    # Initialize a blank depth map for the new image size with zeros
    sparse_depth_map = np.full((new_height, new_width), 0.0, dtype=np.float32)

    # Calculate scaling factors
    scale_y = new_height / original_height
    scale_x = new_width / original_width

    # Iterate through the dataframe and populate the depth map with scaled coordinates
    for index, row in df.iterrows():
        # Scale pixel coordinates to new image size
        pixel_row = int(row['row'] * scale_y)
        pixel_col = int(row.get('column', row.get('col', 0)) * scale_x)
        depth_value = float(row['depth'])

        # Ensure the scaled coordinates are within the bounds of the new image size
        if 0 <= pixel_row < new_height and 0 <= pixel_col < new_width:
            sparse_depth_map[pixel_row, pixel_col] = depth_value

    sparse_depth_map = sparse_depth_map[..., np.newaxis]
    # print(np.shape(sparse_depth_map))
    return sparse_depth_map

feature_path = '/media/jay/apple/FLSea_latest/archive/canyons/u_canyon/u_canyon/imgs/matched_features/16233043763678300_features.csv'
sparse_feature_map = generate_feature_map_for_ga(feature_path, original_height=240, original_width=320, new_height=288, new_width=384)

plt.imshow(sparse_feature_map, cmap='viridis')
# plt.imshow(depth_map_array)
plt.colorbar()
# plt.title('Depth Map from TIF')
plt.show()