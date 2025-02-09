import numpy as np
import pandas as pd
import cv2
import os

# Path to your original CSV file
csv_file = '/home/jay/Downloads/diode_dataset/val_outdoor.csv'
df = pd.read_csv(csv_file, header=None)

# Base directory to replace './' with
base_directory = '/home/jay/Downloads/diode_dataset'

# List to store the new data with the additional features path column
new_data = []

for index, row in df.iterrows():
    # Replace './' with the base directory in each path
    image_path = row[0].replace('./', base_directory + '/')
    depth_map_path = row[1].replace('./', base_directory + '/')
    valid_mask_path = row[2].replace('./', base_directory + '/')

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image {image_path}")
        continue

    # Read the depth map
    if not os.path.exists(depth_map_path):
        print(f"Depth map file {depth_map_path} does not exist.")
        continue
    depth_map = np.load(depth_map_path)

    # Remove singleton dimension if depth_map has shape (H, W, 1)
    if depth_map.ndim == 3 and depth_map.shape[2] == 1:
        depth_map = depth_map.squeeze(axis=2)

    # Read the valid mask
    if not os.path.exists(valid_mask_path):
        print(f"Valid mask file {valid_mask_path} does not exist.")
        continue
    valid_mask = np.load(valid_mask_path).astype(bool)

    # Ensure that depth_map and valid_mask have the same shape
    if depth_map.shape != valid_mask.shape:
        print(f"Depth map and valid mask shapes do not match for image {image_path}")
        print(f"Depth map shape: {depth_map.shape}, Valid mask shape: {valid_mask.shape}")
        continue

    # Run SIFT feature extraction
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints = sift.detect(gray_image, None)

    # Sort keypoints by their response (strength)
    keypoints = sorted(keypoints, key=lambda x: -x.response)

    # Collect up to 200 valid keypoints
    valid_keypoints = []
    max_keypoints = 200
    h, w = depth_map.shape
    num_keypoints = len(keypoints)
    kp_index = 0

    while len(valid_keypoints) < max_keypoints and kp_index < num_keypoints:
        kp = keypoints[kp_index]
        x, y = kp.pt  # x and y are floats
        col, row_kp = int(round(x)), int(round(y))  # Convert to integers

        # Check if pixel coordinates are within image bounds
        if 0 <= row_kp < h and 0 <= col < w:
            # Check if depth is valid
            if valid_mask[row_kp, col]:
                depth = depth_map[row_kp, col]
                valid_keypoints.append((row_kp, col, depth))
        kp_index += 1

    # Check if the required number of keypoints was found
    if len(valid_keypoints) < max_keypoints:
        print(f"Image {image_path}: Only {len(valid_keypoints)} valid keypoints found. Skipping.")
        continue  # Skip this image and do not include it in the training data list

    print(f"Image {image_path}: Found {len(valid_keypoints)} valid keypoints")

    # Generate output filename based on the image name and directory
    image_dir = os.path.dirname(image_path)
    image_base = os.path.basename(image_path)
    image_name, image_ext = os.path.splitext(image_base)
    output_filename = f"{image_name}_features.txt"
    output_file = os.path.join(image_dir, output_filename)

    # Save to txt file with header 'row col depth'
    with open(output_file, 'w') as f:
        f.write('row col depth\n')
        for row_val, col_val, depth_val in valid_keypoints:
            f.write(f"{row_val} {col_val} {depth_val}\n")

    # Append the new row to the new_data list, updating paths to use the absolute paths
    new_row = [image_path, depth_map_path, valid_mask_path, output_file]
    new_data.append(new_row)

# Create a new DataFrame with the updated data
new_df = pd.DataFrame(new_data)

# Save the new training data to a text file with space-separated values
new_training_data_file = '/home/jay/Downloads/diode_dataset/diode_validating_data_with_features.txt'
new_df.to_csv(new_training_data_file, header=False, index=False, sep=' ')

print(f"\nNew training data file saved as '{new_training_data_file}'.")
