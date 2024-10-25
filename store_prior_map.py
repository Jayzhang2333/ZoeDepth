import csv
import numpy as np

def get_distance_maps(height, width, idcs_height, idcs_width):
    """Generates distance maps from feature points to every pixel."""
    num_features = len(idcs_height)
    dist_maps = np.empty((num_features, height, width))
    
    for idx in range(num_features):
        y, x = idcs_height[idx], idcs_width[idx]
        y_grid, x_grid = np.ogrid[:height, :width]
        dist_maps[idx] = np.sqrt((y_grid - y) ** 2 + (x_grid - x) ** 2)
    
    return dist_maps

def get_probability_maps(distance_maps):
    """Convert pixel distance to probability."""
    max_dist = np.sqrt(distance_maps.shape[-2] ** 2 + distance_maps.shape[-1] ** 2)
    probabilities = np.exp(-distance_maps / max_dist)
    return probabilities

def get_depth_prior_from_features(features, prior_channels, height=240, width=320):
    """Takes lists of pixel indices and their respective depth probes and
    returns a dense depth prior parametrization using NumPy, displaying results
    using Matplotlib."""
    
    # batch_size = features.shape[0]

    # depth prior maps
    prior_maps = np.empty((height, width))

    # euclidean distance maps
    distance_maps = np.empty((height, width))

    # for i in range(batch_size):
    # use only entries with valid depth
    mask = features[ :, 2] > 0.0

    if not np.any(mask):
        max_dist = np.sqrt(height ** 2 + width ** 2)
        prior_maps[ ...] = 0.0
        distance_maps[ ...] = max_dist
        print(f"WARNING: Img has no valid features, using placeholders.")
        # continue

    # get list of indices and depth values
    idcs_height = np.round(features[ mask, 0]).astype(int)
    idcs_width = np.round(features[ mask, 1]).astype(int)
    depth_values = features[ mask, 2]

    # get distance maps for each feature
    sample_dist_maps = get_distance_maps(height, width, idcs_height, idcs_width)

    # find min and argmin
    dist_map_min = np.min(sample_dist_maps, axis=0)
    dist_argmin = np.argmin(sample_dist_maps, axis=0)

    # nearest neighbor prior map
    prior_map = depth_values[dist_argmin]

    # concat
    prior_maps[...] = prior_map
    distance_maps[...] = dist_map_min

    # # Display results for each image
    # plt.figure(figsize=(10, 5))

    # # Display depth prior map
    # plt.subplot(1, 2, 1)
    # plt.title(f"Depth Prior Map {i+1}")
    # plt.imshow(prior_map, cmap='jet')
    # plt.colorbar()

    # # Display probability map
    # plt.subplot(1, 2, 2)
    # probability_map = get_probability_maps(dist_map_min)
    # plt.title(f"Probability Map {i+1}")
    # plt.imshow(probability_map, cmap='jet')
    # plt.colorbar()

    # plt.show()

    # parametrization (concatenating depth map and probability map for each image)
    if prior_channels<=1:
        return np.expand_dims(prior_map, axis=2)
    else:
        parametrization = np.stack([prior_maps, get_probability_maps(distance_maps)], axis=2)
        # print(parametrization.shape)

        return parametrization

def load_features_from_csv(csv_file):
    """Loads features from a CSV file into a NumPy array."""
    features = []

    with open(csv_file, mode='r') as file:
        csv_reader = csv.DictReader(file)
        col_name = 'col' if 'col' in csv_reader.fieldnames else 'column'

        # Extract the relevant columns ('row', col_name, 'depth')
        for row in csv_reader:
            features.append([float(row['row']), float(row[col_name]), float(row['depth'])])

    return np.array(features)

def generate_sparse_feature_map(featrue_path, prior_channels, height, width):
    # print(featrue_path)
    features = load_features_from_csv(featrue_path)

    # Reshape features into the required batch format (batch_size, num_features, 3)
    # Since we're working with one image, we expand the dimensions to simulate a batch of size 1
    # i think the np.newaxis should be repalced with batch size
    # features = features[np.newaxis, :, :]
    # print(features.shape)

    # Process and visualize the depth prior
    # parametrization = get_depth_prior_from_features(features,prior_channels, height=height, width=width)
    parametrization = get_depth_prior_from_features(features,prior_channels, height=240, width=320)

    return parametrization

import matplotlib.pyplot as plt
def display_image(title, img):
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

import os

def process_txt_file(txt_file, base_directory, prior_channels, height, width):
    """Reads the txt file, extracts CSV file paths, calls generate_sparse_feature_map, saves result as a .npy file, and writes a new txt file with .npy paths."""
    
    # Get the directory and filename of the original txt file
    txt_directory = os.path.dirname(txt_file)
    txt_filename = os.path.basename(txt_file)
    
    # Create the new txt file path by prepending 'NPY_' to the original filename
    new_txt_file = os.path.join(txt_directory, f"NPY_{txt_filename}")
    
    # Open the original txt file for reading and the new txt file for writing
    with open(txt_file, mode='r') as file, open(new_txt_file, mode='w') as new_file:
        # Loop through each line in the txt file
        for line in file:
            # Split the line into columns (assuming it's space or tab delimited)
            columns = line.split()
            
            # The fourth column (index 3) contains the CSV file directory
            csv_relative_path = columns[-1]
            
            # Prepend the base directory to the relative CSV path
            csv_full_path = os.path.join(base_directory, csv_relative_path)
            
            # Call the function with the constructed CSV file path
            feature_map = generate_sparse_feature_map(csv_full_path, prior_channels, height, width)
            
            # Determine the directory and file name for saving the .npy file
            csv_directory = os.path.dirname(csv_full_path)
            csv_filename = os.path.basename(csv_full_path).replace('.csv', '.npy')
            npy_save_path = os.path.join(csv_directory, csv_filename)
            
            # Save the feature map as a .npy file
            np.save(npy_save_path, feature_map)
            print(f"Saved NumPy array to: {npy_save_path}")
            
            # Replace the CSV path in the line with the new .npy file path
            columns[-1] = os.path.relpath(npy_save_path, base_directory)
            
            # Write the modified line to the new text file
            new_file.write(' '.join(columns) + '\n')
    
    print(f"Saved new txt file with .npy paths to: {new_txt_file}")

# Example usage
txt_file_path = './train_test_inputs/flsea_test_with_features.txt'
base_dir = ''
prior_channels = 2
height = 240
width = 320

process_txt_file(txt_file_path, base_dir, prior_channels, height, width)

