import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def get_depth_prior_from_features(features, height=240, width=320):
    """Takes lists of pixel indices and their respective depth probes and
    returns a dense depth prior parametrization using NumPy, displaying results
    using Matplotlib."""
    
    batch_size = features.shape[0]

    # depth prior maps
    prior_maps = np.empty((batch_size, height, width))

    # euclidean distance maps
    distance_maps = np.empty((batch_size, height, width))

    for i in range(batch_size):
        # use only entries with valid depth
        mask = features[i, :, 2] > 0.0

        if not np.any(mask):
            max_dist = np.sqrt(height ** 2 + width ** 2)
            prior_maps[i, ...] = 0.0
            distance_maps[i, ...] = max_dist
            print(f"WARNING: Img {i+1} has no valid features, using placeholders.")
            continue

        # get list of indices and depth values
        idcs_height = np.round(features[i, mask, 0]).astype(int)
        idcs_width = np.round(features[i, mask, 1]).astype(int)
        depth_values = features[i, mask, 2]

        # get distance maps for each feature
        sample_dist_maps = get_distance_maps(height, width, idcs_height, idcs_width)

        # find min and argmin
        dist_map_min = np.min(sample_dist_maps, axis=0)
        dist_argmin = np.argmin(sample_dist_maps, axis=0)

        # nearest neighbor prior map
        prior_map = depth_values[dist_argmin]

        # concat
        prior_maps[i, ...] = prior_map
        distance_maps[i, ...] = dist_map_min

        # Display results for each image
        plt.figure(figsize=(10, 5))

        # Display depth prior map
        plt.subplot(1, 2, 1)
        plt.title(f"Depth Prior Map {i+1}")
        plt.imshow(prior_map, cmap='jet')
        plt.colorbar()

        # Display probability map
        plt.subplot(1, 2, 2)
        probability_map = get_probability_maps(dist_map_min)
        plt.title(f"Probability Map {i+1}")
        plt.imshow(probability_map, cmap='jet')
        plt.colorbar()

        plt.show()

    # parametrization (concatenating depth map and probability map for each image)
    parametrization = np.stack([prior_maps, get_probability_maps(distance_maps)], axis=1)

    return parametrization

def load_features_from_csv(csv_file):
    """Loads features from a CSV file into a NumPy array."""
    df = pd.read_csv(csv_file)
    features = df[['row', 'col', 'depth']].to_numpy()
    return features

# Example usage:
csv_file = '/home/jay/shortcuts/datasets/nyu_depth_v2/sync/kitchen_0028b/rgb_00045_sift_depth.csv'  # Replace with the path to your CSV file
features = load_features_from_csv(csv_file)

# Reshape features into the required batch format (batch_size, num_features, 3)
# Since we're working with one image, we expand the dimensions to simulate a batch of size 1
features = features[np.newaxis, :, :]

# Process and visualize the depth prior
parametrization = get_depth_prior_from_features(features, height=480, width=640)
