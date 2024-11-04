import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the depth estimation from a numpy file
depth_estimation = np.load("/home/jay/shortcuts/datasets/nyu_depth_v2/sync/kitchen_0028b/new_probability_rgb_00045_sift_depth.npy")[:,:,1]
depth_estimation_1 = np.load("/home/jay/shortcuts/datasets/nyu_depth_v2/sync/kitchen_0028b/subsampled_200_rgb_00045_sift_depth.npy")[:,:,1]

# Load the ground truth depth map as a grayscale image
# ground_truth_depth = Image.open("/home/jay/shortcuts/datasets/nyu_depth_v2/sync/basement_0001a/sync_depth_00004.png")
# ground_truth_depth = np.asarray(ground_truth_depth, dtype=np.float32)
# ground_truth_depth = np.array(ground_truth_depth)/1000.0  # Normalize if necessary

# Create a mask where ground truth depth is greater than 0
# mask = ground_truth_depth > 0

# Calculate RMSE
# rmse = np.sqrt(np.mean((depth_estimation[mask] - ground_truth_depth[mask]) ** 2))

# print("RMSE:", rmse)

plt.figure(figsize=(12, 6))

# RGB Image
plt.subplot(1, 2, 1)
plt.imshow(depth_estimation, cmap='plasma')
plt.colorbar()
plt.title("Depth Image")

# Depth Map
plt.subplot(1, 2, 2)
plt.imshow(depth_estimation_1, cmap='plasma')
plt.colorbar()
plt.title("Depth Map")

plt.tight_layout()
plt.show()