import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

rgb_image_path = "/home/jay/shortcuts/datasets/nyu_depth_v2/sync/living_room_0069a/rgb_00000.jpg"
depth_map_path = "/home/jay/shortcuts/datasets/nyu_depth_v2/sync/living_room_0069a/sync_depth_00000.png"

# Read the RGB image
rgb_image = Image.open(rgb_image_path).convert("RGB")
rgb_image = np.array(rgb_image)  # Convert to NumPy array for easier handling

# Read the depth map (assuming it’s a single-channel image)
depth_map = Image.open(depth_map_path)
depth_map = np.array(depth_map)  # Convert to NumPy array

data = np.load('/home/jay/shortcuts/datasets/nyu_depth_v2/sync/living_room_0069a/new_probability_subsampled_200_rgb_00000_sift_depth.npy')
print(np.shape(data))

resized_channel_1 = cv2.resize(data[:, :, 0], (32, 24), interpolation=cv2.INTER_LINEAR)
resized_channel_2 = cv2.resize(data[:, :, 1], (31, 24), interpolation=cv2.INTER_LINEAR)

plt.figure(figsize=(12, 6))

# RGB Image
plt.subplot(3, 2, 1)
plt.imshow(rgb_image)
plt.title("RGB Image")

# Depth Map
plt.subplot(3, 2, 2)
plt.imshow(depth_map, cmap='plasma')
plt.colorbar()
plt.title("Depth Map")

# Resized Channel 1
plt.subplot(3, 2, 3)
plt.imshow(data[:,:,0], cmap='viridis')
plt.colorbar()
plt.title("original Channel 1")

# Resized Channel 2
plt.subplot(3, 2, 4)
plt.imshow(data[:,:,1], cmap='viridis')
plt.colorbar()
plt.title("original Channel 2")

# Resized Channel 1
plt.subplot(3, 2, 5)
plt.imshow(resized_channel_1, cmap='viridis')
plt.colorbar()
plt.title("Resized Channel 1")

# Resized Channel 2
plt.subplot(3, 2, 6)
plt.imshow(resized_channel_2, cmap='viridis')
plt.colorbar()
plt.title("Resized Channel 2")

plt.tight_layout()
plt.show()