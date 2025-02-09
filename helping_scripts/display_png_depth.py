from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# Assuming the depth map is stored in a file called 'depth_map.tif'
input_sparse_depth_fp = '/home/jay/Downloads/diode_dataset/train/outdoor/scene_00007/scan_00082/00007_00082_outdoor_000_020_rel_depth.png'
depth_map_array = np.array(Image.open(input_sparse_depth_fp)) 
print(np.shape(depth_map_array))
print(np.max(depth_map_array))
# Convert the image to a NumPy array


# depth_map = np.load('depth_map.npy')

# Display the depth map using matplotlib with the 'plasma' colormap
# plt.imshow(depth_map_array, cmap='inferno')
plt.imshow(depth_map_array)
# plt.colorbar()
plt.title('Depth Map Normlaised')
plt.show()

# plt.imshow(depth_map_array * 10, cmap='inferno')
# plt.colorbar()
# plt.title('Depth Map Scaleds')
# plt.show()

# mask = np.where((depth_map_array >= 0) & (depth_map_array <= 3), 1, 0)

# # Display the mask using matplotlib
# plt.imshow(mask, cmap='gray')
# plt.title('Mask for Depth Values Between 0 and 3 Meters')
# plt.show()