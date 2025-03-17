from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# # Assuming the depth map is stored in a file called 'depth_map.tif'
input_sparse_depth_fp = '/media/jay/Lexar/SeaThru/sea_thru/D1/D1/depth/depthT_S02951.tif'
depth_map_array = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32)
print(np.max(depth_map_array))
print(np.min(depth_map_array))
# depth_map_array = depth_map_array/256.0
print(np.shape(depth_map_array))
# Convert the image to a NumPy array


# depth_map = np.load('depth_map.npy')

# Display the depth map using matplotlib with the 'plasma' colormap
# plt.imshow(depth_map_array, cmap='viridis')
plt.imshow(depth_map_array)
# plt.colorbar()
# plt.title('Depth Map from TIF')
plt.show()

# mask = np.where((depth_map_array >= 0) & (depth_map_array <= 3), 1, 0)

# # Display the mask using matplotlib
# plt.imshow(mask, cmap='gray')
# plt.title('Mask for Depth Values Between 0 and 3 Meters')
# plt.show()