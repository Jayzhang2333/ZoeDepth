from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Assuming the depth map is stored in a file called 'depth_map.tif'
depth_map_tif = Image.open('/media/jay/apple/FLSea_latest/archive/canyons/flatiron/flatiron/depth/16233051861828957_SeaErra_abs_depth.tif')

# Convert the image to a NumPy array
depth_map_array = np.array(depth_map_tif)

# Display the depth map using matplotlib with the 'plasma' colormap
plt.imshow(depth_map_array, cmap='plasma')
plt.colorbar()
plt.title('Depth Map from TIF')
plt.show()

# mask = np.where((depth_map_array >= 0) & (depth_map_array <= 3), 1, 0)

# # Display the mask using matplotlib
# plt.imshow(mask, cmap='gray')
# plt.title('Mask for Depth Values Between 0 and 3 Meters')
# plt.show()
