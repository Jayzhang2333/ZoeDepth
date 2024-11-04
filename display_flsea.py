from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Assuming the depth map is stored in a file called 'depth_map.tif'
depth_map_tif = Image.open('/media/jay/apple/uw_depth_lizard_data/depth_tiff/0422.tiff')

print(np.shape(depth_map_tif))
# Convert the image to a NumPy array
depth_map_array = np.array(depth_map_tif)

# depth_map = np.load('depth_map.npy')

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
