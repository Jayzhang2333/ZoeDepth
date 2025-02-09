from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# # Assuming the depth map is stored in a file called 'depth_map.tif'
# input_sparse_depth_fp = '/home/jay/Downloads/void_release/void_150/data/copyroom4/ground_truth/1552625653.9864.png'
# depth_map_array = np.array(Image.open(input_sparse_depth_fp), dtype=np.uint16) / 256.0
# print(np.shape(depth_map_array))
# # Convert the image to a NumPy array


# # depth_map = np.load('depth_map.npy')

# # Display the depth map using matplotlib with the 'plasma' colormap
# plt.imshow(depth_map_array, cmap='inferno')
# plt.colorbar()
# plt.title('Depth Map from TIF')
# plt.show()

# mask = np.where((depth_map_array >= 0) & (depth_map_array <= 3), 1, 0)

# # Display the mask using matplotlib
# plt.imshow(mask, cmap='gray')
# plt.title('Mask for Depth Values Between 0 and 3 Meters')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

def read_pfm(file):
    with open(file, "rb") as f:
        header = f.readline().decode().rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise ValueError("Not a PFM file.")

        dim_line = f.readline().decode().rstrip()
        width, height = map(int, dim_line.split())

        scale = float(f.readline().decode().rstrip())
        endian = '<' if scale < 0 else '>'
        scale = abs(scale)

        data = np.fromfile(f, endian + 'f')
        data = np.reshape(data, (height, width, 3) if color else (height, width))
        data = np.flipud(data)
        return data

# Specify your PFM file path
file_path = "/home/jay/VI-Depth/output/sml_depth/16316026388705614.pfm"

# Read the PFM file
image_data = read_pfm(file_path)

# Display the image using matplotlib
plt.imshow(1.0/image_data, cmap='inferno')
plt.colorbar()
plt.title("PFM Image Display")
plt.show()

