import numpy as np
import matplotlib.pyplot as plt
# Define the path to your .bin file
file_path = '/media/jay/apple/uw_depth_lizard_data/depth/0528.bin'

def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

# Read the .bin file
# with open(file_path, 'rb') as f:
#     # Load data as a 1D array with the specified data type (e.g., float32)
#     depth_data = np.fromfile(f, dtype=np.float32)

# # Ignore the first 3 values
# print(depth_data[0])
# print(depth_data[1])
# print(depth_data[2])
# depth_data = depth_data[3:]

# Reshape the data to (height, width) with width=1355 and height=1002
# width, height = 1355, 1002
try:
    # depth_map = depth_data.reshape((height, width))
    depth_map = read_array(file_path)
    print("Depth map dimensions:", depth_map.shape)

    # Display the depth map using matplotlib
    plt.imshow(depth_map, cmap='viridis', origin='upper')  # 'upper' places the first pixel at the top-left
    plt.colorbar(label='Depth (meters)')
    plt.title('Depth Map')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.show()
except ValueError:
    print("Failed to reshape. The depth data is not complete")
    # print("Depth data shape:", np.shape(depth_data))
# Read the .bin file
# with open(file_path, 'rb') as f:
#     # Load data as a 1D array with the specified data type (e.g., float32)
#     depth_data = np.fromfile(f, dtype=np.float32)

# non_negative_count = np.sum(depth_data >= 0)
# print("Number of non-negative depth values:", non_negative_count)

# indices_less_or_equal_1e = np.where(depth_data <= 1e-16)[0]

# print("Number of data less tha 1e-16:", len(indices_less_or_equal_1e))

# indices_less_or_equal_0 = np.where(depth_data <= 0.1)[0]

# print("Number of data less tha 0.1:", len(indices_less_or_equal_0))

# print("Difference: ", len(indices_less_or_equal_0) -len(indices_less_or_equal_1e) )
# # Print the minimum and maximum values
# print(depth_data[0])
# print(depth_data[1])
# print(depth_data[2])
# print(depth_data[3])
# print(depth_data[4])
# print(depth_data[5])
# print(depth_data[6])
# print(depth_data[7])
# print(depth_data[8])
# print(depth_data[9])
# print(depth_data[10])
# print(depth_data[11])
# print(depth_data[-1])
# print(depth_data[-2])
# print(depth_data[-3])
# print(depth_data[-4])
# print(depth_data[-5])
# print(depth_data[-6])
# print(depth_data[-7])
# print(depth_data[-8])
# print(depth_data[-9])
# print(depth_data[-10])
# print(depth_data[-11])
# print(depth_data[-12])
# print(depth_data[-13])
# print(depth_data[-14])
# print(depth_data[-15])

# print("length of each depth map:", len(depth_data))
# # print("Shape of each depth map:", np.shape(depth_data))
# print("Minimum depth value:", np.min(depth_data))
# print("Maximum depth value:", np.max(depth_data))
