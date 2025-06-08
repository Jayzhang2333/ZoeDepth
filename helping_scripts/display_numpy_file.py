import numpy as np
import matplotlib.pyplot as plt

# Load the numpy file (change 'your_file.npy' to the actual file path)
file_path = '/media/jay/Marine/Tartanair/seasonsforest_winter/Easy/P000/depth_left/001108_left_depth.npy'  # Replace with the correct file path
data = np.load(file_path)
print(np.shape(data))

# Display the data using matplotlib
plt.imshow(data, cmap='inferno')
plt.colorbar()
plt.title('Numpy Data Visualization')
plt.show()
