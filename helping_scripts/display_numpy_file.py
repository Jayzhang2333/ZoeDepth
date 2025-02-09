import numpy as np
import matplotlib.pyplot as plt

# Load the numpy file (change 'your_file.npy' to the actual file path)
file_path = '/media/jay/Seagate/Tartanair/ocean/Easy/P001/seg_left/000462_left_seg.npy'  # Replace with the correct file path
data = np.load(file_path)
print(np.shape(data))

# Display the data using matplotlib
plt.imshow(data, cmap='inferno')
plt.colorbar()
plt.title('Numpy Data Visualization')
plt.show()
