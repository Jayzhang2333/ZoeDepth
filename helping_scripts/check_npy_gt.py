import numpy as np
import matplotlib.pyplot as plt

# Load the NumPy file
file_path = "/media/jay/Seagate/Tartanair/abandonedfactory/Easy/P000/depth_left/000000_left_depth.npy"  # Replace with your file path
data = np.load(file_path)

# Display the data
plt.imshow(data, cmap="viridis")  # Change cmap if needed
plt.colorbar()
plt.title("Loaded NumPy Array")
plt.show()
