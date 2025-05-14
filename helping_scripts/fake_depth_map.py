import numpy as np
from PIL import Image

# Define the resolution
width, height = 709, 518

# Create a NumPy array with all values set to 5.
# Using uint8 is common for depth maps with small values, but you can adjust the dtype if needed.
depth_map = np.full((height, width), 5, dtype=np.uint8)

# Convert the NumPy array to a PIL Image object.
depth_image = Image.fromarray(depth_map)

# Save the image as a TIFF file to the specified path.
output_path = "fake_depth_map.tiff"
depth_image.save(output_path)

print(f"Fake depth map saved as '{output_path}' with resolution {width}x{height}.")