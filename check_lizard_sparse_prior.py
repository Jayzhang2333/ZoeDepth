import numpy as np
from PIL import Image

# Read the text file, skipping the header and reading only the first 10 rows of data
data_file = '/media/jay/apple/uw_depth_lizard_data/features-saved/0422.txt'  # Replace with your file path
data = []
with open(data_file, 'r') as file:
    file.readline()  # Skip the header
    for i in range(10):  # Read only the first 10 rows after the header
        line = file.readline().strip()
        if line:
            row, col, extracted_depth = map(float, line.split())
            data.append((int(row), int(col), extracted_depth))

# Load the ground truth depth map from a TIFF image
ground_truth_file = '/media/jay/apple/uw_depth_lizard_data/depth_tiff/0422.tiff'  # Replace with your ground truth depth map file path
ground_truth_image = Image.open(ground_truth_file)
ground_truth_depth_map = np.array(ground_truth_image)

# Find and print the ground truth value and the difference for each data point
for row, col, extracted_depth in data:
    gt_depth_value = ground_truth_depth_map[row, col]
    depth_difference = gt_depth_value - extracted_depth
    depth_ratio = gt_depth_value / extracted_depth
    print(f"Data Point - Row: {row}, Column: {col}, Extracted Depth: {extracted_depth}, "
          f"Ground Truth Depth: {gt_depth_value}, Difference: {depth_difference}, Ratio: {depth_ratio}")
