import os
import pandas as pd
from PIL import Image

# Function to check if a file is valid
def is_valid_file(file_path):
    if not os.path.exists(file_path):
        return False
    try:
        if file_path.endswith((".tiff", ".tif")):
            with Image.open(file_path) as img:
                img.verify()  # Check if the image is corrupted
        elif file_path.endswith(".csv"):
            pd.read_csv(file_path)  # Try reading the CSV file
        return True
    except Exception:
        return False

# Read the text file and process each line
input_file = "./train_test_inputs/flease_testing/FLSea_combined.txt"  # Change this to your input file path
output_file = "./train_test_inputs/flease_testing/FLSea_combined_cleaned_output.txt"

cleaned_lines = []
with open(input_file, "r") as f:
    for line in f:
        file_paths = line.strip().split()
        if all(is_valid_file(fp) for fp in file_paths):
            cleaned_lines.append(line)

# Write the cleaned lines to a new file
with open(output_file, "w") as f:
    f.writelines(cleaned_lines)

print(f"Cleaned file saved as {output_file}")
