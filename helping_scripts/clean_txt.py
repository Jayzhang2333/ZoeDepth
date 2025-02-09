from PIL import Image
import os

# Read the text file
file_path = 'train_test_inputs/flsea_red_sea/northeast_path_test_with_matched_features.txt'  # Replace with your file path
with open(file_path, 'r') as f:
    lines = f.readlines()

# List to hold valid lines
valid_lines = []

# Check each line
for line in lines:
    # Get the path to the image (assuming it is the first column)
    image_path = line.strip().split()[0]
    
    try:
        # Attempt to open the image
        with Image.open(image_path) as img:
            img.verify()  # Verify that it is not broken
        # If successful, add the line to the list of valid lines
        valid_lines.append(line)
    except (IOError, SyntaxError):
        # Handle image loading errors, print or log if needed
        print(f"Broken image detected and removed: {image_path}")

# Write the valid lines back to the file (overwriting it)
with open(file_path, 'w') as f:
    f.writelines(valid_lines)

print("Finished checking and removing broken images.")
