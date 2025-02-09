import os
import random
from tqdm import tqdm  # Progress bar

# Define file paths
input_path_file = "/media/jay/Seagate/Tartanair/tartanair_training_swap.txt"  # Change to your actual path
output_path_file = "/media/jay/Seagate/Tartanair/tartanair_training_swap_filtered.txt"  # Output path for updated paths

# Read input path file
with open(input_path_file, "r") as f:
    lines = f.readlines()

updated_lines = []

# Initialize progress bar
with tqdm(total=len(lines), desc="Processing files", unit="file") as pbar:
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 3:
            pbar.update(1)
            continue  # Skip if not properly formatted

        img_path, depth_img_path, txt_path = parts

        if not os.path.exists(txt_path):
            pbar.update(1)
            continue  # Skip if the file does not exist

        # Read depth data file
        with open(txt_path, "r") as f:
            depth_data = f.readlines()

        # Skip the header
        data_points = [line.strip().split() for line in depth_data[1:]]

        if len(data_points) <= 200:
            pbar.update(1)
            continue  # Skip processing if fewer than 200 points

        # Randomly select 200 points
        selected_points = random.sample(data_points, 200)

        # Filter out points with depth > 50
        filtered_points = [point for point in selected_points if float(point[2]) <= 50]

        if len(filtered_points) >= 150:
            # Write updated depth file
            updated_txt_path = txt_path.replace(".txt", "_filtered.txt")
            with open(updated_txt_path, "w") as f:
                f.write("row column depth\n")  # Write header
                for point in filtered_points:
                    f.write(" ".join(point) + "\n")

            # Store the updated path file reference
            updated_lines.append(f"{img_path} {depth_img_path} {updated_txt_path}\n")

        # Update progress bar
        pbar.update(1)

# Save updated path file
with open(output_path_file, "w") as f:
    f.writelines(updated_lines)

print("Processing completed. Updated path file saved as:", output_path_file)
