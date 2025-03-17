import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Input and output file paths (modify these as needed)
input_txt = "train_test_inputs/flease_testing/ssd_update_path_FLSea_combined_cleaned_output.txt"
output_txt = "train_test_inputs/flease_testing/flsea_preserved_testing.txt"
# Enable interactive mode for matplotlib
plt.ion()

# Open the output file in write mode (it will be created if it doesn't exist)
with open(output_txt, "w") as out_file:
    # Open and process the input file line by line
    with open(input_txt, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # skip empty lines

            # Each row should contain three paths separated by spaces.
            # We only use the first two paths.
            paths = line.split()
            if len(paths) < 2:
                print("Skipping line (not enough paths):", line)
                continue

            image_path = paths[0]
            depth_map_path = paths[1]

            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                continue
            if not os.path.exists(depth_map_path):
                print(f"Depth map file not found: {depth_map_path}")
                continue

            try:
                image = Image.open(image_path)
                depth = Image.open(depth_map_path)
            except Exception as e:
                print(f"Error opening files: {e}")
                continue

            depth_array = np.array(depth)

            # Create a figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            ax1.imshow(image)
            ax1.set_title("Image")
            ax1.axis("off")

            im = ax2.imshow(depth_array, cmap="viridis")
            ax2.set_title("Depth Map")
            ax2.axis("off")
            fig.colorbar(im, ax=ax2)

            # Display the figure in non-blocking mode
            plt.show(block=False)
            plt.pause(0.1)  # give time for the figure to render

            # Prompt the user. The figure window will remain open until you answer.
            answer = input("Preserve this row? (y/n): ").strip().lower()
            plt.close(fig)  # close the figure after the answer

            if answer and answer[0] == 'y':
                out_file.write(line + "\n")
                print("Row preserved.\n")
            else:
                print("Row skipped.\n")