import os
from PIL import Image

# Specify your input and output text file names
input_txt = '/media/jay/Lexar/SeaThru/sea_thru/sub_sampled_image_depth_sparse_paths.txt'
output_txt = '/media/jay/Lexar/SeaThru/sea_thru/sub_sampled_image_depth_sparse_paths_downsampled.txt.txt'

with open(input_txt, 'r') as fin, open(output_txt, 'w') as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        # Each line has three paths separated by space:
        rgb_path, depth_path, third_path = line.split()

        # --- Process the RGB image ---
        # Open the image
        rgb_img = Image.open(rgb_path)
        # Compute new dimensions (downsample by a factor of 4)
        new_rgb_size = (rgb_img.width // 4, rgb_img.height // 4)
        # Resize the image using high-quality downsampling
        rgb_downsampled = rgb_img.resize(new_rgb_size, resample=Image.ANTIALIAS)
        # Build the new filename by adding "_downsampled" before the extension
        rgb_dir, rgb_file = os.path.split(rgb_path)
        rgb_name, rgb_ext = os.path.splitext(rgb_file)
        new_rgb_name = f"{rgb_name}_downsampled{rgb_ext}"
        new_rgb_path = os.path.join(rgb_dir, new_rgb_name)
        # Save the downsampled image in the same folder
        rgb_downsampled.save(new_rgb_path)

        # --- Process the depth map (TIF) ---
        # Open the depth map image
        depth_img = Image.open(depth_path)
        # Compute new dimensions
        new_depth_size = (depth_img.width // 4, depth_img.height // 4)
        # Resize the depth map
        depth_downsampled = depth_img.resize(new_depth_size, resample=Image.ANTIALIAS)
        # Build the new filename for the depth image
        depth_dir, depth_file = os.path.split(depth_path)
        depth_name, depth_ext = os.path.splitext(depth_file)
        new_depth_name = f"{depth_name}_downsampled{depth_ext}"
        new_depth_path = os.path.join(depth_dir, new_depth_name)
        # Save the downsampled depth map
        depth_downsampled.save(new_depth_path)

        # --- Write the new paths to the output text file ---
        # The line contains: new_rgb_path, new_depth_path, and the third path
        fout.write(f"{new_rgb_path} {new_depth_path} {third_path}\n")
