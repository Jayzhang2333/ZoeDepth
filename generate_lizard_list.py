import os

# Paths to the folders
depth_folder = '/media/jay/apple/uw_depth_lizard_data/depth_tiff'
rgb_folder = '/media/jay/apple/uw_depth_lizard_data/images'
feature_folder = '/media/jay/apple/uw_depth_lizard_data/features-saved'
output_txt = '/media/jay/apple/uw_depth_lizard_data/image_depth_paths.txt'

# Open the text file to write
with open(output_txt, 'w') as f:
    # Loop through each file in the depth folder
    for file_name in os.listdir(depth_folder):
        if file_name.endswith('.tiff'):
            # Construct paths
            depth_path = os.path.join(depth_folder, file_name)
            rgb_path = os.path.join(rgb_folder, file_name.replace('.tiff', '.png'))
            feature_path = os.path.join(feature_folder, file_name.replace('.tiff', '.txt'))
            
            # Write the RGB and depth paths to the text file
            f.write(f"{rgb_path} {depth_path} {feature_path}\n")

print(f"Path file saved as {output_txt}")
