input_file = '/media/jay/Lexar/SeaThru/sea_thru/sub_sampled_image_depth_sparse_paths_downsampled.txt.txt'

# Read all lines from the file
with open(input_file, 'r') as f:
    lines = f.readlines()

# Filter out any line that contains "D5" or "D4"
filtered_lines = [line for line in lines if 'D5' not in line and 'D4' not in line]

# Write the filtered lines back to the same file
with open(input_file, 'w') as f:
    f.writelines(filtered_lines)