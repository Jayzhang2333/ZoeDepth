import random

# Input and output file paths
input_file = "train_test_inputs/tartanair/tartanair_testing_swap_filtered_no_ocean_soulcity.txt"  # Replace with your file path
output_file = "train_test_inputs/tartanair/tartanair_testing_swap_filtered_no_ocean_soulcity_subsampled.txt"

# Read all lines from the input file
with open(input_file, "r") as f:
    lines = f.readlines()

# Calculate 10% of the total lines
num_lines_to_select = max(1, int(0.05 * len(lines)))

# Randomly select 10% of the lines
selected_lines = random.sample(lines, num_lines_to_select)

# Write the selected lines to the new file
with open(output_file, "w") as f:
    f.writelines(selected_lines)

print(f"Selected {num_lines_to_select} lines and saved them to {output_file}")
