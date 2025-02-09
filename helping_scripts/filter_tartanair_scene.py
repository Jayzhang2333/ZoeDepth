import os

def remove_ocean_soulcity(file_path):
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Filter out lines containing 'ocean' or 'soulcity'
    filtered_lines = [line for line in lines if 'ocean' not in line and 'soulcity' not in line]

    # Create a new filename
    base_name, ext = os.path.splitext(file_path)
    new_file_path = f"{base_name}_no_ocean_soulcity{ext}"

    # Write the filtered lines to the new file
    with open(new_file_path, 'w') as f:
        f.writelines(filtered_lines)

    print(f"Filtered file saved as: {new_file_path}")

# Example usage
file_path = "train_test_inputs/tartanair/tartanair_testing_swap_filtered.txt"  # Replace with your actual file path
remove_ocean_soulcity(file_path)
