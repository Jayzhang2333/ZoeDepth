input_file = "train_test_inputs/void_150_combined_testing_list_random.txt"  # Replace with your file name
output_file = "train_test_inputs/void_150_combined_testing_list_random_filtered.txt"  # Replace with your desired output file name

# Open and filter the file
with open(input_file, 'r') as file:
    lines = file.readlines()

# Remove lines containing 'birthplace'
filtered_lines = [line for line in lines if 'birthplace' not in line]

# Save the cleaned lines to the output file
with open(output_file, 'w') as file:
    file.writelines(filtered_lines)

print(f"Rows containing 'birthplace' have been removed. Cleaned file saved as {output_file}.")
