import os

def remove_bathroom_lines(file_path):
    # Get the directory and filename
    directory, original_filename = os.path.split(file_path)
    
    # Create the new filename
    new_filename = f"removed_bathroom_{original_filename}"
    new_file_path = os.path.join(directory, new_filename)
    
    # Read the original file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Remove lines containing the word 'bathroom' (as a word or part of a word)
    filtered_lines = [line for line in lines if 'bathroom' not in line]
    
    # Write the filtered lines to the new file
    with open(new_file_path, 'w') as new_file:
        new_file.writelines(filtered_lines)
    
    print(f"New file saved as: {new_file_path}")

# Example usage:
remove_bathroom_lines('train_test_inputs/nyu_extract_test_sparse_depth.txt')
