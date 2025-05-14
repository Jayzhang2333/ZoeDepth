import os
import pandas as pd

def subsample_data_file(data_file_path):
    """
    Reads a data file with a header 'row column depth' (space-separated),
    randomly subsamples half of the data rows, and writes the subsampled
    data to a new file using the same space-separated format.
    Returns the new file path.
    """
    # Read the file using whitespace as the delimiter
    df = pd.read_csv(data_file_path, delim_whitespace=True)
    
    # Randomly subsample half of the rows
    subsampled_df = df.sample(frac=0.5)
    
    # Create new file name by adding '_subsampled' before the extension
    dir_name, base_name = os.path.split(data_file_path)
    name, ext = os.path.splitext(base_name)
    new_file_name = f"{name}_subsampled{ext}"
    new_file_path = os.path.join(dir_name, new_file_name)
    
    # Write the subsampled data to the new file using space as the separator
    subsampled_df.to_csv(new_file_path, sep=' ', index=False)
    
    return new_file_path

def update_main_file(main_file_path, updated_main_file_path):
    """
    Reads the main file which contains three file paths per line.
    For each line, it processes the third file path by subsampling its data rows.
    Then, it updates the line with the new file path and writes all lines
    to a new main file.
    """
    updated_lines = []
    
    with open(main_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            # Assuming file paths are whitespace-separated
            file_paths = line.split()
            if len(file_paths) < 3:
                print("Skipping line (not enough file paths):", line)
                continue
            
            original_data_file = file_paths[2]
            # Subsample the third file and get the new file path
            new_data_file = subsample_data_file(original_data_file)
            # Update the third file path in the line
            file_paths[2] = new_data_file
            updated_lines.append(" ".join(file_paths))
    
    # Write the updated lines to a new file
    with open(updated_main_file_path, 'w') as f:
        for updated_line in updated_lines:
            f.write(updated_line + "\n")
    
    print(f"Updated main file saved to: {updated_main_file_path}")

if __name__ == "__main__":
    # Specify your main file that contains the three file paths per line
    main_file_path = "./train_test_inputs/lizard/lizard_image_depth_paths.txt"  # change this to your actual file path
    
    # Specify the output path for the updated main file (with updated data file paths)
    updated_main_file_path = "./train_test_inputs/lizard/lizard_image_depth_paths_subsampled_depth.txt"
    
    update_main_file(main_file_path, updated_main_file_path)
