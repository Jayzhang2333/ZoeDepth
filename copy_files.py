import os
import shutil
from tqdm import tqdm
def copy_files(file_list_path, new_base, new_list_path, src_base):
    new_lines = []

    # Resolve absolute paths for safety.
    src_base = os.path.abspath(src_base)
    new_base = os.path.abspath(new_base)
    
    # Safety check: Ensure new_base is not inside src_base.
    if os.path.commonpath([src_base, new_base]) == src_base:
        raise ValueError("new_base should not be a subdirectory of src_base.")

    # Read all rows from the file (each row contains three space-separated paths)
    with open(file_list_path, 'r') as f:
        rows = [line.strip() for line in f if line.strip()]

    # Count total files (each row contains three paths)
    total_files = sum(len(row.split()) for row in rows)
    progress_bar = tqdm(total=total_files, desc="Copying files", unit="file")

    for row in rows:
        # Split the row into individual file paths
        paths = row.split()
        new_paths = []
        for src_path in paths:
            src_path = os.path.abspath(src_path)
            
            # Check if src_path is under src_base.
            # os.path.commonpath returns the common prefix; if it isn’t src_base,
            # the file is not inside the source base.
            if os.path.commonpath([src_base, src_path]) != src_base:
                print(f"Warning: {src_path} is not under the source base {src_base}. Skipping.")
                progress_bar.update(1)
                continue

            # Compute the relative path from src_base
            rel_path = os.path.relpath(src_path, src_base)
            # Construct the destination path by joining new_base with the relative path
            dst_path = os.path.join(new_base, rel_path)
            dst_dir = os.path.dirname(dst_path)

            # Create the destination directory if it doesn't exist
            os.makedirs(dst_dir, exist_ok=True)

            # Copy the file (including metadata)
            shutil.copy2(src_path, dst_path)
            new_paths.append(dst_path)
            progress_bar.update(1)

        # Join the new paths with a space and add to our output list
        new_lines.append(" ".join(new_paths))
    
    progress_bar.close()

    # Write the new file paths to the output text file
    with open(new_list_path, 'w') as out_file:
        for line in new_lines:
            out_file.write(line + "\n")

    print(f"\nAll new paths saved to: {new_list_path}")

# Example usage:
if __name__ == "__main__":
    # Path to the text file that contains the original file paths (one per line)
    file_list = 'train_test_inputs/flease_testing/update_path_FLSea_combined_cleaned_output.txt'
    
    # The base directory of the source files
    source_base = '/media/jay/Data1'
    
    # The new base directory where files will be copied to
    new_base_dir = '/media/jay/Lexar'
    
    # The output text file that will list the new file paths
    output_file = 'train_test_inputs/flease_testing/ssd_update_path_FLSea_combined_cleaned_output.txt'
    
    copy_files(file_list, new_base_dir, output_file, source_base)
