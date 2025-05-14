import os
import random

def subsample_feature_file(feature_file_path, num_points=50):
    """
    Reads a feature file, randomly samples up to num_points points if there are more than num_points,
    and writes the (subsampled) points to a new file in the same directory with a prefix added.
    Returns the new file path.
    """
    try:
        with open(feature_file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {feature_file_path}: {e}")
        return feature_file_path  # fallback: return original path

    if not lines:
        print(f"No data in {feature_file_path}")
        return feature_file_path

    # Assume first line is header, remaining lines are data points
    header = lines[0].strip()
    data_lines = [line for line in lines[1:] if line.strip()]  # skip empty lines

    # If there are more than num_points, sample randomly; otherwise, keep all
    if len(data_lines) > num_points:
        sampled_lines = random.sample(data_lines, num_points)
    else:
        sampled_lines = data_lines

    # Create new filename by prefixing "subsampled_200_" to the original filename
    dir_name = os.path.dirname(feature_file_path)
    base_name = os.path.basename(feature_file_path)
    new_base_name = f"subsampled_50_{base_name}"
    new_file_path = os.path.join(dir_name, new_base_name)

    # Write the header and the sampled points to the new file
    try:
        with open(new_file_path, 'w') as f:
            f.write(header + "\n")
            for line in sampled_lines:
                f.write(line)
    except Exception as e:
        print(f"Error writing to {new_file_path}: {e}")
        return feature_file_path

    return new_file_path

def create_new_dataset_file(old_dataset_path, new_dataset_path, num_points=50):
    """
    Reads the original dataset file (each line with three space-separated paths),
    subsamples the feature file (third path) if needed,
    and writes a new dataset file with the updated third path.
    """
    new_lines = []
    try:
        with open(old_dataset_path, 'r') as dataset_file:
            for line in dataset_file:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue  # Skip malformed lines
                path1, path2, feature_path = parts[0], parts[1], parts[2]
                
                # Subsample the feature file (if needed)
                new_feature_path = subsample_feature_file(feature_path, num_points)
                
                # Construct new line with first two paths unchanged and third path updated
                new_line = f"{path1} {path2} {new_feature_path}\n"
                new_lines.append(new_line)
    except Exception as e:
        print(f"Error processing dataset file {old_dataset_path}: {e}")
        return

    # Write the new dataset file with updated feature file paths
    try:
        with open(new_dataset_path, 'w') as new_file:
            new_file.writelines(new_lines)
    except Exception as e:
        print(f"Error writing new dataset file {new_dataset_path}: {e}")

if __name__ == '__main__':
    # Example usage: specify your original dataset file and the desired new dataset filename.
    old_dataset_file = "./train_test_inputs/flease_testing/flsea_preserved_testing.txt"         # replace with your dataset file path
    new_dataset_file = "./train_test_inputs/flease_testing/subsample_50_flsea_preserved_testing.txt"  # output dataset file name

    create_new_dataset_file(old_dataset_file, new_dataset_file)
    print(f"New dataset file created: {new_dataset_file}")
