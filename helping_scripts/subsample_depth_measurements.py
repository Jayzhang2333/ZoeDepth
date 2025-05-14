import os
import pandas as pd
import argparse

def subsample_csv(csv_path, num_points):
    """
    Reads a CSV file and returns a subsampled DataFrame.
    If the number of points in the CSV is less than num_points,
    the original DataFrame is returned.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading '{csv_path}': {e}")
        return None

    if len(df) > num_points:
        # Randomly sample num_points rows; fix random_state for reproducibility.
        subsampled_df = df.sample(n=num_points, random_state=42)
    else:
        # Not enough points to sample; return the original DataFrame.
        subsampled_df = df

    return subsampled_df

def generate_subsampled_filename(csv_path, num_points):
    """
    Generates a new filename for the subsampled CSV.
    For example, if csv_path is 'data.csv' and num_points is 50,
    the new name becomes 'data_subsampled_50.csv'.
    """
    base, ext = os.path.splitext(csv_path)
    new_filename = f"{base}_subsampled_{num_points}{ext}"
    return new_filename

def process_data_txt(data_txt_path, sample_points=[10, 50, 100, 200]):
    """
    Processes the input text file containing data paths.
    For each line, it subsamples the CSV file (third path) for each sample size.
    Then, for each sample size, a new text file is generated with the updated CSV path.
    """
    # Read lines from the original data paths txt file.
    with open(data_txt_path, "r") as f:
        lines = f.read().splitlines()

    # Create a dictionary to store new file entries for each sample size.
    data_dict = {s: [] for s in sample_points}

    for line in lines:
        # Skip empty lines.
        if not line.strip():
            continue

        # Expecting each line in the format: <path1> <path2> <csv_path>
        parts = line.strip().split()
        if len(parts) < 3:
            print(f"Skipping line with insufficient parts: {line}")
            continue

        path1, path2, csv_path = parts[0], parts[1], parts[2]

        for s in sample_points:
            df_subsampled = subsample_csv(csv_path, s)
            if df_subsampled is None:
                # Skip processing this line for sample size `s` if reading fails.
                continue

            new_csv_path = generate_subsampled_filename(csv_path, s)
            try:
                # Save the subsampled CSV.
                df_subsampled.to_csv(new_csv_path, index=False)
                print(f"Saved subsampled CSV: {new_csv_path}")
            except Exception as e:
                print(f"Error saving subsampled CSV '{new_csv_path}': {e}")
                continue

            # Append the modified line to the data dictionary.
            # This new line uses the original path1, path2, and the new CSV path.
            new_line = f"{path1} {path2} {new_csv_path}"
            data_dict[s].append(new_line)

    # For each sample size, write a new text file that includes the updated CSV paths.
    for s in sample_points:
        base_txt, ext_txt = os.path.splitext(data_txt_path)
        new_txt_filename = f"{base_txt}_subsampled_{s}{ext_txt}"
        try:
            with open(new_txt_filename, "w") as f:
                for new_line in data_dict[s]:
                    f.write(new_line + "\n")
            print(f"Written {new_txt_filename}")
        except Exception as e:
            print(f"Error writing file '{new_txt_filename}': {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Subsample CSV files referenced in a data paths text file."
    )
    parser.add_argument(
        "data_txt", help="Path to the txt file containing data paths (each line: <path1> <path2> <csv_path>)."
    )
    args = parser.parse_args()

    process_data_txt(args.data_txt)

if __name__ == '__main__':
    main()
