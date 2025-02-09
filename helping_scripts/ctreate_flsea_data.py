import pandas as pd
import os
from PIL import Image

# Function to process the CSV file
def process_csv_to_txt(input_csv_path, output_txt_path):
    try:
        # Read the CSV file
        df = pd.read_csv(input_csv_path, header=None)

        # Initialize an empty list for valid rows
        valid_rows = []

        # Iterate through each row
        for row in df.itertuples(index=False):
            row_valid = True
            updated_row = []

            for path in row:
                # Replace '/home/auv/FLSea' with '/media/jay/apple/FLSea_latest'
                new_path = path.replace('/home/auv/FLSea', '/media/jay/apple/FLSea_latest')

                # Check if the file exists and is loadable
                if new_path.endswith(('.tiff', '.tif', '.csv')) and os.path.exists(new_path):
                    try:
                        if new_path.endswith(('.tiff', '.tif')):
                            # Try loading TIFF/TIF files
                            Image.open(new_path).verify()
                        elif new_path.endswith('.csv'):
                            # Try loading CSV files
                            pd.read_csv(new_path)
                    except Exception:
                        row_valid = False
                        break
                else:
                    row_valid = False
                    break

                # Append the valid path to the updated row
                updated_row.append(new_path)

            # Add the row to valid_rows if all paths are valid
            if row_valid:
                valid_rows.append(" ".join(updated_row))

        # Write the valid rows to the output txt file
        with open(output_txt_path, 'w') as output_file:
            output_file.write("\n".join(valid_rows))

        print(f"Processed file saved to: {output_txt_path}")

    except Exception as e:
        print(f"Error: {e}")

# Example usage
input_csv_path = "/media/jay/apple/FLSea_latest/archive/canyons/tiny_canyon/tiny_canyon/imgs/dataset_with_matched_features.csv"
output_txt_path = "./train_test_inputs/flease_canyon/tiny_canyon_with_features.txt"
process_csv_to_txt(input_csv_path, output_txt_path)
