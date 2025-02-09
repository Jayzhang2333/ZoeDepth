import os
import random

# Function to combine all TXT files and shuffle rows
def combine_and_shuffle_txt(folder_path, output_txt_path):
    try:
        combined_rows = []

        # Iterate through all files in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # Check if the file is a TXT file
            if os.path.isfile(file_path) and file_name.endswith('.txt'):
                with open(file_path, 'r') as file:
                    rows = file.readlines()
                    combined_rows.extend([row.strip() for row in rows if row.strip()])

        # Shuffle the rows
        random.shuffle(combined_rows)

        # Write the shuffled rows to the output file
        with open(output_txt_path, 'w') as output_file:
            output_file.write("\n".join(combined_rows))

        print(f"Combined and shuffled file saved to: {output_txt_path}")

    except Exception as e:
        print(f"Error: {e}")

# Example usage
folder_path = "/home/jay/ZoeDepth/train_test_inputs/flsea_training"
output_txt_path = "/home/jay/ZoeDepth/train_test_inputs/flsea_training/flsea_training_combined.txt"
combine_and_shuffle_txt(folder_path, output_txt_path)
