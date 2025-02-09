import pandas as pd

# Read the CSV file
file_path = 'train_test_inputs/nyu_extract_test_sparse_depth.txt'  # Replace with your file path
data = pd.read_csv(file_path)

# Randomly sample 1400 rows from the data
sampled_data = data.sample(n=190)

# Save the sampled data to a new CSV file
output_file_path = 'train_test_inputs/subsampled_nyu_extract_test_sparse_depth.txt'  # Replace with your desired output path
sampled_data.to_csv(output_file_path, index=False)

# Display the first few rows of the sampled data to the user
# import ace_tools as tools; tools.display_dataframe_to_user(name="Sampled Data", dataframe=sampled_data)
