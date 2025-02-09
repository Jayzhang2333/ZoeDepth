import pandas as pd
import os

# Load data from the three text files
images_list_path = '/home/jay/Downloads/void_release/void_150/test_image.txt'
ground_truth_list_path = '/home/jay/Downloads/void_release/void_150/test_ground_truth.txt'
sparse_prior_list_path = '/home/jay/Downloads/void_release/void_150/test_sparse_depth.txt'
base_path = '/home/jay/Downloads/void_release'
# Read the text files
images_list = pd.read_csv(images_list_path, header=None)
ground_truth_list = pd.read_csv(ground_truth_list_path, header=None)
sparse_prior_list = pd.read_csv(sparse_prior_list_path, header=None)

# Concatenate the three lists into one DataFrame
combined_df = pd.concat([images_list, ground_truth_list, sparse_prior_list], axis=1)

# Rename columns for clarity
combined_df.columns = ['Image Path', 'Ground Truth Path', 'Sparse Prior Path']

# base_path = '/media/jay/External/void-dataset/void_release'

# Add the base path in front of each file path
combined_df['Image Path'] = base_path + '/' + combined_df['Image Path'].astype(str)
combined_df['Ground Truth Path'] = base_path + '/' + combined_df['Ground Truth Path'].astype(str)
combined_df['Sparse Prior Path'] = base_path + '/' + combined_df['Sparse Prior Path'].astype(str)

filtered_df = combined_df[
    ~combined_df['Image Path'].str.contains('birthplace_of_internet') &
    ~combined_df['Ground Truth Path'].str.contains('birthplace_of_internet') &
    ~combined_df['Sparse Prior Path'].str.contains('birthplace_of_internet')
]

filtered_df = combined_df[
    ~combined_df['Image Path'].str.contains('model') &
    ~combined_df['Ground Truth Path'].str.contains('model') &
    ~combined_df['Sparse Prior Path'].str.contains('model')
]

filtered_df = combined_df[
    ~combined_df['Image Path'].str.contains('corner0') &
    ~combined_df['Ground Truth Path'].str.contains('corner0') &
    ~combined_df['Sparse Prior Path'].str.contains('corner0')
]

filtered_df = combined_df[
    ~combined_df['Image Path'].str.contains('corner1') &
    ~combined_df['Ground Truth Path'].str.contains('corner1') &
    ~combined_df['Sparse Prior Path'].str.contains('corner1')
]

filtered_df = combined_df[
    ~combined_df['Image Path'].str.contains('stairs') &
    ~combined_df['Ground Truth Path'].str.contains('stairs') &
    ~combined_df['Sparse Prior Path'].str.contains('stairs')
]

shuffled_df = filtered_df.sample(frac=1).reset_index(drop=True)
# Save the combined DataFrame to a new text file
output_path = '/home/jay/Downloads/void_release/void_150/void_150_combined_testing_list_random.txt'
shuffled_df.to_csv(output_path, index=False, header=False, sep=' ')

shuffled_df.head()  # Displaying first few rows for verification



