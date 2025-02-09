import csv

# Specify the input CSV file and output TXT file
input_csv_file = '/media/jay/apple/FLSea_latest/archive/red_sea/big_dice_loop/big_dice_loop/imgs/dataset_with_matched_features.csv'
output_txt_file = '/media/jay/apple/FLSea_latest/archive/red_sea/big_dice_loop/big_dice_loop/imgs/big_dice_loop_test_with_matched_features.txt'

# Open and read the CSV file, and write to the TXT file
with open(input_csv_file, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    with open(output_txt_file, 'w') as txt_file:
        for row in csv_reader:
            # Join the row elements with a space and write to the TXT file
            txt_file.write(' '.join(row) + '\n')

print(f"Converted {input_csv_file} to {output_txt_file} with space-separated values.")
