# Open the input file in read mode and output file in write mode
with open('train_test_inputs/tartanair/tartanair_testing_swap_filtered.txt', 'r') as infile, open('train_test_inputs/tartanair/tartanair_ocean_testing.txt', 'w') as outfile:
    # Iterate through each line in the input file
    for line in infile:
        # Check if the line contains the word 'ocean'
        if 'ocean' in line:
            outfile.write(line)
