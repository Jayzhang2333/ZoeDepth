def swap_paths_in_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            parts = line.strip().split()
            if len(parts) == 3:
                parts[1], parts[2] = parts[2], parts[1]  # Swap second and third elements
            outfile.write(' '.join(parts) + '\n')

# Example usage
input_file = "/media/jay/Seagate/Tartanair/tartanair_testing.txt"  # Change this to your actual file
output_file = "/media/jay/Seagate/Tartanair/tartanair_testing_swap.txt"
swap_paths_in_file(input_file, output_file)