# Open the text file and read its contents
with open('/media/jay/apple/FLSea_latest/archive/canyons/u_canyon/u_canyon/imgs/test_with_matched_features.txt', 'r') as file:
    content = file.read()

# Replace commas with spaces
modified_content = content.replace(',', ' ')

# Print the modified content
print(modified_content)

# Optionally, write the modified content back to a new file
with open('/media/jay/apple/FLSea_latest/archive/canyons/u_canyon/u_canyon/imgs/modified_test_with_matched_features.txt', 'w') as modified_file:
    modified_file.write(modified_content)
