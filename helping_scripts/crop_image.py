from PIL import Image

def crop_bottom(input_path, output_path, crop_percentage=0.8):
    """
    Crop the bottom of the image so that only the top portion remains.
    For example, a crop_percentage of 0.8 keeps the top 80% of the image.
    
    :param input_path: Path to the input image file.
    :param output_path: Path to save the cropped image.
    :param crop_percentage: The fraction (between 0 and 1) of the original height to keep from the top.
    """
    # Open the image
    image = Image.open(input_path)
    width, height = image.size

    # Calculate the new height that will be kept
    new_height = int(height * crop_percentage)

    # Crop the image from the top to new_height
    # The crop box is a 4-tuple: (left, upper, right, lower)
    cropped_image = image.crop((0, 0, width, new_height))
    
    # Save the cropped image
    cropped_image.save(output_path)
    print(f"Cropped image saved to {output_path}")

# Example usage:
if __name__ == '__main__':
    input_file = '/media/jay/apple/Sunboat_03-09-2023/2023-09-03-07-58-37/camera/00018.png'    # Path to your image file
    output_file = '/media/jay/apple/Sunboat_03-09-2023/2023-09-03-07-58-37/camera/00018_crop.png'  # Desired output path
    crop_bottom(input_file, output_file, crop_percentage=0.8)
