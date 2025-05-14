import cv2
import os

def downsample_image(input_path, output_path, scale=0.5):
    # Load the original image from file
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Image file '{input_path}' not found.")

    # Get original dimensions and compute new dimensions based on the scale factor
    original_height, original_width = image.shape[:2]
    # new_width = int(original_width * scale)
    # new_height = int(original_height * scale)
    new_width = 448
    new_height = 336

    # Resize the image using INTER_AREA interpolation (good for downsampling)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create the output directory if it does not exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the resized image to the given output location
    cv2.imwrite(output_path, resized_image)
    print(f"Image resized from ({original_width}x{original_height}) to ({new_width}x{new_height}) and saved as '{output_path}'.")

if __name__ == "__main__":
    input_path = "/home/jay/Downloads/first_100_cropped_resized/00002.png"          # Replace with your input image file path
    output_path = "/home/jay/00002_resized.png"  # Replace with your desired output file location
    downsample_image(input_path, output_path)
