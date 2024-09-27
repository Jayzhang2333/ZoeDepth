import cv2
import numpy as np
import matplotlib.pyplot as plt
from extract_nyu_sparse_depth import extract_sift_from_patches

# Function to display the image
def display_image(title, img):
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Function to extract SIFT keypoints from valid regions
def test_sift_on_valid_regions(image_path, depth_path):
    # Load the RGB image and depth map
    img = cv2.imread(image_path)
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if img is None or depth_img is None:
        print(f"Error loading images. Check paths:\nImage: {image_path}\nDepth: {depth_path}")
        return

    # Convert RGB image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # # Create a mask for non-white regions in the RGB image and non-black regions in the depth map
    # rgb_mask = cv2.inRange(img, (0, 0, 0), (254, 254, 254))  # Mask out white boundaries (255,255,255)
    # depth_mask = (depth_img > 0).astype(np.uint8)  # Mask out black boundaries (depth=0)

    # # Combine both masks to get a valid region mask
    # combined_mask = cv2.bitwise_and(rgb_mask, depth_mask)

    # # Apply the mask to the grayscale image (for visual testing)
    # masked_image = cv2.bitwise_and(gray_img, gray_img, mask=combined_mask)

    # # Detect SIFT keypoints in the valid region
    # sift = cv2.SIFT_create()
    # keypoints, descriptors = sift.detectAndCompute(masked_image, combined_mask)

    #
    valid_keypoints, valid_descriptors = extract_sift_from_patches(img, gray_img)
    # Draw keypoints on the original image (for visualization)
    # keypoints_img = cv2.drawKeypoints(img, valid_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # # Display results
    # display_image("Original Image", img)
    # # display_image("Grayscale Image with Mask Applied", masked_image)
    # display_image("Depth Image", depth_img)
    # display_image("Keypoints in Valid Regions", keypoints_img)

    # print(f"Total keypoints detected in valid regions: {len(valid_keypoints)}")

# Test paths (replace these with your own image and depth map paths)
image_path = '/home/jay/shortcuts/datasets/nyu_depth_v2/sync/basement_0001a/rgb_00000.jpg'  # Replace with your image path
depth_path = '/home/jay/shortcuts/datasets/nyu_depth_v2/sync/basement_0001a/sync_depth_00000.png'  # Replace with your depth map path

# Run the test
test_sift_on_valid_regions(image_path, depth_path)
