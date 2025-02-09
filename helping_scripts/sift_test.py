import cv2
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


@dataclass
class CropParams:
    top: int
    bottom: int
    left: int
    right: int

def get_border_params(gray_image, tolerance=0.1, cut_off=20, value=255, level_diff_threshold=5, channel_axis=-1, min_border=5) -> CropParams:
    # Convert the image to grayscale by averaging across the color channels
    # if len(rgb_image.shape) == 3:  # Check if it's a color image (has 3 channels)
    #     gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    # else:
    #     # If the image is already grayscale
    #     gray_image = rgb_image
    # gray_image = np.mean(rgb_image, axis=channel_axis)
    # print(gray_image.shape)
    h, w = gray_image.shape

    def num_value_pixels(arr):
        return np.sum(np.abs(arr - value) < level_diff_threshold)

    def is_above_tolerance(arr, total_pixels):
        return (num_value_pixels(arr) / total_pixels) > tolerance

    # Crop top border until number of value pixels become below tolerance
    top = min_border
    while is_above_tolerance(gray_image[top, :], w) and top < h - 1:
        top += 1
        if top > cut_off:
            break

    # Crop bottom border until number of value pixels become below tolerance
    bottom = h - min_border
    while is_above_tolerance(gray_image[bottom, :], w) and bottom > 0:
        bottom -= 1
        if h - bottom > cut_off:
            break

    # Crop left border until number of value pixels become below tolerance
    left = min_border
    while is_above_tolerance(gray_image[:, left], h) and left < w - 1:
        left += 1
        if left > cut_off:
            break

    # Crop right border until number of value pixels become below tolerance
    right = w - min_border
    while is_above_tolerance(gray_image[:, right], h) and right > 0:
        right -= 1
        if w - right > cut_off:
            break

    return CropParams(top, bottom, left, right)

def extract_sift_from_patches(image, depth_img, total_keypoints=1000, n_rows=4, n_cols=4):
    # print(image.shape)
    height, width = image.shape[:2]

    # Get border parameters to define the region of interest (ROI)
    crop_params = get_border_params(image)

    # Create a mask based on the region of interest (ROI) defined by the borders
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[crop_params.top:crop_params.bottom, crop_params.left:crop_params.right] = 255  # Valid region set to 255

    # Combine with depth mask (excluding pixels with zero depth)
    depth_mask = (depth_img > 0).astype(np.uint8)
    combined_mask = cv2.bitwise_and(mask, depth_mask)

    # Calculate patch size based on the number of rows and columns
    patch_height = height // n_rows
    patch_width = width // n_cols

    # Calculate the number of keypoints per patch
    keypoints_per_patch = total_keypoints // (n_rows * n_cols)

    sift = cv2.SIFT_create(nfeatures=keypoints_per_patch, contrastThreshold=0.0)
    all_keypoints = []
    all_descriptors = []

    for i in range(n_rows):
        for j in range(n_cols):
            # Calculate patch boundaries
            y_start = i * patch_height
            y_end = (i + 1) * patch_height if (i + 1) * patch_height < height else height
            x_start = j * patch_width
            x_end = (j + 1) * patch_width if (j + 1) * patch_width < width else width

            # Extract patch and corresponding mask
            patch = image[y_start:y_end, x_start:x_end]
            patch_mask = combined_mask[y_start:y_end, x_start:x_end]

            # Detect keypoints and compute descriptors in the patch using the patch mask
            patch_keypoints, patch_descriptors = sift.detectAndCompute(patch, patch_mask)
            print(type(patch_keypoints))

            # Adjust keypoint coordinates to the full image
            for kp in patch_keypoints:
                kp.pt = (kp.pt[0] + x_start, kp.pt[1] + y_start)  # Adjust keypoint location to original image

            # Sort by response (strength) and take the strongest keypoints_per_patch
            patch_keypoints = sorted(patch_keypoints, key=lambda x: x.response, reverse=True)[:keypoints_per_patch]

            # Collect the keypoints and descriptors from this patch
            if patch_keypoints:
                all_keypoints.extend(patch_keypoints)
                if patch_descriptors is not None:
                    all_descriptors.append(patch_descriptors[:keypoints_per_patch])

    # Display results
    # display_image("Original Image", image)
    # display_image("Grayscale Image with Mask Applied", combined_mask)
    # display_image("Depth Image", depth_img)
    # keypoints_img = cv2.drawKeypoints(image, all_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # display_image("Keypoints in Valid Regions", keypoints_img)

    

    # Concatenate descriptors from all patches if they exist
    if all_descriptors:
        all_descriptors = np.vstack(all_descriptors)
    else:
        all_descriptors = None

    # Filter valid keypoints based on depth and the mask, and return keypoints and descriptors
    valid_keypoints = []
    valid_descriptors = []
    for idx, kp in enumerate(all_keypoints):
        if len(valid_keypoints) >= total_keypoints:
            break

        col, row = int(kp.pt[0]), int(kp.pt[1])

        if 0 <= row < depth_img.shape[0] and 0 <= col < depth_img.shape[1]:
            depth_value = depth_img[row, col]
            if depth_value > 0 and combined_mask[row, col]:  # Ensure keypoint is in the valid region
                valid_keypoints.append(all_keypoints[idx])
                if all_descriptors is not None:
                    valid_descriptors.append(all_descriptors[idx])

    return valid_keypoints, valid_descriptors

# Read the image
image_path = '/home/jay/shortcuts/datasets/nyu_depth_v2/sync/office_0004/rgb_00087.jpg'  # Replace with your image path
depth_path = '/home/jay/shortcuts/datasets/nyu_depth_v2/sync/office_0004/sync_depth_00087.png'
img = cv2.imread(image_path)
depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

# Convert to grayscale (SIFT works on grayscale images)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

keypoints, _  = extract_sift_from_patches(gray_img, depth_img)


# Create a SIFT detector object
# sift = cv2.SIFT_create(contrastThreshold=0.0)

# Detect keypoints in the image
# keypoints, _ = sift.detectAndCompute(gray_img, None)

# Sort keypoints by strength (response) and keep the top 200
if len(keypoints) > 700:
    keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:700]

# Print the number of keypoints after filtering
print(f"Number of features detected: {len(keypoints)}")

print(np.shape(keypoints))
# Draw keypoints on the image
img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the image with keypoints
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Hide axes
plt.show()
