import cv2
import matplotlib.pyplot as plt

# Read the image
image_path = '/home/jay/shortcuts/datasets/nyu_depth_v2/sync/basement_0001a/rgb_00000.jpg'  # Replace with your image path
img = cv2.imread(image_path)

# Convert to grayscale (SIFT works on grayscale images)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create a SIFT detector object
sift = cv2.SIFT_create()

# Detect keypoints in the image
keypoints, _ = sift.detectAndCompute(gray_img, None)

# Sort keypoints by strength (response) and keep the top 200
if len(keypoints) > 200:
    keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:200]

# Print the number of keypoints after filtering
print(f"Number of features detected: {len(keypoints)}")

# Draw keypoints on the image
img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the image with keypoints
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Hide axes
plt.show()
