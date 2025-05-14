import cv2
import numpy as np

# Original camera intrinsic matrix (example values)
camera_matrix = np.array([[265.9079217283398, 0, 482.7633260641041],
                          [0, 266.0764281386958, 303.7255345443481],
                          [0,  0,  1]], dtype=np.float64)

# Original distortion coefficients [k1, k2, p1, p2, k3]
dist_coeffs = np.array([
    -0.00889757606638573,   # k1
    0.000464578619998849,   # k2
    -0.00889757606638573,   # p1
    2.993110428465776e-05   # p2
], dtype=float)

# Load your distorted image
# img = cv2.imread('distorted_image.png')
h = 608
w = 968

# Compute optimal new camera matrix
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix,
                                                       dist_coeffs,
                                                       (w, h),
                                                       alpha=0)  # alpha=0 means all pixels in rectified image are valid, 1 preserves all original pixels with black regions

# # Undistort (rectify) the image
# rectified_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

# # Crop the image based on the returned region of interest (optional but recommended)
# x, y, w, h = roi
# rectified_img = rectified_img[y:y+h, x:x+w]

# # Save or display the result
# cv2.imwrite('rectified_image.png', rectified_img)

# Your new intrinsic parameters:
print("New intrinsic matrix:")
print(new_camera_matrix)
