import cv2
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import pandas as pd

# ---------------------------
# 1. Load images and extract SIFT features
# ---------------------------
# Replace these with your actual image paths
left_img = cv2.imread('/media/jay/Lexar/cape_don/stereo_left/5562.png')
right_img = cv2.imread('/media/jay/Lexar/cape_don/stereo_right/5562.png')

if left_img is None or right_img is None:
    raise IOError("One or both images not found. Check your file paths.")

if left_img is None or right_img is None:
    raise IOError("One or both images not found. Check your file paths.")

# left_img = cv2.resize(left_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
# right_img = cv2.resize(right_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)

# Convert images to grayscale
left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector with adjusted parameters:
# - nfeatures: maximum number of keypoints to retain (set high to get more points)
# - contrastThreshold: lowering this value detects more keypoints in low-contrast areas
sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.01)

# Detect keypoints and compute descriptors
kp_left, des_left = sift.detectAndCompute(left_gray, None)
kp_right, des_right = sift.detectAndCompute(right_gray, None)

# Print the total number of extracted keypoints
print("Number of SIFT keypoints in left image:", len(kp_left))
print("Number of SIFT keypoints in right image:", len(kp_right))

# Draw keypoints on the images
img_left_kp = cv2.drawKeypoints(left_img, kp_left, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_right_kp = cv2.drawKeypoints(right_img, kp_right, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the keypoints side by side
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img_left_kp, cv2.COLOR_BGR2RGB))
plt.title('Left Image: SIFT Keypoints')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_right_kp, cv2.COLOR_BGR2RGB))
plt.title('Right Image: SIFT Keypoints')
plt.axis('off')
plt.show()

# ---------------------------
# 2. Initial matching using BFMatcher and Lowe's ratio test
# ---------------------------
# Create BFMatcher object (L2 norm is used for SIFT)
bf = cv2.BFMatcher()

# Find the 2 best matches for each descriptor
matches = bf.knnMatch(des_left, des_right, k=2)

# Apply Lowe's ratio test to filter ambiguous matches
good_matches = []
ratio_thresh = 0.75
for m, n in matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

# Print the total number of good matches after Lowe's ratio test
print("Number of good matches after Lowe's ratio test:", len(good_matches))

# Draw initial matches
img_matches = cv2.drawMatches(left_img, kp_left, right_img, kp_right, good_matches,
                              None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(15, 7))
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title('Initial Matches (After Lowe\'s Ratio Test)')
plt.axis('off')
plt.show()

# ---------------------------
# 3. Epipolar constraint filtering
# ---------------------------
# Define intrinsic matrices for left and right cameras (replace with your calibration data)
K_left = np.array([[2408.38523,    0.     , 1268.81566],
            [0.     , 2406.83443, 1008.25582],
            [0.     ,    0.     ,    1.     ]], dtype=np.float64)

K_right = np.array([[2399.49081,    0.     , 1252.92453],
            [0.     , 2398.71712, 1012.79098],
            [0.     ,    0.     ,    1.     ]], dtype=np.float64)

# Define extrinsic parameters: rotation R and translation t (from left to right)
# R = np.eye(3, dtype=np.float64)
R = np.array([[1,0,0],
              [0,1,0],
              [0,0,1]], dtype=np.float64)
# For example, assume the right camera is shifted by 0.1 m along the x-axis relative to the left camera.
t = np.array([-0.2, 0.0, 0.0 ], dtype=np.float64)

# Compute the essential matrix: E = [t]_x R.
def skew_symmetric(vec):
    return np.array([[    0, -vec[2],  vec[1]],
                     [ vec[2],     0, -vec[0]],
                     [-vec[1],  vec[0],     0]])
E = skew_symmetric(t) @ R

# Compute the fundamental matrix: F = K_right^{-T} * E * K_left^{-1}
F = np.linalg.inv(K_right).T @ E @ np.linalg.inv(K_left)

# Filter matches using the epipolar constraint
epipolar_threshold = 500.0  # in pixels; adjust based on your application
filtered_matches = []

for match in good_matches:
    # Get homogeneous coordinates for the left and right keypoints
    pt_left = np.array([kp_left[match.queryIdx].pt[0], kp_left[match.queryIdx].pt[1], 1])
    pt_right = np.array([kp_right[match.trainIdx].pt[0], kp_right[match.trainIdx].pt[1], 1])
    
    # Compute the epipolar line in the right image for the left point: l' = F * pt_left
    l = F @ pt_left  # line coefficients [a, b, c] satisfying a*x + b*y + c = 0
    # Compute distance from pt_right to the epipolar line
    distance = np.abs(np.dot(l, pt_right)) / np.sqrt(l[0]**2 + l[1]**2)
    
    if distance < epipolar_threshold:
        filtered_matches.append(match)

# Print the total number of filtered matches after applying the epipolar constraint
print("Number of filtered matches after epipolar constraint:", len(filtered_matches))

# Draw the filtered matches
img_filtered_matches = cv2.drawMatches(left_img, kp_left, right_img, kp_right, filtered_matches,
                                       None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(15, 7))
plt.imshow(cv2.cvtColor(img_filtered_matches, cv2.COLOR_BGR2RGB))
plt.title('Filtered Matches (Epipolar Constraint)')
plt.axis('off')
plt.show()

# ---------------------------
# 4. Depth (Range) computation using triangulation
# ---------------------------
# Create projection matrices:
# Left camera: P_left = K_left [I | 0]
# Right camera: P_right = K_right [R | t]
P_left = K_left @ np.hstack((np.eye(3), np.zeros((3, 1))))
P_right = K_right @ np.hstack((R, t.reshape(3, 1)))

# Gather point coordinates from filtered matches
pts_left = np.array([kp_left[m.queryIdx].pt for m in filtered_matches]).T  # shape: (2, N)
pts_right = np.array([kp_right[m.trainIdx].pt for m in filtered_matches]).T  # shape: (2, N)

# Triangulate points (results are in homogeneous coordinates)
points_4d = cv2.triangulatePoints(P_left, P_right, pts_left, pts_right)
# Convert to Euclidean coordinates (divide by the fourth homogeneous coordinate)
points_3d = points_4d[:3] / points_4d[3]

# Extract depth (z-coordinate) of points in the left camera coordinate system
depths = points_3d[2, :]

# ---------------------------
# 5. Filter points: only interested in points within 2 m
# ---------------------------
mask = (depths > 0) & (depths < 200.0)   # boolean mask for points within 2 m
pts_left_filtered = pts_left[:, mask]
depths_filtered = depths[mask]

# Update total number of filtered points and convert depths to a Pandas Series for interactive annotation
total_points = pts_left_filtered.shape[1]
depth_series = pd.Series(depths_filtered)

# Save the filtered depth points along with their pixel locations (row and column)
# Note: x corresponds to column and y corresponds to row
x = pts_left_filtered[0]
y = pts_left_filtered[1]
df = pd.DataFrame({
    'row': y,
    'column': x,
    'depth': depths_filtered
})
df.to_csv('/media/jay/Lexar/cape_don/stereo_left/5562.csv', index=False)
print("Filtered depth points (with pixel locations) saved to 'filtered_depth_points.csv'.")

# ---------------------------
# 6. Final Visualization: Overlay Depth Points Using Subplots
# ---------------------------
height, width = left_img.shape[:2]
depth_matrix = np.full((height, width), np.nan, dtype=np.float32)
for i in range(total_points):
    x_int = int(round(x[i]))
    y_int = int(round(y[i]))
    if 0 <= x_int < width and 0 <= y_int < height:
        depth_matrix[y_int, x_int] = depth_series.iloc[i]
depth_matrix_disp = np.nan_to_num(depth_matrix, nan=0)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
ax1 = axs[0]
ax1.imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB), origin='upper')
scatter = ax1.scatter(x, y, c=depths_filtered, cmap='viridis', s=30)
ax1.set_title('Overlay Image with Depth Points (0 < depth < 200 m)')
ax1.set_xlabel('Column')
ax1.set_ylabel('Row')
ax1.text(0.05, 0.95, f"Total Points: {total_points}",
         transform=ax1.transAxes, fontsize=12, color='white',
         bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

cursor = mplcursors.cursor(scatter, hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(
    f"Depth: {depth_series.iloc[sel.index]:.2f} m"))

ax2 = axs[1]
im = ax2.imshow(depth_matrix_disp, origin='upper', cmap='viridis')
ax2.set_title('Depth Matrix')
ax2.set_xlabel('Column')
ax2.set_ylabel('Row')
cbar = fig.colorbar(im, ax=ax2)
cbar.set_label('Depth (m)')

plt.tight_layout()
plt.show()