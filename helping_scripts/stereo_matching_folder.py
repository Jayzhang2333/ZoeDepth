import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import pandas as pd

# ---------------------------
# Configuration and Setup
# ---------------------------
# Folder containing left images (update this path as needed)
left_folder_path = '/home/jay/Downloads/MinimizeBlur-20250318T231652Z-001/MinimizeBlur/RV_STEREO_20-02-2025_05-23-30_cork-pipe-gain-max-10/4_RECTIFIED_L'
# List all PNG files in the left folder (adjust extension if needed)
left_image_files = glob.glob(os.path.join(left_folder_path, '*.png'))

if not left_image_files:
    raise IOError("No left images found in the specified folder.")

# Pre-initialize SIFT detector
sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.01)

# Define camera calibration matrices and extrinsics (use your calibration data)
K_left = np.array([[265.9079217283398,   0, 482.7633260641041],
                   [  0, 266.0764281386958, 303.7255345443481],
                   [  0,   0,   1]], dtype=np.float64)
K_right = np.array([[267.4608331041126,   0, 480.3668657845864],
                    [  0, 267.5633105328833, 299.1577484876895],
                    [  0,   0,   1]], dtype=np.float64)
R = np.array([[0.9999797421797889,   -0.004138713224455445, 0.004835936609310673],
              [  0.004125989256907841, 0.9999880081199976, 0.0026381487547311156],
              [ -0.0048467971584784395,   -0.0026181422890908484,   0.999984826829017]], dtype=np.float64)
t = np.array([-0.05770203065611219, -0.0003244103628350639, 0.000502764081998493], dtype=np.float64)

# Helper function to compute skew-symmetric matrix
def skew_symmetric(vec):
    return np.array([[    0, -vec[2],  vec[1]],
                     [ vec[2],     0, -vec[0]],
                     [-vec[1],  vec[0],     0]])

E = skew_symmetric(t) @ R
F = np.linalg.inv(K_right).T @ E @ np.linalg.inv(K_left)
epipolar_threshold = 10.0  # in pixels; adjust as needed

# ---------------------------
# Process each image pair in the folder
# ---------------------------
for left_image_path in left_image_files:
    # Construct the right image path by replacing the folder substring:
    right_image_path = left_image_path.replace("4_RECTIFIED_L", "5_RECTIFIED_R")
    
    print(f"Processing left image: {left_image_path}")
    print(f"Corresponding right image: {right_image_path}")
    
    # Load images
    left_img = cv2.imread(left_image_path)
    right_img = cv2.imread(right_image_path)
    
    if left_img is None or right_img is None:
        print(f"Warning: Could not load one or both images for {left_image_path}. Skipping...")
        continue

    # Convert images to grayscale
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    # ---------------------------
    # 1. SIFT Keypoint Detection
    # ---------------------------
    kp_left, des_left = sift.detectAndCompute(left_gray, None)
    kp_right, des_right = sift.detectAndCompute(right_gray, None)
    
    print("Number of SIFT keypoints in left image:", len(kp_left))
    print("Number of SIFT keypoints in right image:", len(kp_right))
    
    # Draw keypoints (optional visualization)
    img_left_kp = cv2.drawKeypoints(left_img, kp_left, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_right_kp = cv2.drawKeypoints(right_img, kp_right, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
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
    # 2. Matching using BFMatcher and Lowe's Ratio Test
    # ---------------------------
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_left, des_right, k=2)
    
    good_matches = []
    ratio_thresh = 0.75
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    
    print("Number of good matches after Lowe's ratio test:", len(good_matches))
    
    img_matches = cv2.drawMatches(left_img, kp_left, right_img, kp_right, good_matches,
                                  None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(15, 7))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title("Initial Matches (After Lowe's Ratio Test)")
    plt.axis('off')
    plt.show()
    
    # ---------------------------
    # 3. Epipolar Constraint Filtering
    # ---------------------------
    filtered_matches = []
    for match in good_matches:
        pt_left = np.array([kp_left[match.queryIdx].pt[0], kp_left[match.queryIdx].pt[1], 1])
        pt_right = np.array([kp_right[match.trainIdx].pt[0], kp_right[match.trainIdx].pt[1], 1])
        l = F @ pt_left
        distance = np.abs(np.dot(l, pt_right)) / np.sqrt(l[0]**2 + l[1]**2)
        if distance < epipolar_threshold:
            filtered_matches.append(match)
    print("Number of filtered matches after epipolar constraint:", len(filtered_matches))
    
    img_filtered_matches = cv2.drawMatches(left_img, kp_left, right_img, kp_right, filtered_matches,
                                           None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(15, 7))
    plt.imshow(cv2.cvtColor(img_filtered_matches, cv2.COLOR_BGR2RGB))
    plt.title("Filtered Matches (Epipolar Constraint)")
    plt.axis('off')
    plt.show()
    
    # ---------------------------
    # 4. Depth (Range) Computation using Triangulation
    # ---------------------------
    # Create projection matrices for left and right cameras
    P_left = K_left @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P_right = K_right @ np.hstack((R, t.reshape(3, 1)))
    
    pts_left = np.array([kp_left[m.queryIdx].pt for m in filtered_matches]).T
    pts_right = np.array([kp_right[m.trainIdx].pt for m in filtered_matches]).T
    
    points_4d = cv2.triangulatePoints(P_left, P_right, pts_left, pts_right)
    points_3d = points_4d[:3] / points_4d[3]
    depths = points_3d[2, :]
    
    # ---------------------------
    # 5. Filter Points: Only interested in points within 2 m
    # ---------------------------
    mask = (depths > 0) & (depths < 2.0)
    pts_left_filtered = pts_left[:, mask]
    depths_filtered = depths[mask]
    
    total_points = pts_left_filtered.shape[1]
    depth_series = pd.Series(depths_filtered)
    
    x = pts_left_filtered[0]
    y = pts_left_filtered[1]
    df = pd.DataFrame({
        'row': y,
        'column': x,
        'depth': depths_filtered
    })
    
    # ---------------------------
    # 6. Save CSV with Depth Points
    # ---------------------------
    # CSV file name: use the left image's name prefixed with "sparse_depth_"
    left_basename = os.path.basename(left_image_path)
    name_without_ext, _ = os.path.splitext(left_basename)
    csv_filename = os.path.join(left_folder_path, f'sparse_depth_{name_without_ext}.csv')
    df.to_csv(csv_filename, index=False)
    print(f"Filtered depth points saved to '{csv_filename}'.")
    
    # ---------------------------
    # 7. Final Visualization: Overlay Depth Points
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
    ax1.set_title('Overlay Image with Depth Points (0 < depth < 2 m)')
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
