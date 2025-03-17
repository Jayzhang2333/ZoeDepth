import os
import cv2
import numpy as np

def process_image(image_path, depth_path, sift, max_features=250):
    """
    Process an image: detect SIFT keypoints, select the top keypoints,
    then filter them based on the corresponding depth values from the depth map.
    Returns a list of (row, column, depth) tuples.
    """
    # Load the image and convert to grayscale
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect SIFT features
    keypoints, _ = sift.detectAndCompute(gray, None)
    # Sort keypoints by their response in descending order and select top max_features
    keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)[:max_features]
    
    # Load the depth map; use IMREAD_UNCHANGED to preserve the raw depth values
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        print(f"Could not read depth map: {depth_path}")
        return []
    
    features = []
    for kp in keypoints:
        # SIFT keypoints use (x, y) as (column, row)
        col = int(round(kp.pt[0]))
        row = int(round(kp.pt[1]))
        # Check that the point is within the depth map boundaries
        if row < 0 or row >= depth_img.shape[0] or col < 0 or col >= depth_img.shape[1]:
            continue
        # Get the depth value; convert to float if needed
        depth_value = float(depth_img[row, col])
        # Filter: keep only if depth is between 0.1 and 18 meters
        if depth_value < 0.1 or depth_value > 18:
            continue
        features.append((row, col, depth_value))
    
    return features

def save_sparse_features(sparse_file_path, features):
    """
    Save the features to a text file with header "row column depth"
    followed by each feature on a new line.
    """
    with open(sparse_file_path, 'w') as f:
        f.write("row column depth\n")
        for row, col, depth in features:
            f.write(f"{row} {col} {depth}\n")

def main():
    # Base directory where the 'sea_thru' folder is located
    base_dir = '/media/jay/Lexar/SeaThru/sea_thru'
    # List of subfolders
    folders = ['D1', 'D2', 'D3', 'D4', 'D5']
    
    # This list will store rows for the master file.
    # Each row: image_path, depth_path, sparse_features_file_path
    master_rows = []
    
    # Create a SIFT detector. (Requires opencv-contrib-python)
    sift = cv2.SIFT_create()
    
    for folder in folders:
        # Construct paths for image and depth directories.
        # e.g. /media/jay/Lexar/SeaThru/sea_thru/D1/D1/linearPNG/
        image_folder = os.path.join(base_dir, folder, folder, "linearPNG")
        depth_folder = os.path.join(base_dir, folder, folder, "depth")
        
        if not os.path.exists(image_folder):
            print(f"Image folder does not exist: {image_folder}. Skipping folder {folder}.")
            continue
        
        # Process each PNG image in the image_folder
        for filename in os.listdir(image_folder):
            if not filename.lower().endswith('.png'):
                continue
            image_path = os.path.join(image_folder, filename)
            # Construct corresponding depth filename. 
            # Example: "T_S02951.png" -> "depthT_S02951.tif"
            base_name = os.path.splitext(filename)[0]
            depth_filename = "depth" + base_name + ".tif"
            depth_path = os.path.join(depth_folder, depth_filename)
            
            if not os.path.exists(depth_path):
                print(f"Depth file not found for image {image_path}: {depth_path}. Skipping this file.")
                continue
            
            # Process the image to get valid features with depth values
            features = process_image(image_path, depth_path, sift)
            
            # Create the sparse feature file path.
            # Save in the same folder as the image, with the image base name plus '_sparse_features.txt'
            sparse_file_name = base_name + "_sparse_features.txt"
            sparse_file_path = os.path.join(image_folder, sparse_file_name)
            
            # Save the features to the sparse features file.
            save_sparse_features(sparse_file_path, features)
            
            # Append the paths (image, depth, sparse features) to the master list.
            master_rows.append(f"{image_path} {depth_path} {sparse_file_path}")
    
    # Save the master list to a text file.
    master_file_path = os.path.join(base_dir, "image_depth_sparse_paths.txt")
    with open(master_file_path, 'w') as master_file:
        for row in master_rows:
            master_file.write(row + "\n")
    
    print(f"Processing complete. Master file saved at: {master_file_path}")

if __name__ == "__main__":
    main()
