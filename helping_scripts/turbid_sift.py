import cv2
import csv
import argparse
import numpy as np

def extract_sift_points(image_path, output_csv, num_points=20):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    # Normalize intensity to full 0-255 range for better feature detection
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    img = img.astype(np.uint8)

    # Optionally apply histogram equalization
    img = cv2.equalizeHist(img)

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints
    keypoints = sift.detect(img, None)
    if not keypoints:
        print("No keypoints found in the image.")
        return

    # Sort by response (strength) and take top num_points
    keypoints = sorted(keypoints, key=lambda k: k.response, reverse=True)[:num_points]

    # Write to CSV
    with open(output_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['row', 'column'])  # header
        for kp in keypoints:
            # kp.pt returns (x, y)
            x, y = kp.pt
            row = int(round(y))
            col = int(round(x))
            writer.writerow([row, col])

    print(f"Wrote {len(keypoints)} SIFT points to {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract SIFT feature points and save to CSV.")
    parser.add_argument('image_path', help='Path to the input PNG image')
    parser.add_argument('output_csv', help='Path to the output CSV file')
    parser.add_argument('--num_points', type=int, default=50, help='Number of SIFT points to extract (default: 20)')
    args = parser.parse_args()

    extract_sift_points(args.image_path, args.output_csv, args.num_points)
