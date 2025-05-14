
import os
import csv
import argparse
import numpy as np
import cv2

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute MAE between sparse depth CSV and GT depth map")
    p.add_argument('sparse_csv',
                   help="path to sparse depth CSV (row,column,depth)")
    p.add_argument('depth_map',
                   help="path to ground truth depth TIFF")
    return p.parse_args()

def load_sparse(sparse_csv):
    pts = []
    with open(sparse_csv, newline='') as f:
        rdr = csv.reader(f)
        next(rdr)  # skip header
        for row in rdr:
            r = int(row[0])
            c = int(row[1])
            d_pred = float(row[2])
            pts.append((r, c, d_pred))
    return pts

def load_depth_map(depth_tif):
    """Reads the GT depth map as a float32 array."""
    img = cv2.imread(depth_tif, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError("Could not load depth map {}".format(depth_tif))
    return img.astype(np.float32)

def compute_mae(sparse_pts, depth_map):
    errors = []
    h, w = depth_map.shape[:2]
    for r, c, d_pred in sparse_pts:
        if 0 <= r < h and 0 <= c < w:
            d_gt = depth_map[r, c]
            # only consider positive (valid) GT depths
            if d_gt > 0:
                errors.append(abs(d_pred - d_gt))
    if not errors:
        raise ValueError("No valid points found (all GT <= 0 or out of bounds)")
    arr = np.array(errors, dtype=np.float64)
    return arr.mean(), len(arr)

def main():
    args = parse_args()

    sparse_pts = load_sparse(args.sparse_csv)
    depth_map  = load_depth_map(args.depth_map)

    mae, n = compute_mae(sparse_pts, depth_map)
    print("Found {} valid sparse points".format(n))
    print("Mean absolute error: {:.4f} (same units as depth map)".format(mae))

if __name__ == "__main__":
    main()
