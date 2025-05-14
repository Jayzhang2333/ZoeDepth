import os
import csv
import argparse
import numpy as np
import cv2

def parse_args():
    p = argparse.ArgumentParser(
        description="Filter list of (GT depth, sparse-depth CSV) pairs by MAE")
    p.add_argument('list_file',
                   help="path to txt file where each row has: <any> <gt_depth.tif> <sparse.csv>")
    p.add_argument('output_file',
                   help="path to write the filtered list (one row per valid entry)")
    p.add_argument('--threshold', type=float, default=0.5,
                   help="MAE threshold (default: 0.5)")
    return p.parse_args()

def load_sparse(sparse_csv):
    pts = []
    with open(sparse_csv, newline='') as f:
        rdr = csv.reader(f)
        next(rdr)  # skip header
        for row in rdr:
            r, c, d_pred = int(row[0]), int(row[1]), float(row[2])
            pts.append((r, c, d_pred))
    return pts

def load_depth_map(depth_tif):
    img = cv2.imread(depth_tif, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Could not load depth map {depth_tif}")
    return img.astype(np.float32)

def compute_mae(sparse_pts, depth_map):
    errors = []
    h, w = depth_map.shape[:2]
    for r, c, d_pred in sparse_pts:
        if 0 <= r < h and 0 <= c < w:
            d_gt = depth_map[r, c]
            if d_gt > 0:
                errors.append(abs(d_pred - d_gt))
    if not errors:
        raise ValueError("No valid points found (all GT ≤0 or OOB)")
    return float(np.mean(errors)), len(errors)

def main():
    args = parse_args()
    valid_lines = []

    # Read all lines from the input list
    with open(args.list_file, 'r') as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]

    for idx, line in enumerate(lines, 1):
        parts = line.split()
        if len(parts) < 3:
            print(f"[Line {idx}] Skipping—need at least 3 columns, got {len(parts)}")
            continue

        gt_depth_path = parts[1]
        sparse_csv_path = parts[2]

        # sanity check
        if not os.path.exists(gt_depth_path):
            print(f"[Line {idx}] Missing GT file: {gt_depth_path}")
            continue
        if not os.path.exists(sparse_csv_path):
            print(f"[Line {idx}] Missing CSV file: {sparse_csv_path}")
            continue

        try:
            sparse_pts = load_sparse(sparse_csv_path)
            depth_map = load_depth_map(gt_depth_path)
            mae, count = compute_mae(sparse_pts, depth_map)
        except Exception as e:
            print(f"[Line {idx}] Error computing MAE: {e}")
            continue

        print(f"[Line {idx}] MAE={mae:.4f} over {count} points")
        if mae < args.threshold:
            valid_lines.append(line)

    # Write the filtered list
    with open(args.output_file, 'w') as fo:
        for ln in valid_lines:
            fo.write(ln + '\n')

    print(f"\nDone: kept {len(valid_lines)}/{len(lines)} entries with MAE < {args.threshold}")

if __name__ == "__main__":
    main()
