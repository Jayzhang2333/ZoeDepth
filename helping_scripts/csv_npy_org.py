#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_sparse_map_from_csv(
    feature_fp,
    height=480,
    width=640,
    inverse_depth=False
):
    """
    Build a (height×width) sparse map from a CSV or TXT listing row, col, depth.
    """
    ext = os.path.splitext(feature_fp)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(feature_fp)
    elif ext == '.txt':
        df = pd.read_csv(feature_fp, delim_whitespace=True, header=0,
                         names=['row','column','depth'])
    else:
        raise ValueError("Only .csv or .txt supported")

    out = np.zeros((height, width), dtype=np.float32)
    for _, r in df.iterrows():
        pr = int(r['row'])
        pc = int(r['column'])
        if 0 <= pr < height and 0 <= pc < width:
            d = float(r['depth'])
            out[pr, pc] = (1.0/d) if inverse_depth else d
    return out

def load_sparse_map_from_npy(npy_fp, inverse_depth=False):
    """
    Load the full-resolution sparse map from .npy.
    """
    arr = np.load(npy_fp)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if inverse_depth:
        mask = arr != 0
        inv = np.zeros_like(arr, dtype=np.float32)
        inv[mask] = 1.0 / arr[mask]
        arr = inv
    return arr

def plot_side_by_side(map_csv, map_npy):
    """
    Display two 2D arrays side by side with a shared color scale.
    """
    vmin, vmax = 0.0, max(map_csv.max(), map_npy.max())
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    im0 = axs[0].imshow(map_csv, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0].set_title('From CSV/TXT')
    axs[0].axis('off')

    im1 = axs[1].imshow(map_npy, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1].set_title('From NPY')
    axs[1].axis('off')

    cbar = fig.colorbar(im1, ax=axs, fraction=0.046, pad=0.04)
    cbar.set_label('Depth value')
    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: display_original.py <feature.csv|.txt> <feature.npy> [--inverse-depth]")
        sys.exit(1)

    feature_fp = sys.argv[1]
    npy_fp     = sys.argv[2]
    inv        = (len(sys.argv) == 4 and sys.argv[3] == "--inverse-depth")

    if not os.path.isfile(feature_fp):
        print(f"Error: not found {feature_fp}")
        sys.exit(1)
    if not os.path.isfile(npy_fp):
        print(f"Error: not found {npy_fp}")
        sys.exit(1)

    print(f"Loading at original resolution (480×640), inverse_depth={inv}\n")
    map_csv = load_sparse_map_from_csv(feature_fp, inverse_depth=inv)
    map_npy = load_sparse_map_from_npy(npy_fp, inverse_depth=inv)

    plot_side_by_side(map_csv, map_npy)

if __name__ == "__main__":
    main()
