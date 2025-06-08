#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# -----------------------------
# 1) Load full-res CSV/TXT map
# -----------------------------
def load_full_csv(
    feature_fp,
    height=480, width=640,
    inverse_depth=False
):
    ext = os.path.splitext(feature_fp)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(feature_fp)
    elif ext == '.txt':
        df = pd.read_csv(feature_fp, sep=r'\s+', header=0,
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

# -----------------------------
# 2) Load full-res NPY map
# -----------------------------
def load_full_npy(npy_fp, inverse_depth=False):
    arr = np.load(npy_fp)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if inverse_depth:
        mask = arr != 0
        inv  = np.zeros_like(arr, dtype=np.float32)
        inv[mask] = 1.0 / arr[mask]
        arr = inv
    return arr

# ----------------------------------------
# 3) CPU: your original down-sampling
# ----------------------------------------
def generate_feature_map_for_ga(
    feature_fp,
    orig_h=480, orig_w=640,
    new_h=336, new_w=448,
    inverse_depth=False
):
    ext = os.path.splitext(feature_fp)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(feature_fp)
    else:  # .txt
        df = pd.read_csv(feature_fp, sep=r'\s+', header=0,
                         names=['row','column','depth'])

    sy = new_h / orig_h
    sx = new_w / orig_w
    out = np.zeros((new_h, new_w), dtype=np.float32)

    for _, r in df.iterrows():
        pr = int(r['row'] * sy)
        pc = int(r['column'] * sx)
        if 0 <= pr < new_h and 0 <= pc < new_w:
            d = float(r['depth'])
            out[pr, pc] = (1.0/d) if inverse_depth else d

    return out[..., None]

# ------------------------------------------------:
# 4) GPU module: exact floor-and-scatter logic
# ------------------------------------------------:
class SparseRescale(torch.nn.Module):
    def __init__(self,
                 orig_size=(480, 640),
                 new_size =(336, 448),
                 inverse_depth=False):
        super().__init__()
        self.orig_h, self.orig_w = orig_size
        self.new_h,  self.new_w = new_size
        self.sy = self.new_h / self.orig_h
        self.sx = self.new_w / self.orig_w
        self.inverse_depth = inverse_depth

    def forward(self, arr: torch.Tensor):
        # arr: (H,W) or (H,W,1)
        if arr.ndim == 3:
            arr = arr[..., 0]
        device, dtype = arr.device, arr.dtype

        if self.inverse_depth:
            mask = arr != 0
            arr = torch.where(mask, 1.0/arr, torch.zeros_like(arr))

        coords = torch.nonzero(arr, as_tuple=False)  # (N,2)
        if coords.numel() == 0:
            return torch.zeros(self.new_h, self.new_w, 1,
                               device=device, dtype=dtype)

        vals = arr[coords[:,0], coords[:,1]]
        ny = torch.floor(coords[:,0].float() * self.sy).long().clamp(0, self.new_h-1)
        nx = torch.floor(coords[:,1].float() * self.sx).long().clamp(0, self.new_w-1)

        out = torch.zeros(self.new_h, self.new_w, device=device, dtype=dtype)
        out[ny, nx] = vals
        return out.unsqueeze(-1)

# -----------------------------
# 5) Plot side by side helper
# -----------------------------
def plot_side_by_side(maps, titles):
    """
    maps: list of 2D arrays
    titles: list of strings
    """
    n = len(maps)
    vmin = 0.0
    vmax = max(m.max() for m in maps)

    fig, axs = plt.subplots(1, n, figsize=(6*n, 6), constrained_layout=True)
    if n == 1:
        axs = [axs]
    for ax, m, t in zip(axs, maps, titles):
        im = ax.imshow(m, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(t)
        ax.axis('off')

    # shared colorbar
    fig.colorbar(im, ax=axs, fraction=0.046, pad=0.02, label='Depth')
    plt.show()

# -----------------------------
# 6) Main comparison driver
# -----------------------------
def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: compare_rescale.py <feature.csv|.txt> <feature.npy> [--inverse-depth]")
        sys.exit(1)

    feature_fp, npy_fp = sys.argv[1], sys.argv[2]
    inv = (len(sys.argv) == 4 and sys.argv[3] == "--inverse-depth")

    # 1) Original resolution check
    map_csv_full = load_full_csv(feature_fp, inverse_depth=inv)
    map_npy_full = load_full_npy(npy_fp, inverse_depth=inv)
    same_full = np.array_equal(map_csv_full, map_npy_full)
    print(f"Original 480×640 match? {'✅ Yes' if same_full else '❌ No'}")
    if not same_full:
        diff_full = map_csv_full - map_npy_full
        print(f" • differing pixels: {np.count_nonzero(diff_full)}")
        print(f" • max diff: {diff_full.max():.6f}, min diff: {diff_full.min():.6f}")
        # Plot original to inspect
        plot_side_by_side(
            [map_csv_full, map_npy_full],
            ['CSV/TXT full-res', 'NPY full-res']
        )
        print("→ These original differences may explain downstream mismatches.")

    # 2) New resolution comparison
    cpu_map = generate_feature_map_for_ga(feature_fp, inverse_depth=inv)
    # GPU rescale
    arr = map_npy_full
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t = torch.from_numpy(arr).to(device).float()
    rescaler = SparseRescale(inverse_depth=inv).to(device)
    with torch.no_grad():
        gpu_map = rescaler(t).cpu().numpy()[..., 0]

    gpu_map = gpu_map[..., None]
    same_new = np.array_equal(cpu_map, gpu_map)
    print(f"Rescaled {cpu_map.shape} match? {'✅ Yes' if same_new else '❌ No'}")
    if not same_new:
        diff_new = cpu_map[...,0] - gpu_map[...,0]
        print(f" • differing pixels: {np.count_nonzero(diff_new)}")
        print(f" • max diff: {diff_new.max():.6f}, min diff: {diff_new.min():.6f}")
        # Plot new resolution
        plot_side_by_side(
            [cpu_map[...,0], gpu_map[...,0]],
            ['CPU downsample', 'GPU SparseRescale']
        )

if __name__ == "__main__":
    main()
