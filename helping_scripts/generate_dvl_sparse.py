import os, argparse, csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser("Case 1: DVL 4-point square with optional viz")
    p.add_argument('--input_txt',   required=True, help="input paths.txt")
    p.add_argument('--output_txt',  required=True, help="filtered paths file")
    p.add_argument('--side_length', type=float, required=True,
                   help="square side length in original pixels")
    p.add_argument('--visualize',   action='store_true',
                   help="overlay sparse points (and depth values) on original image")
    args = p.parse_args()

    orig_w, orig_h = 968, 608
    norm_w, norm_h = 320, 240
    cx, cy = orig_w/2, orig_h/2
    half = args.side_length/2.0
    corners = [
        (cx-half, cy-half), (cx+half, cy-half),
        (cx-half, cy+half), (cx+half, cy+half),
    ]

    out_lines = []
    pattern = f"dvl_{int(args.side_length)}_pixels"

    with open(args.input_txt) as f:
        for L in f:
            img_p, depth_p, *_ = L.strip().split()
            depth = np.array(Image.open(depth_p))
            samples = []  # (orig_y, orig_x, norm_y, norm_x, depth)
            for x, y in corners:
                xi, yi = int(round(x)), int(round(y))
                if 0<=yi<depth.shape[0] and 0<=xi<depth.shape[1]:
                    d = float(depth[yi, xi])
                    if d>0:
                        ni = int(round(yi*(norm_h/orig_h)))
                        nj = int(round(xi*(norm_w/orig_w)))
                        samples.append((yi, xi, ni, nj, d))
            if len(samples)<4:
                continue

            imgs_dir = os.path.dirname(img_p)
            base_dir = os.path.dirname(imgs_dir)
            out_dir = os.path.join(base_dir, pattern)
            os.makedirs(out_dir, exist_ok=True)

            base = os.path.splitext(os.path.basename(img_p))[0]
            out_csv = os.path.join(
                out_dir,
                f"{base}_sparse_depth_{pattern}.csv"
            )
            with open(out_csv,'w',newline='') as cf:
                w = csv.writer(cf)
                w.writerow(['row','column','depth'])
                for yi, xi, ni, nj, d in samples:
                    w.writerow([ni, nj, d])

            out_lines.append(f"{img_p} {depth_p} {out_csv}")

            if args.visualize:
                img = np.array(Image.open(img_p))
                xs = [xi for yi, xi, *_ in samples]
                ys = [yi for yi, xi, *_ in samples]
                ds = [d for yi, xi, ni, nj, d in samples]
                plt.figure(figsize=(6,4))
                plt.imshow(img)
                plt.scatter(xs, ys, c='r', marker='x')
                for xpt, ypt, val in zip(xs, ys, ds):
                    plt.text(xpt+2, ypt+2, f"{val:.2f}",
                             fontsize=8, color='yellow')
                plt.title(base + '   ' + pattern)
                plt.axis('off')
                plt.show()

    with open(args.output_txt,'w') as tf:
        tf.write("\n".join(out_lines))

if __name__=='__main__':
    main()