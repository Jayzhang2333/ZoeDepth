import os, argparse, csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser("Case 2: 2D‐sonar line with vertical jitter")
    p.add_argument(
        '--input_txt', required=True,
        help="input paths.txt (each line: <rgb_path> <depth_path> ...)"
    )
    p.add_argument(
        '--output_txt', required=True,
        help="output file listing: <rgb_path> <depth_path> <sparse_csv>"
    )
    p.add_argument(
        '--spacing', type=int, required=True,
        help="horizontal pixel spacing between sonar samples"
    )
    p.add_argument(
        '--std', type=float, default=5.0,
        help="vertical jitter (std dev in pixels) around center‐row"
    )
    p.add_argument(
        '--visualize', action='store_true',
        help="overlay sparse points (and depth values) on original image"
    )
    args = p.parse_args()

    # Original image size (in pixels) and normalized size
    orig_w, orig_h = 968, 608
    norm_w, norm_h = 320, 240

    # Center‐row (mean of the normal distribution)
    cy = orig_h // 2

    # Which x‐coordinates to sample (0, spacing, 2*spacing, ...)
    cols = np.arange(0, orig_w, args.spacing, dtype=int)

    out_lines = []
    pattern = f"sonar_{args.spacing}_pixels"

    with open(args.input_txt) as f:
        for L in f:
            img_p, depth_p, *_ = L.strip().split()
            depth = np.array(Image.open(depth_p))  # H×W depth map (uint16/float/etc.)

            samples = []  # will hold tuples: (yi, xi, ni, nj, d)
            for xi in cols:
                # 1) draw a vertical sample position around cy
                yi = int(round(np.random.normal(loc=cy, scale=args.std)))
                yi = np.clip(yi, 0, orig_h - 1)

                d = float(depth[yi, xi])
                if d <= 0:
                    # skip invalid or zero depth
                    continue

                # 2) normalize coordinates to (norm_h × norm_w) grid
                ni = int(round(yi * (norm_h / orig_h)))
                nj = int(round(xi * (norm_w / orig_w)))
                samples.append((yi, xi, ni, nj, d))

            if len(samples) < 4:
                # if too few valid samples, skip this frame
                continue

            # create output directory for this spacing‐pattern
            imgs_dir = os.path.dirname(img_p)
            base_dir = os.path.dirname(imgs_dir)
            out_dir = os.path.join(base_dir, pattern)
            os.makedirs(out_dir, exist_ok=True)

            base = os.path.splitext(os.path.basename(img_p))[0]
            out_csv = os.path.join(
                out_dir,
                f"{base}_sparse_depth_{pattern}.csv"
            )
            # write CSV: header + rows [row, column, depth]
            with open(out_csv, 'w', newline='') as cf:
                w = csv.writer(cf)
                w.writerow(['row', 'column', 'depth'])
                for yi, xi, ni, nj, d in samples:
                    w.writerow([ni, nj, d])

            out_lines.append(f"{img_p} {depth_p} {out_csv}")

            if args.visualize:
                img = np.array(Image.open(img_p))
                xs = [xi for yi, xi, *_ in samples]
                ys = [yi for yi, xi, *_ in samples]
                ds = [d for yi, xi, ni, nj, d in samples]

                plt.figure(figsize=(6, 4))
                plt.imshow(img)
                plt.scatter(xs, ys, c='r', marker='x')
                for xpt, ypt, val in zip(xs, ys, ds):
                    plt.text(xpt + 2, ypt + 2, f"{val:.2f}",
                             fontsize=8, color='yellow')
                plt.title(f"{base}   {pattern}")
                plt.axis('off')
                plt.show()

    # write out the filtered list of (rgb_path, depth_path, sparse_csv)
    with open(args.output_txt, 'w') as tf:
        tf.write("\n".join(out_lines))


if __name__ == '__main__':
    main()
