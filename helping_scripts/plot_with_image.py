import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# def plot_grid_and_sparse_on_image(grid_npy_path, sparse_npy_path, image_path):
#     """
#     Load grid coords, sparse depth, and an image, then plot:
#       - Background image resized to 336×448 (height×width).
#       - Shade the 16×16 pixel block centered at each grid coord (normalized [-1,1]).
#       - Overlay grid coord centers in red.
#       - Overlay sparse depth points colored by depth (viridis).

#     Args:
#         grid_npy_path (str): .npy of shape (Hg, Wg, 2) with normalized (y,x).
#         sparse_npy_path (str): .npy of shape (N,3) with [x_norm, y_norm, depth].
#         image_path (str): Path to the background image.
#     """
#     # --- Load & filter grid points ---
#     grid = np.load(grid_npy_path)            # (Hg, Wg, 2)
#     pts = grid.reshape(-1, 2)                # (Hg*Wg, 2)
#     ys_norm, xs_norm = pts[:, 0], pts[:, 1]
#     valid = (xs_norm > -1) & (xs_norm < 1) & (ys_norm > -1) & (ys_norm < 1)
#     xs_norm, ys_norm = xs_norm[valid], ys_norm[valid]

#     # --- Load & resize image to 448×336 (width×height) ---
#     img = Image.open(image_path).convert('RGB')
#     img = img.resize((448, 336), Image.BILINEAR)
#     img_np = np.array(img)
#     H_img, W_img = img_np.shape[:2]

#     # --- Map normalized coords to pixel positions ---
#     x_pix = (xs_norm + 1) * 0.5 * (W_img - 1)
#     y_pix = (ys_norm + 1) * 0.5 * (H_img - 1)

#     # --- Create shading mask for 16×16 block centered at each grid point ---
#     mask = np.zeros((H_img, W_img), dtype=np.uint8)
#     k = 16
#     half_k = k // 2
#     for x, y in zip(x_pix, y_pix):
#         i_center = int(round(y))
#         j_center = int(round(x))
#         start_i = i_center - half_k
#         start_j = j_center - half_k
#         for di in range(k):
#             for dj in range(k):
#                 ii = np.clip(start_i + di, 0, H_img - 1)
#                 jj = np.clip(start_j + dj, 0, W_img - 1)
#                 mask[ii, jj] = 1

#     # --- Load sparse depth points ---
#     sparse = np.load(sparse_npy_path)        # (N,3)
#     xs_s, ys_s, depths = sparse[:, 0], sparse[:, 1], sparse[:, 2]
#     x_s_pix = (xs_s + 1) * 0.5 * (W_img - 1)
#     y_s_pix = (ys_s + 1) * 0.5 * (H_img - 1)

#     # --- Plot everything ---
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.imshow(img_np, origin='upper')
#     ax.imshow(mask, cmap='Reds', alpha=0.4, origin='upper')
#     ax.scatter(x_pix, y_pix, c='red', s=20, label='grid coords')
#     sc = ax.scatter(x_s_pix, y_s_pix, c=depths, cmap='viridis', s=30, label='sparse depth')
    
#     ax.set_xlim(0, W_img - 1)
#     ax.set_ylim(H_img - 1, 0)
#     ax.axis('off')
#     ax.legend(loc='upper right')
#     plt.colorbar(sc, ax=ax, label='depth')
#     plt.tight_layout()
#     plt.show()


def plot_grid_and_sparse_on_image(grid_npy_path, sparse_npy_path, image_path):
    """
    Load grid coords, sparse depth, and an image, then plot:
      - Background image resized to 336×448 (height×width).
      - Overlay sparse depth points in yellow.
      - Overlay grid coord centers in red on top, both with equal size.
      - Legend labels are bold and larger font.

    Args:
        grid_npy_path (str): .npy of shape (Hg, Wg, 2) with normalized (y,x).
        sparse_npy_path (str): .npy of shape (N,3) with [x_norm, y_norm, depth].
        image_path (str): Path to the background image.
    """
    # --- Load & filter grid points ---
    grid = np.load(grid_npy_path)            # (Hg, Wg, 2)
    pts = grid.reshape(-1, 2)                # (Hg*Wg, 2)
    ys_norm, xs_norm = pts[:, 0], pts[:, 1]
    valid = (xs_norm > -1) & (xs_norm < 1) & (ys_norm > -1) & (ys_norm < 1)
    xs_norm, ys_norm = xs_norm[valid], ys_norm[valid]

    # --- Load & resize image to 448×336 (width×height) ---
    img = Image.open(image_path).convert('RGB')
    img = img.resize((448, 336), Image.BILINEAR)
    img_np = np.array(img)
    H_img, W_img = img_np.shape[:2]

    # --- Map normalized coords to pixel positions ---
    x_pix = (xs_norm + 1) * 0.5 * (W_img - 1)
    y_pix = (ys_norm + 1) * 0.5 * (H_img - 1)

    # --- Load sparse depth points ---
    sparse = np.load(sparse_npy_path)        # (N,3)
    xs_s, ys_s = sparse[:, 0], sparse[:, 1]
    x_s_pix = (xs_s + 1) * 0.5 * (W_img - 1)
    y_s_pix = (ys_s + 1) * 0.5 * (H_img - 1)

    # --- Plot everything ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_np, origin='upper')

    point_size = 30  # same size for both

    # sparse depth points first (yellow)
    ax.scatter(x_s_pix, y_s_pix, c='yellow', s=point_size, label='Sparse Depth Points')
    # grid centers on top (red)
    ax.scatter(x_pix, y_pix, c='red', s=point_size, label='Deformed Sample Location')

    ax.set_xlim(0, W_img - 1)
    ax.set_ylim(H_img - 1, 0)
    ax.axis('off')

    # legend with bold, larger text
    ax.legend(loc='upper right', prop={'size': 14, 'weight': 'bold'})

    plt.tight_layout()
    plt.show()
# === Usage ===
plot_grid_and_sparse_on_image("./top_left_deform_demo.npy", "/home/jay/ZoeDepth/top_left_edited_grid.npy", "/media/jay/Marine/FLSea_test/FLSea_latest/archive/red_sea/sub_pier/sub_pier/imgs/16316026402975063.tiff")
