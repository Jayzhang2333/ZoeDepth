import numpy as np
import matplotlib.pyplot as plt

def plot_grid_and_sparse_filtered(grid_npy_path, sparse_npy_path):
    """
    Load two .npy files and plot:
      - Full-grid points (normalized coords) in red, excluding border points
      - Sparse depth points colored by depth using viridis
    
    Args:
        grid_npy_path (str): path to .npy of shape (H, W, 2) with normalized (y,x) coords
        sparse_npy_path (str): path to .npy of shape (N, 3) with [x_norm, y_norm, depth]
    """
    # Load and flatten full-grid coords
    grid = np.load(grid_npy_path)            # shape (H, W, 2)
    grid_pts = grid.reshape(-1, 2)           # (H*W, 2)
    ys_grid, xs_grid = grid_pts[:, 0], grid_pts[:, 1]
    
    # Filter out any border points (x or y == -1 or 1)
    mask = (xs_grid > -1) & (xs_grid < 1) & (ys_grid > -1) & (ys_grid < 1)
    xs_grid_f = xs_grid[mask]
    ys_grid_f = ys_grid[mask]
    
    # Load sparse points
    sparse = np.load(sparse_npy_path)        # shape (N, 3)
    xs_sp, ys_sp, depths = sparse[:, 0], sparse[:, 1], sparse[:, 2]
    
    # Plot
    fig, ax = plt.subplots(figsize=(6,6))
    # Filtered grid points in red
    ax.scatter(xs_grid_f, ys_grid_f, s=10, c='red', label='grid points')
    # Sparse depth points with viridis colormap
    sc = ax.scatter(xs_sp, ys_sp, c=depths, cmap='viridis', s=30, label='sparse depth')
    
    # Axes setup
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.invert_yaxis()
    ax.set_xlabel('x (normalized)')
    ax.set_ylabel('y (normalized)')
    ax.set_title('Grid & Sparse Depth Points (filtered)')
    ax.legend(loc='upper right')
    
    # Colorbar for depth
    plt.colorbar(sc, ax=ax, label='depth value')
    
    plt.tight_layout()
    plt.show()

# === Usage ===
plot_grid_and_sparse_filtered("./edited_grid3.npy", "/home/jay/ZoeDepth/deformable_showcase_sparse_depths.npy")
