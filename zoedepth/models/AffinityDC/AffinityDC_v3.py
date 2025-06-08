# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import itertools
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from zoedepth.models.DepthAnything.dpt import DPTHead, DPTHead_customised, DPTHeadCustomised3
from zoedepth.models.DepthAnything.dinov2 import DINOv2
from zoedepth.models.DepthAnything.blocks import ScaleResidualBlock, ScaleResidualOutputBlock
from zoedepth.models.depth_model import DepthModel
from zoedepth.models.midas.sml import SMLDeformableAttention
from zoedepth.models.model_io import load_state_from_resource
import torch.nn.functional as F
import os
import wandb
from zoedepth.models.layers.global_alignment import LeastSquaresEstimatorTorch, AnchorInterpolator2D
from zoedepth.models.layers.spn import CSPNAccelerate, LightweightAffinityNet9, compute_affinity_map_9, DepthGradientAffinity9



model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], 'layer_idxs': [2, 5, 8, 11]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768], 'layer_idxs': [2, 5, 8, 11]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'layer_idxs': [4, 11, 17, 23]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536], 'layer_idxs': [9, 19, 29, 39]}
}
import matplotlib.gridspec as gridspec
# def display_images(source1, source2, source3, source4):
#     """
#     Display images from four sources in a 2x2 grid:
#         - Top-left: RGB image.
#         - Top-right: Overlay image: RGB background with sparse points overlaid
#                     (points come from the sparse map and are colored by metric depth).
#         - Bottom-left: Relative depth map.
#         - Bottom-right: Metric depth map.

#     Each of the overlay image (top-right) and metric depth map (bottom-right)
#     has an associated colorbar that is set to 80% of the image height and centered vertically.

#     Args:
#         source1: RGB image tensor/array with shape (N, C, H, W)
#         source2: Sparse map tensor/array with shape (N, 1, H, W)
#         source3: Relative depth map tensor/array with shape (N, 1, H, W)
#         source4: Metric depth map tensor/array with shape (N, 1, H, W)
#     """
#     def preprocess_images(source):
#         # Convert to numpy if not already, and move channels from (N, C, H, W) to (N, H, W, C)
#         if not isinstance(source, np.ndarray):
#             source = source.detach().cpu().numpy()
#         return np.transpose(source, (0, 2, 3, 1))
    
#     # Preprocess each source.
#     images1 = preprocess_images(source1)  # RGB image, shape: (N, H, W, 3)
#     images2 = preprocess_images(source2)  # Sparse map, shape: (N, H, W, 1)
#     images3 = preprocess_images(source3)  # Relative depth map, shape: (N, H, W, 1)
#     images4 = preprocess_images(source4)  # Metric depth map, shape: (N, H, W, 1)

#     # For simplicity, assume batch size = 1
#     rgb_img      = images1[0]          # (H, W, 3)
#     sparse_map   = images2[0, :, :, 0]   # (H, W)
#     rel_depth    = images3[0, :, :, 0]   # (H, W)
#     metric_depth = images4[0, :, :, 0]   # (H, W)

#     # Create a figure using GridSpec with 2 rows and 3 columns.
#     # Columns 0 and 1 hold the images; column 2 holds the colorbar.
#     # This is done for each row separately.
#     fig = plt.figure(figsize=(12, 10))
#     gs = gridspec.GridSpec(nrows=2, ncols=3,
#                             width_ratios=[1, 1, 0.05],
#                             wspace=0.0, hspace=0.0)

#     # ------------------ Row 0 ------------------
#     # Top-left: RGB image.
#     ax00 = fig.add_subplot(gs[0, 0])
#     ax00.imshow(rgb_img)
#     ax00.set_title('(a) RGB')
#     ax00.axis('off')

#     # Top-right: Overlay image: show RGB background with scatter overlay.
#     ax01 = fig.add_subplot(gs[0, 1])
#     ax01.imshow(rgb_img)
#     # Find nonzero positions in the sparse map.
#     rows, cols = np.nonzero(sparse_map)
#     # Color the scatter points using the metric depth values.
#     scatter_vals = metric_depth[sparse_map != 0]
#     sc = ax01.scatter(cols, rows, c=scatter_vals, cmap='viridis_r', s=20)
#     ax01.set_title('(b) Sparse Depth Measurement')
#     ax01.axis('off')

#     # Colorbar for the overlay image.
#     cax0 = fig.add_subplot(gs[0, 2])
#     # Adjust colorbar axes: set height to 80% of ax01's height and center vertically.
#     pos_ax01 = ax01.get_position()
#     pos_cax0 = cax0.get_position()
#     new_height = pos_ax01.height * 0.8
#     new_y = pos_ax01.y0 + (pos_ax01.height - new_height) / 2
#     cax0.set_position([pos_ax01.x1, new_y, pos_cax0.width, new_height])
#     plt.colorbar(sc, cax=cax0)

#     # ------------------ Row 1 ------------------
#     # Bottom-left: Relative depth map.
#     ax10 = fig.add_subplot(gs[1, 0])
#     ax10.imshow(rel_depth, cmap='viridis')
#     ax10.set_title('(c) Relative Depth')
#     ax10.axis('off')

#     # Bottom-right: Metric depth map.
#     ax11 = fig.add_subplot(gs[1, 1])
#     im = ax11.imshow(metric_depth, cmap='viridis_r')
#     ax11.set_title('(d) Our Metric Prediction')
#     ax11.axis('off')

#     # Colorbar for the metric depth map.
#     cax1 = fig.add_subplot(gs[1, 2])
#     pos_ax11 = ax11.get_position()
#     pos_cax1 = cax1.get_position()
#     new_height = pos_ax11.height * 0.8
#     new_y = pos_ax11.y0 + (pos_ax11.height - new_height) / 2
#     cax1.set_position([pos_ax11.x1, new_y, pos_cax1.width, new_height])
#     plt.colorbar(im, cax=cax1)

#     plt.show()

def plot_sparse_depth_tensor(sparse_tensor, batch_idx=0, save_path=None):
    """
    Plot non-zero points from a [B, 1, H, W] sparse depth tensor in normalized [-1,1] coords.
    Top-left = (-1,-1), bottom-right = (1,1). Colors indicate depth values.
    Optionally save the (x, y, depth) points to a .npy file.

    Args:
        sparse_tensor (torch.Tensor): shape [B,1,H,W], invalid entries are zero.
        batch_idx (int): index of batch to plot if B > 1 (default 0).
        save_path (str, optional): path to save the points as a .npy file.
    """
    # Validate input shape
    if sparse_tensor.ndim != 4 or sparse_tensor.shape[1] != 1:
        raise ValueError(f"Expected tensor shape [B,1,H,W], got {tuple(sparse_tensor.shape)}")
    B, _, H, W = sparse_tensor.shape

    # Select batch slice
    if B > 1:
        if not (0 <= batch_idx < B):
            raise IndexError(f"batch_idx must be in [0, {B-1}], got {batch_idx}")
        sample = sparse_tensor[batch_idx, 0]
    else:
        sample = sparse_tensor[0, 0]

    # Move to CPU
    sample_cpu = sample.detach().cpu()

    # Find non-zero entries
    rows, cols = torch.nonzero(sample_cpu, as_tuple=True)
    if rows.numel() == 0:
        print("No non-zero points to plot.")
        return

    # Normalize coordinates to [-1,1]
    xs = (cols.float() / (W - 1)) * 2 - 1
    ys = (rows.float() / (H - 1)) * 2 - 1
    vals = sample_cpu[rows, cols]

    # Convert to numpy for plotting
    xs_np = xs.numpy()
    ys_np = ys.numpy()
    vals_np = vals.numpy()

    # Optionally save to .npy
    if save_path is not None:
        points = np.stack([xs_np, ys_np, vals_np], axis=1)  # shape (N, 3)
        np.save(save_path, points)
        print(f"Saved {points.shape[0]} points to {save_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(6,6))
    sc = ax.scatter(xs_np, ys_np, c=vals_np, cmap='viridis', s=25)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.invert_yaxis()
    ax.set_xlabel('x (normalized)')
    ax.set_ylabel('y (normalized)')
    ax.set_title(f'Sparse Depth Map (batch {batch_idx})')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    plt.colorbar(sc, ax=ax, label='depth value')
    plt.tight_layout()
    plt.show()

def display_and_save_images(
    source1, source2, source3, source4,
    save_path: str,
    title_fontsize=16,
    title_fontweight='bold',
    dpi=300
):
    """
    Displays and saves a 1×2 layout:
      - Left: Overlay image (RGB + sparse points colored by metric depth)
      - Right: Metric depth map

    White background is made transparent in the saved PNG.

    Args:
        source1, source2, source3, source4: same as before
        save_path: where to write the transparent PNG
        title_fontsize, title_fontweight: title styling
        dpi: resolution for saving
    """
    def preprocess_images(source):
        if not isinstance(source, np.ndarray):
            source = source.detach().cpu().numpy()
        return np.transpose(source, (0, 2, 3, 1))
    
    # preprocess and unpack
    rgb_img      = preprocess_images(source1)[0]
    sparse_map   = preprocess_images(source2)[0, :, :, 0]
    metric_depth = preprocess_images(source4)[0, :, :, 0]

    # set up figure + grid
    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(1, 5, width_ratios=[1, .05, .01, 1, .05], wspace=0.1)

    # Left: overlay
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(rgb_img.astype(np.uint8))
    ys, xs = np.nonzero(sparse_map)
    sc = ax0.scatter(
        xs, ys,
        c=metric_depth[sparse_map != 0],
        cmap='viridis_r',
        s=20, vmin=0.1, vmax=10
    )
    ax0.set_title('Input:RGB Image + Sparse Depth Points', fontsize=title_fontsize, fontweight=title_fontweight)
    ax0.axis('off')
    cax0 = fig.add_subplot(gs[0, 1])
    pos0 = ax0.get_position()
    h0 = pos0.height * 0.9
    y0 = pos0.y0 + (pos0.height - h0)/2
    cax0.set_position([pos0.x1, y0, cax0.get_position().width, h0])
    cb0 = plt.colorbar(sc, cax=cax0)
    cb0.ax.tick_params(labelsize=12)

    # Right: metric depth
    ax2 = fig.add_subplot(gs[0, 3])
    im2 = ax2.imshow(metric_depth, cmap='viridis_r', vmin=0.1, vmax=10)
    ax2.set_title('Metric Depth', fontsize=title_fontsize, fontweight=title_fontweight)
    ax2.axis('off')
    cax1 = fig.add_subplot(gs[0, 4])
    pos2 = ax2.get_position()
    h1 = pos2.height * 0.9
    y1 = pos2.y0 + (pos2.height - h1)/2
    cax1.set_position([pos2.x1, y1, cax1.get_position().width, h1])
    cb1 = plt.colorbar(im2, cax=cax1)
    cb1.ax.tick_params(labelsize=12)

    # Make backgrounds transparent
    fig.patch.set_facecolor('none')
    for ax in fig.axes:
        ax.patch.set_facecolor('none')

    # Save with transparent background
    fig.savefig(
        save_path,
        transparent=True,
        bbox_inches='tight',
        dpi=dpi
    )
    plt.close(fig)
    print(f"Saved transparent figure to {save_path}")

def display_images(
    source1, source2, source3, source4,
    title_fontsize=16,
    title_fontweight='bold'
):
    """
    Display only the overlay image and the metric depth map side-by-side in a single row with a spacer:
      - Left: Overlay image (RGB background with scatter overlay from the sparse map,
              where the scatter points are colored by metric depth).
      - Right: Metric depth map.

    Each image has an associated colorbar (set to 80% of the image height and centered vertically).

    Args:
        source1: RGB image tensor/array with shape (N, C, H, W)
        source2: Sparse map tensor/array with shape (N, 1, H, W)
        source3: Relative depth map tensor/array with shape (N, 1, H, W) [not used here]
        source4: Metric depth map tensor/array with shape (N, 1, H, W)
        title_fontsize: Font size for the subplot titles.
        title_fontweight: Font weight for the subplot titles.
    """
    def preprocess_images(source):
        if not isinstance(source, np.ndarray):
            source = source.detach().cpu().numpy()
        return np.transpose(source, (0, 2, 3, 1))
    
    images1 = preprocess_images(source1)
    images2 = preprocess_images(source2)
    images4 = preprocess_images(source4)

    rgb_img      = images1[0]
    sparse_map   = images2[0, :, :, 0]
    metric_depth = images4[0, :, :, 0]

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(
        nrows=1, ncols=5,
        width_ratios=[1, 0.05, 0.01, 1, 0.05],
        wspace=0.1
    )

    # Overlay image
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(rgb_img)
    rows, cols = np.nonzero(sparse_map)
    sc = ax0.scatter(
        cols, rows,
        c=metric_depth[sparse_map != 0],
        cmap='viridis_r',
        s=20
    )
    ax0.set_title(
        'Overlay Image',
        fontsize=title_fontsize,
        fontweight=title_fontweight
    )
    ax0.axis('off')

    # Colorbar for overlay
    cax0 = fig.add_subplot(gs[0, 1])
    pos = ax0.get_position()
    height = pos.height * 0.9
    y = pos.y0 + (pos.height - height) / 2
    cax0.set_position([pos.x1, y, cax0.get_position().width, height])
    cb0 = plt.colorbar(sc, cax=cax0)
    cb0.ax.tick_params(labelsize=12)

    # Metric depth map
    ax2 = fig.add_subplot(gs[0, 3])
    im = ax2.imshow(metric_depth, cmap='viridis_r')
    ax2.set_title(
        'Metric Depth',
        fontsize=title_fontsize,
        fontweight=title_fontweight
    )
    ax2.axis('off')

    # Colorbar for metric depth
    cax1 = fig.add_subplot(gs[0, 4])
    pos2 = ax2.get_position()
    height2 = pos2.height * 0.9
    y2 = pos2.y0 + (pos2.height - height2) / 2
    cax1.set_position([pos2.x1, y2, cax1.get_position().width, height2])
    cb1 = plt.colorbar(im, cax=cax1)
    cb1.ax.tick_params(labelsize=12)

    plt.show()

def display_images_veritial(source1, source2, source3, source4):
    """
    Display the overlay image and the metric depth map in two rows, each with its associated colorbar:
        - Top row: Overlay image (RGB background with scatter overlay from the sparse map,
                   where the scatter points are colored by metric depth).
        - Bottom row: Metric depth map.

    Each image has an associated colorbar (set to 80% of the image height and centered vertically).

    Args:
        source1: RGB image tensor/array with shape (N, C, H, W)
        source2: Sparse map tensor/array with shape (N, 1, H, W)
        source3: Relative depth map tensor/array with shape (N, 1, H, W) [not used here]
        source4: Metric depth map tensor/array with shape (N, 1, H, W)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    def preprocess_images(source):
        # Convert to numpy if not already, and move channels from (N, C, H, W) to (N, H, W, C)
        if not isinstance(source, np.ndarray):
            source = source.detach().cpu().numpy()
        return np.transpose(source, (0, 2, 3, 1))
    
    # Preprocess each source.
    images1 = preprocess_images(source1)  # RGB image, shape: (N, H, W, 3)
    images2 = preprocess_images(source2)  # Sparse map, shape: (N, H, W, 1)
    images4 = preprocess_images(source4)  # Metric depth map, shape: (N, H, W, 1)

    # For simplicity, assume batch size = 1.
    rgb_img      = images1[0]            # (H, W, 3)
    sparse_map   = images2[0, :, :, 0]     # (H, W)
    metric_depth = images4[0, :, :, 0]     # (H, W)

    # Create a figure with 2 rows and 2 columns:
    # Column 0: Image, Column 1: its colorbar.
    fig = plt.figure(figsize=(8, 12))
    gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[1, 0.05], hspace=0.3)

    # ------------------ Top Row: Overlay Image ------------------
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(rgb_img)
    # Find nonzero positions in the sparse map.
    rows, cols = np.nonzero(sparse_map)
    # Use the corresponding metric depth values for coloring.
    scatter_vals = metric_depth[sparse_map != 0]
    sc = ax0.scatter(cols, rows, c=scatter_vals, cmap='viridis_r', s=20)
    ax0.set_title('Overlay Image')
    ax0.axis('off')

    # Colorbar for the overlay image.
    cax0 = fig.add_subplot(gs[0, 1])
    pos_ax0 = ax0.get_position()
    pos_cax0 = cax0.get_position()
    new_height = pos_ax0.height * 0.9
    new_y = pos_ax0.y0 + (pos_ax0.height - new_height) / 2
    cax0.set_position([pos_ax0.x1, new_y, pos_cax0.width, new_height])
    plt.colorbar(sc, cax=cax0)

    # ------------------ Bottom Row: Metric Depth Map ------------------
    ax1 = fig.add_subplot(gs[1, 0])
    im = ax1.imshow(metric_depth, cmap='viridis_r')
    ax1.set_title('Metric Depth')
    ax1.axis('off')

    # Colorbar for the metric depth map.
    cax1 = fig.add_subplot(gs[1, 1])
    pos_ax1 = ax1.get_position()
    pos_cax1 = cax1.get_position()
    new_height = pos_ax1.height * 0.9
    new_y = pos_ax1.y0 + (pos_ax1.height - new_height) / 2
    cax1.set_position([pos_ax1.x1, new_y, pos_cax1.width, new_height])
    plt.colorbar(im, cax=cax1)

    plt.show()

def show_images(tensor_images):
    tensor_images = tensor_images.detach().cpu().numpy()  # Convert to numpy if tensor
    tensor_images = np.transpose(tensor_images, (0, 2, 3, 1))  # Change from CHW to HWC
    
    # Display the images
    batch_size = tensor_images.shape[0]
    fig, axes = plt.subplots(1, batch_size, figsize=(15, 5))  # Adjust number of subplots as needed
    if batch_size == 1:  # Handle case where batch size is 1
        axes = [axes]
    
    for idx in range(batch_size):
        im = axes[idx].imshow(tensor_images[idx], cmap='inferno')
        axes[idx].axis('off')
        # Add a colorbar to the current subplot
        fig.colorbar(im, ax=axes[idx])
        
    
    plt.show()
from matplotlib.colors import LogNorm
def show_images_three_sources(source1, source2, source3):
    """
    Display images from four sources side by side in a row.
    
    Args:
        source1, source2, source3, source4: Four input tensors or numpy arrays of images.
                                            Expected shapes: (N, C, H, W) for tensors.
    """
    def preprocess_images(source):
        if isinstance(source, np.ndarray):
            images = source
        else:  # Assume tensor
            images = source.detach().cpu().numpy()
        images = np.transpose(images, (0, 2, 3, 1))  # Convert from CHW to HWC
        return images

    # Preprocess all four sources
    images1 = preprocess_images(source1)
    images2 = preprocess_images(source2)
    images3 = preprocess_images(source3)
    
    # Ensure the batch sizes match
    assert images1.shape[0] == images2.shape[0] == images3.shape[0] , "Batch sizes must match."
    
    batch_size = images1.shape[0]
    fig, axes = plt.subplots(batch_size, 3, figsize=(20, 5 * batch_size))  # 4 columns for sources
    
    if batch_size == 1:  # Handle case where batch size is 1
        axes = [axes]
    
    # for idx in range(batch_size):
    #     # Display source1 image
    #     axes[idx][0].imshow(images1[idx])
    #     axes[idx][0].set_title("Source 1")
    #     axes[idx][0].axis('off')
        
    #     # Display source2 image
    #     axes[idx][1].imshow(images2[idx])
    #     axes[idx][1].set_title("Source 2")
    #     axes[idx][1].axis('off')
        
    #     # Display source3 image
    #     axes[idx][2].imshow(images3[idx])
    #     axes[idx][2].set_title("Source 3")
    #     axes[idx][2].axis('off')

    for idx in range(batch_size):
        # Display source1 image
        im1 = axes[idx][0].imshow(images1[idx],cmap='viridis')
        axes[idx][0].set_title("Source 1")
        axes[idx][0].axis('off')
        plt.colorbar(im1, ax=axes[idx][0], fraction=0.046, pad=0.04)

        # Display source2 image
        #,cmap='Spectral',vmin=0.1, vmax=15
        im2 = axes[idx][1].imshow(images2[idx],cmap='viridis')
        axes[idx][1].set_title("Source 2")
        axes[idx][1].axis('off')
        plt.colorbar(im2, ax=axes[idx][1], fraction=0.046, pad=0.04)

        # Display source3 image
        im3 = axes[idx][2].imshow(images3[idx],cmap='viridis_r')
        axes[idx][2].set_title("Source 3")
        axes[idx][2].axis('off')
        plt.colorbar(im3, ax=axes[idx][2], fraction=0.046, pad=0.04)
        
    
    plt.tight_layout()
    plt.show()


def show_images_four_sources(source1, source2, source3, source4):
    """
    Display images from four sources side by side in a row.
    
    Args:
        source1, source2, source3, source4: Four input tensors or numpy arrays of images.
                                            Expected shapes: (N, C, H, W) for tensors.
    """
    def preprocess_images(source):
        if isinstance(source, np.ndarray):
            images = source
        else:  # Assume tensor
            images = source.detach().cpu().numpy()
        images = np.transpose(images, (0, 2, 3, 1))  # Convert from CHW to HWC
        return images

    # Preprocess all four sources
    images1 = preprocess_images(source1)
    images2 = preprocess_images(source2)
    images3 = preprocess_images(source3)
    images4 = preprocess_images(source4)
    
    # Ensure the batch sizes match
    assert images1.shape[0] == images2.shape[0] == images3.shape[0] == images4.shape[0], "Batch sizes must match."
    
    batch_size = images1.shape[0]
    fig, axes = plt.subplots(batch_size, 4, figsize=(20, 5 * batch_size))  # 4 columns for sources
    
    if batch_size == 1:  # Handle case where batch size is 1
        axes = [axes]
    
    for idx in range(batch_size):
        # Display source1 image
        im1 = axes[idx][0].imshow(images1[idx])
        axes[idx][0].set_title("Source 1")
        axes[idx][0].axis('off')
        plt.colorbar(im1, ax=axes[idx][0], fraction=0.046, pad=0.04)

        # Display source2 image
        im2 = axes[idx][1].imshow(images2[idx])
        axes[idx][1].set_title("Source 2")
        axes[idx][1].axis('off')
        plt.colorbar(im2, ax=axes[idx][1], fraction=0.046, pad=0.04)

        # Display source3 image
        im3 = axes[idx][2].imshow(images3[idx])
        axes[idx][2].set_title("Source 3")
        axes[idx][2].axis('off')
        plt.colorbar(im3, ax=axes[idx][2], fraction=0.046, pad=0.04)

        # Display source4 image
        im4 = axes[idx][3].imshow(images4[idx])
        axes[idx][3].set_title("Source 4")
        axes[idx][3].axis('off')
        plt.colorbar(im4, ax=axes[idx][3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()


def resize_sparse_depth(depth, new_sizes):
    """
    Resizes a sparse depth tensor to multiple new sizes while preserving the sparse measurements.
    
    For each valid measurement in the original tensor (depth > 0), it computes
    its normalized spatial coordinates and maps it to a new output grid.
    If multiple valid measurements land in the same location in the new grid,
    their values are averaged.
    
    Parameters:
        depth (torch.Tensor): Input tensor of shape (B, 1, H, W) with sparse depth measurements.
        new_sizes (list): List of desired output spatial sizes. Each element can be a tuple (new_H, new_W)
                          or an int (for a square output where new_H = new_W).
    
    Returns:
        list of torch.Tensor: List of resized depth tensors, one per new size.
    """
    outputs = []
    B, C, H, W = depth.shape
    
    # Precompute the original coordinate grid (shared across new sizes)
    ys_orig = torch.arange(H, device=depth.device).view(H, 1).expand(H, W).flatten()  # row indices
    xs_orig = torch.arange(W, device=depth.device).view(1, W).expand(H, W).flatten()  # column indices

    for new_size in new_sizes:
        if isinstance(new_size, int):
            new_H, new_W = new_size, new_size
        else:
            new_H, new_W = new_size

        # Prepare tensors to accumulate sums and counts for averaging
        sum_tensor = torch.zeros((B, 1, new_H, new_W), device=depth.device, dtype=depth.dtype)
        count_tensor = torch.zeros((B, 1, new_H, new_W), device=depth.device, dtype=depth.dtype)

        # Map the original pixel indices to the new grid coordinates for this new size
        new_ys = (ys_orig.float() / H * new_H).floor().long().clamp(0, new_H - 1)
        new_xs = (xs_orig.float() / W * new_W).floor().long().clamp(0, new_W - 1)

        # Loop over each item in the batch
        for b in range(B):
            # Flatten the depth values for the current image
            depth_flat = depth[b, 0].flatten()
            valid = depth_flat > 0  # valid measurement mask
            valid_depth = depth_flat[valid]
            valid_new_ys = new_ys[valid]
            valid_new_xs = new_xs[valid]
            # Compute flattened indices for the new grid (row-major order)
            idx = valid_new_ys * new_W + valid_new_xs

            # Flatten the corresponding slice of the output tensors
            sum_flat = sum_tensor[b, 0].view(-1)
            count_flat = count_tensor[b, 0].view(-1)
            # Scatter-add the valid depth values
            sum_flat.scatter_add_(0, idx, valid_depth)
            # Scatter-add counts
            ones = torch.ones_like(valid_depth)
            count_flat.scatter_add_(0, idx, ones)

        # Compute the final averaged output for this new size
        output = torch.zeros_like(sum_tensor)
        mask = count_tensor > 0
        output[mask] = sum_tensor[mask] / count_tensor[mask]
        outputs.append(output)
    
    return outputs

def show_images_two_sources(
    source1, source2,
    title_fontsize: int = 16,
    title_fontweight: str = 'bold'
):
    """
    Display images from two sources side by side in a row, with bold, larger titles.
    
    Args:
        source1, source2: Two input tensors or numpy arrays of images.
                          Expected shapes: (N, C, H, W) for tensors.
        title_fontsize: Font size for titles.
        title_fontweight: Font weight for titles.
    """
    def preprocess_images(source):
        if isinstance(source, np.ndarray):
            images = source
        else:  # Assume tensor
            images = source.detach().cpu().numpy()
        # Convert from (N, C, H, W) to (N, H, W, C)
        return np.transpose(images, (0, 2, 3, 1))

    images1 = preprocess_images(source1)
    images2 = preprocess_images(source2)
    assert images1.shape[0] == images2.shape[0], "Batch sizes must match."

    batch_size = images1.shape[0]
    fig, axes = plt.subplots(batch_size, 2, figsize=(10, 5 * batch_size))

    # If batch_size == 1, wrap axes in a list for uniform indexing
    if batch_size == 1:
        axes = [axes]

    for idx in range(batch_size):
        # Source 1: RGB image
        ax1 = axes[idx][0]
        ax1.imshow(images1[idx].astype(np.uint8))
        ax1.set_title(
            "RGB Image",
            fontsize=title_fontsize,
            fontweight=title_fontweight
        )
        ax1.axis('off')

        # Source 2: Relative depth prediction
        ax2 = axes[idx][1]
        im2 = ax2.imshow(images2[idx], cmap='viridis')
        ax2.set_title(
            "Relative Depth Prediction",
            fontsize=title_fontsize,
            fontweight=title_fontweight
        )
        ax2.axis('off')

    plt.tight_layout()
    plt.show()


def save_images_two_sources(source1, source2, save_dir='.', prefix='figure'):
    """
    Save images from two sources side by side in a row with an incrementing index.
    
    Args:
        source1, source2: Two input tensors or numpy arrays of images.
                          Expected shapes: (N, C, H, W) for tensors.
        save_dir (str): Directory where figures will be saved.
        prefix (str): Filename prefix for saved figures.
    """
    # Initialize the counter attribute on first call
    if not hasattr(save_images_two_sources, 'counter'):
        save_images_two_sources.counter = 0

    def preprocess_images(source):
        if isinstance(source, np.ndarray):
            images = source
        else:  # Assume tensor
            images = source.detach().cpu().numpy()
        return np.transpose(images, (0, 2, 3, 1))  # CHW -> HWC

    # Create output directory if needed
    os.makedirs(save_dir, exist_ok=True)

    # Preprocess both sources
    images1 = preprocess_images(source1)
    images2 = preprocess_images(source2)

    # Ensure batch sizes match
    assert images1.shape[0] == images2.shape[0], "Batch sizes must match."

    batch_size = images1.shape[0]
    fig, axes = plt.subplots(batch_size, 2, figsize=(10, 5 * batch_size))

    if batch_size == 1:
        axes = [axes]

    for idx in range(batch_size):
        axes[idx][0].imshow(images1[idx])
        axes[idx][0].set_title("RGB Image")
        axes[idx][0].axis('off')

        im2 = axes[idx][1].imshow(images2[idx], cmap='viridis_r')
        axes[idx][1].set_title("Metric Depth Prediction")
        axes[idx][1].axis('off')
        plt.colorbar(im2, ax=axes[idx][1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Save and close figure
    filename = os.path.join(
        save_dir,
        f"{prefix}_{save_images_two_sources.counter:03d}.png"
    )
    fig.savefig(filename)
    plt.close(fig)

    save_images_two_sources.counter += 1
    print(f"Saved figure: {filename}")

def masked_downsample(input_tensor, mask, scale_factor=0.25, mode='bilinear'):
    """
    Correctly downsamples sparse maps by ignoring zeros during interpolation.
    
    Args:
        input (torch.Tensor): Sparse map (B, 1, H, W)
        mask (torch.Tensor): Binary mask of known pixels, same shape as input
        scale_factor (float): Downsampling factor
    Returns:
        torch.Tensor: Correctly downsampled map
        torch.Tensor: Corresponding downsampled mask
    """
    # Interpolate residual with zeros at unknown locations
    input_weighted = input_tensor * mask

    # Interpolate residual and mask separately
    residual_ds = F.interpolate(input_weighted, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    mask_ds = F.interpolate(mask, scale_factor=scale_factor, mode='bilinear', align_corners=False)

    # Avoid division by zero
    mask_eps = 1e-6
    downsampled = residual_ds / (mask_ds + mask_eps)

    # Set zero at locations without any data
    downsampled[mask_ds < mask_eps] = 0

    return downsampled, mask_ds

def joint_bilateral_filter(sparse_residual, guidance, kernel_size=9, sigma_spatial=3.0, sigma_depth=0.1):
    B, C, H, W = sparse_residual.shape
    pad = kernel_size // 2

    unfolded_residual = F.unfold(F.pad(sparse_residual, [pad]*4, mode='reflect'), kernel_size=kernel_size).view(B, 1, kernel_size**2, H, W)
    unfolded_guidance = F.unfold(F.pad(guidance, [pad]*4, mode='reflect'), kernel_size=kernel_size).view(B, 1, kernel_size**2, H, W)

    # spatial kernel
    grid_y, grid_x = torch.meshgrid(torch.arange(-pad, pad+1), torch.arange(-pad, pad+1), indexing='ij')
    spatial_kernel = torch.exp(-(grid_x**2 + grid_y**2)/(2*sigma_spatial**2)).view(1,1,-1,1,1).to(sparse_residual.device)

    # depth kernel
    depth_kernel = torch.exp(-((unfolded_guidance - guidance.unsqueeze(2))**2)/(2*sigma_depth**2))

    combined_kernel = spatial_kernel * depth_kernel * (unfolded_residual != 0).float()

    dense_residual = (unfolded_residual * combined_kernel).sum(dim=2)
    weight_sum = combined_kernel.sum(dim=2) + 1e-8

    dense_residual /= weight_sum
    return dense_residual

def downsampled_joint_bilateral_filter(sparse_residual, guidance, downscale_factor=4, kernel_size=9, sigma_spatial=5.0, sigma_depth=0.1):
    mask = (sparse_residual > 0).float()

    # Correct masked downsampling
    sparse_residual_ds, mask_ds = masked_downsample(sparse_residual, mask, scale_factor=1/downscale_factor)
    guidance_ds = F.interpolate(guidance, scale_factor=1/downscale_factor, mode='bilinear', align_corners=False)

    # Filter at low-resolution
    dense_ds = joint_bilateral_filter(sparse_residual_ds, guidance_ds, kernel_size, sigma_spatial, sigma_depth)

    # Upsample back
    dense_residual = F.interpolate(dense_ds, size=sparse_residual.shape[-2:], mode='bilinear', align_corners=False)

    return dense_residual

from torchvision.transforms.functional import gaussian_blur
def densify_sparse_residual_gaussian(sparse_scale_residual, kernel_size=(15, 15), sigma=(5.0, 5.0)):
    """
    Densify a sparse scale residual map using Gaussian blur.

    Args:
        sparse_scale_residual (torch.Tensor): Sparse residual tensor of shape (B, 1, H, W),
                                               containing zeros at unknown points.
        kernel_size (tuple): Gaussian blur kernel size, default (15, 15).
        sigma (tuple): Gaussian blur standard deviation, default (5.0, 5.0).

    Returns:
        torch.Tensor: Dense residual map with interpolated values at previously unknown points.
    """
    mask_known = (sparse_scale_residual > 0).float()
    filled_residual = sparse_scale_residual.clone()

    # Replace unknown values with mean of known values to aid smoothing
    mean_known = sparse_scale_residual[mask_known.bool()].mean()
    filled_residual[mask_known == 0] = mean_known

    # Apply Gaussian blur to interpolate/densify
    dense_residual = gaussian_blur(filled_residual, kernel_size=kernel_size, sigma=sigma)

    return dense_residual

def bilateral_filter(sparse_residual, GA, kernel_size=7, sigma_spatial=3, sigma_depth=0.1):
    B, C, H, W = sparse_residual.shape
    padding = kernel_size // 2

    # Pad input tensors
    padded_residual = F.pad(sparse_residual, [padding]*4, mode='reflect')
    padded_depth = F.pad(GA, [padding]*4, mode='reflect')

    dense_residual = torch.zeros_like(sparse_residual)
    weight_sum = torch.zeros_like(sparse_residual)

    # Coordinate grids for spatial kernel
    device = sparse_residual.device
    y, x = torch.meshgrid(torch.arange(-padding, padding+1, device=device),
                          torch.arange(-padding, padding+1, device=device), indexing='ij')
    spatial_kernel = torch.exp(-(x**2 + y**2) / (2 * sigma_spatial**2))

    for i in range(kernel_size):
        for j in range(kernel_size):
            shifted_residual = padded_residual[:, :, i:i+H, j:j+W]
            shifted_depth = padded_depth[:, :, i:i+H, j:j+W]

            depth_diff = (GA - shifted_depth).abs()
            depth_kernel = torch.exp(-(depth_diff**2) / (2 * sigma_depth**2))

            mask_valid = (shifted_residual != 0).float()

            combined_kernel = spatial_kernel[i, j] * depth_kernel * mask_valid

            dense_residual += shifted_residual * combined_kernel
            weight_sum += combined_kernel

    dense_residual /= (weight_sum + 1e-8)
    return dense_residual

def kernel_interpolate(sparse_map, mask_known, kernel_size=7, sigma=2.0):
    B, _, H, W = sparse_map.shape
    
    # Create Gaussian kernel
    grid_y, grid_x = torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size))
    grid_y, grid_x = grid_y - kernel_size//2, grid_x - kernel_size//2
    gaussian_kernel = torch.exp(-(grid_x**2 + grid_y**2)/(2*sigma**2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).to(sparse_map.device)
    
    # Weighted sum of known points
    weighted_residual = F.conv2d(sparse_map * mask_known, gaussian_kernel, padding=kernel_size//2)
    weighted_mask = F.conv2d(mask_known, gaussian_kernel, padding=kernel_size//2)
    
    dense_residual = weighted_residual / (weighted_mask + 1e-6)
    return dense_residual


def densify_residual_interpolation(sparse_scale_residual, scale_factor=0.25):
    """
    Densify a sparse scale residual map using bilinear interpolation.

    Args:
        sparse_scale_residual (torch.Tensor): Sparse residual tensor of shape (B, 1, H, W),
                                              zeros at unknown locations.
        scale_factor (float): Factor to downscale before upscaling for smoothing.

    Returns:
        torch.Tensor: Dense residual tensor of shape (B, 1, H, W).
    """
    mask_known = (sparse_scale_residual > 0).float()
    filled_residual = sparse_scale_residual.clone()

    # Replace unknown values with mean of known values
    mean_known = sparse_scale_residual[mask_known.bool()].mean()
    filled_residual[mask_known == 0] = mean_known

    # Downscale then upscale for smoothing
    low_res = F.interpolate(filled_residual, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    dense_residual = F.interpolate(low_res, size=sparse_scale_residual.shape[-2:], mode='bilinear', align_corners=False)

    return dense_residual


def joint_bilateral_filter_filled_ones(sparse, depth, sigma_s, sigma_r, kernel_size):
    """
    Joint bilateral filter that propagates sparse scale residual values using a dense depth map.
    In this version, missing values in the sparse map are set to 1, and any pixel with a value
    not equal to 1 is considered valid.
    
    Args:
        sparse (torch.Tensor): Sparse scale residual map of shape (B, 1, H, W) with missing values = 1.
        depth (torch.Tensor): Dense depth map of shape (B, 1, H, W).
        sigma_s (float): Standard deviation for the spatial Gaussian.
        sigma_r (float): Standard deviation for the range (depth difference) Gaussian.
        kernel_size (int): Size of the square filter window (should be odd).
    
    Returns:
        torch.Tensor: Filtered scale residual map of shape (B, 1, H, W). Pixels with no valid neighbors remain 1.
    """
    B, C, H, W = sparse.shape
    pad = kernel_size // 2
    device = sparse.device

    # Precompute spatial weights for each kernel offset.
    coords = torch.arange(-pad, pad+1, device=device, dtype=torch.float)
    spatial_weights = {}
    for i in coords:
        for j in coords:
            weight = torch.exp(-((i**2 + j**2) / (2 * sigma_s**2)))
            spatial_weights[(int(i.item()), int(j.item()))] = weight

    # Pad the sparse and depth maps.
    # For sparse, pad with constant value 1; for depth, use reflect padding.
    sparse_padded = F.pad(sparse, (pad, pad, pad, pad), mode='constant', value=1)
    depth_padded = F.pad(depth, (pad, pad, pad, pad), mode='reflect')

    # Create valid mask: valid if the sparse value is not 1.
    valid_mask = (sparse_padded != 1).float()

    # The center region (for output) in the padded maps.
    center_depth = depth_padded[..., pad:pad+H, pad:pad+W]

    # Initialize accumulators.
    numerator = torch.zeros((B, 1, H, W), device=device)
    denominator = torch.zeros((B, 1, H, W), device=device)

    # Loop over each offset in the kernel window.
    for (dy, dx), s_weight in spatial_weights.items():
        # Define the slicing indices for the shifted window.
        shifted_depth = depth_padded[..., pad+dy:pad+dy+H, pad+dx:pad+dx+W]
        shifted_sparse = sparse_padded[..., pad+dy:pad+dy+H, pad+dx:pad+dx+W]
        shifted_valid = valid_mask[..., pad+dy:pad+dy+H, pad+dx:pad+dx+W]

        # Compute the range kernel: based on the depth difference.
        range_weight = torch.exp(-((shifted_depth - center_depth) ** 2) / (2 * sigma_r**2))
        # Overall weight combines spatial weight, range weight, and valid mask.
        overall_weight = s_weight * range_weight * shifted_valid

        numerator += overall_weight * shifted_sparse
        denominator += overall_weight

    # Compute the filtered output.
    filtered = numerator / (denominator + 1e-8)
    # For pixels with no valid neighbors, set the result to 1.
    filtered = torch.where(denominator < 1e-8, torch.ones_like(filtered), filtered)

    return filtered



def compute_gradient_map(depth):
    # depth: [B, 1, H, W]
    
    # Compute horizontal differences (dx)
    grad_x = depth[:, :, :, 1:] - depth[:, :, :, :-1]
    
    # Compute vertical differences (dy)
    grad_y = depth[:, :, 1:, :] - depth[:, :, :-1, :]
    
    # Pad to keep the same H and W
    grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
    grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
    
    # Compute gradient magnitude per pixel
    gradient_map = torch.sqrt(grad_x**2 + grad_y**2)
    angle_map = torch.atan(grad_y / (grad_x + 1e-10))
    
    return gradient_map, angle_map

def compute_surface_normals(depth):
    """
    Computes surface normals from a relative depth map.
    
    Args:
        depth (torch.Tensor): Depth map tensor of shape (B, 1, H, W).
        
    Returns:
        torch.Tensor: Normal map of shape (B, 3, H, W), where each normal is unit length.
    """
    # Compute x and y gradients using finite differences
    grad_x = depth[:, :, :, 1:] - depth[:, :, :, :-1]  # shape: (B, 1, H, W-1)
    grad_y = depth[:, :, 1:, :] - depth[:, :, :-1, :]  # shape: (B, 1, H-1, W)
    
    # Pad the gradients to match the original depth dimensions
    grad_x = F.pad(grad_x, (0, 1, 0, 0))  # pad one column on the right -> (B, 1, H, W)
    grad_y = F.pad(grad_y, (0, 0, 0, 1))  # pad one row at the bottom -> (B, 1, H, W)
    
    # Construct the normal vector at each pixel: (-grad_x, -grad_y, 1)
    ones = torch.ones_like(depth)
    normals = torch.cat((-grad_x, -grad_y, ones), dim=1)  # shape: (B, 3, H, W)
    show_images(normals)
    # Normalize the normals to unit length.
    normals = F.normalize(normals, p=2, dim=1)
    return normals

from torch.functional import norm
class SurfaceNet(nn.Module):
    def __init__(self):
        super(SurfaceNet, self).__init__()
        # Optional: you can still define convolution layers if needed.
        # Here we define the kernels directly in forward.
    
    def forward(self, x):
        # Expect input shape: (B, 1, H, W)
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError("Input must have shape (B, 1, H, W)")
        
        B, C, H, W = x.shape

        # Define kernels for x and y derivatives
        delzdelxkernel = torch.tensor([[0.0, 0.0, 0.0],
                                       [-1.0, 0.0, 1.0],
                                       [0.0, 0.0, 0.0]],
                                      dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        # Use padding=1 to preserve spatial dimensions
        delzdelx = F.conv2d(x, delzdelxkernel, padding=1)

        delzdelykernel = torch.tensor([[0.0, -1.0, 0.0],
                                       [0.0,  0.0, 0.0],
                                       [0.0,  1.0, 0.0]],
                                      dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        delzdely = F.conv2d(x, delzdelykernel, padding=1)

        # For the derivative along z, we simply use ones
        delzdelz = torch.ones_like(delzdely)

        # Concatenate along the channel dimension to get shape (B, 3, H, W)
        surface_norm = torch.cat((-delzdelx, -delzdely, delzdelz), dim=1)
        
        # Normalize each vector to have unit length
        norm_val = torch.norm(surface_norm, dim=1, keepdim=True) + 1e-8  # avoid division by zero
        # print(norm_val)
        surface_norm = surface_norm / norm_val

        return surface_norm
    
class AffinityDC(DepthModel):
    patch_size = 14  # patch size of the pretrained dinov2 model
    use_bn = False
    use_clstoken = False
    output_act = 'identity'

    def __init__(self,
                 da_pretrained_resource,
                 **kwargs):
        super().__init__()

        self.min_pred = 0.1
        self.max_pred = 20
        self.ga_max = 20
        self.biliteral_kernal_size = 5
        self.min_pred_inv = 1.0/self.min_pred
        self.max_pred_inv = 1.0/self.max_pred
        self.train_dino = kwargs['train_dino']
        self.dino_lr_factor = kwargs['dino_lr_factor']
        model_config = model_configs[kwargs['da_model_type']]

        self.encoder = kwargs['da_model_type']
        self.model_config = model_config
        
        self.pretrained = DINOv2(model_name=kwargs['da_model_type'])
        if not self.train_dino:
            for param in self.pretrained.parameters():
                param.requires_grad = False
        
        dim = self.pretrained.blocks[0].attn.qkv.in_features
        self.depth_head = DPTHeadCustomised3(in_channels=dim,
                                  features=model_config['features'],
                                  out_channels=model_config['out_channels'],
                                  use_bn=self.use_bn,
                                  use_clstoken=self.use_clstoken,
                                  output_act=self.output_act)
        # self.depth_head = DPTHead(in_channels=dim,
        #                           features=model_config['features'],
        #                           out_channels=model_config['out_channels'],
        #                           use_bn=self.use_bn,
        #                           use_clstoken=self.use_clstoken,
        #                           )

        
        for name, param in self.depth_head.named_parameters():
                param.requires_grad = False

       

        self.scale_map_learner = SMLDeformableAttention(features=64, non_negative=False, channels_last=False, align_corners=True,
        blocks={'expand': True}, min_pred=0.1, max_pred=self.max_pred)

      
        
        self.load_checkpoint(da_pretrained_resource)

    def load_checkpoint(self, ckpt_path):
        if os.path.exists(ckpt_path):
            print(f'Loading checkpoint from {ckpt_path}')
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            # print(checkpoint.keys())
            # self.load_state_dict({k[9:]: v for k, v in checkpoint['state_dict'].items()})
            self.load_state_dict(checkpoint, strict=False)
        else:
            print(f'Checkpoint {ckpt_path} not found')





    def forward(self, x, prompt_depth=None, image_no_norm = None):
        assert prompt_depth is not None, 'prompt_depth is required'

        sparse_mask = (prompt_depth < self.max_pred) * (prompt_depth > self.min_pred)
        sparse_mask = sparse_mask.float()
        sparse_depth_inv = prompt_depth * sparse_mask
        sparse_depth_inv = torch.where(sparse_depth_inv == 0, sparse_depth_inv, 1.0 / sparse_depth_inv)

        h, w = x.shape[-2:]
        # print(x.shape)
        features = self.pretrained.get_intermediate_layers(x, self.model_config['layer_idxs'],return_class_token=True)
        
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        rel_depth, extracted_features = self.depth_head(features, patch_h, patch_w, prompt_depth)

        # sparse_depth_inv = sparse_depth_inv.half()
        # sparse_mask = sparse_mask.half()
        # rel_depth = rel_depth.half()
        estimator = LeastSquaresEstimatorTorch(rel_depth, sparse_depth_inv, sparse_mask)
        estimator.compute_scale_and_shift() 
        estimator.apply_scale_and_shift()
        estimator.clamp_min_max(clamp_min=self.min_pred,clamp_max=self.ga_max)
        ga_result = estimator.output
        d = estimator.output.clone()
        # show_images(1.0/ga_result)

        ga_mask = (rel_depth >0).float() # where infinte area is, actually using the reltive map is better
        sparse_and_ga_mask  = sparse_mask * ga_mask

        scale_residual = torch.zeros_like(sparse_depth_inv)
        scale_residual[sparse_and_ga_mask.bool()] = sparse_depth_inv[sparse_and_ga_mask.bool()] / ga_result[sparse_and_ga_mask.bool()]

        scale_residual[scale_residual==0] = 1
        dense_residual = joint_bilateral_filter_filled_ones(scale_residual, ga_result, sigma_s=2, sigma_r=0.001,kernel_size=self.biliteral_kernal_size)
        # show_images_two_sources(ga_result,dense_residual)
        
     
        pred, scales = self.scale_map_learner(ga_result, dense_residual, d, extracted_features)
       
        # display_and_save_images(image_no_norm*255, prompt_depth, rel_depth, 1.0/pred, save_path='./turbid_test.png')
        # breakpoint()
        # plot_sparse_depth_tensor(prompt_depth)
        # display_images(image_no_norm, prompt_depth, rel_depth, 1.0/pred)
        # show_images_two_sources(image_no_norm*255,1.0/pred)
        # show_images_three_sources(image_no_norm,ga_result, pred)
        # show_images_two_sources(1.0/ga_result,1.0/pred)
        # show_images_four_sources(ga_result,dense_residual,ga_result*dense_residual,pred)
        output = dict(metric_depth=1.0/pred)
        output['inverse_depth'] = pred
        return output

    @torch.no_grad()
    def predict(self,
                image: torch.Tensor,
                prompt_depth: torch.Tensor):
        return self.forward(image, prompt_depth)

    
    def get_lr_params(self, lr):
        param_conf = []

        # # Add pretrained DINO branch parameters only if self.train_dino is True
        # if self.train_dino:
        #     pretrained_params = list(self.pretrained.parameters())
        #     param_conf.append({
        #         'params': pretrained_params,
        #         'lr': lr / self.dino_lr_factor,
        #         'name': 'pretrained'
        #     })

        # Separate parameters based on whether '_prompt' is in their name, 
        # and skip the ones from pretrained to avoid duplication.
        scale_block_params = []
        for name, param in self.named_parameters():
            if 'pretrained' not in name and 'depth_head' not in name:
                if 'scale' in name:
                    # print(name)
                    scale_block_params.append(param)
        # breakpoint()
        if scale_block_params:
            param_conf.append({
                'params': scale_block_params,
                'lr': lr ,
                'name': 'scale modules'
            })

        return param_conf

    @staticmethod
    def build(da_pretrained_recource, pretrained_resource = None, **kwargs):
        
        model = AffinityDC(da_pretrained_recource,  **kwargs)
        if pretrained_resource:
            assert isinstance(pretrained_resource, str), "pretrained_resource must be a string"
            model = load_state_from_resource(model, pretrained_resource)
        return model

    @staticmethod
    def build_from_config(config):
        return AffinityDC.build(**config)
