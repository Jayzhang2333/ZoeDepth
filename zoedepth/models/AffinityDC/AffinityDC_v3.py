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

def show_images_two_sources(source1, source2):
    """
    Display images from two sources side by side in a row.
    
    Args:
        source1, source2: Two input tensors or numpy arrays of images.
                          Expected shapes: (N, C, H, W) for tensors.
    """
    def preprocess_images(source):
        if isinstance(source, np.ndarray):
            images = source
        else:  # Assume tensor
            images = source.detach().cpu().numpy()
        images = np.transpose(images, (0, 2, 3, 1))  # Convert from CHW to HWC
        return images

    # Preprocess both sources
    images1 = preprocess_images(source1)
    images2 = preprocess_images(source2)

    # Ensure the batch sizes match
    assert images1.shape[0] == images2.shape[0], "Batch sizes must match."

    batch_size = images1.shape[0]
    fig, axes = plt.subplots(batch_size, 2, figsize=(10, 5 * batch_size))  # 2 columns for sources

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

    plt.tight_layout()
    plt.show()

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



def bilateral_filter_vector(sparse_residual, GA, kernel_size=7, sigma_spatial=3, sigma_depth=0.1):
    B, C, H, W = sparse_residual.shape
    padding = kernel_size // 2

    # Pad inputs
    padded_residual = F.pad(sparse_residual, [padding]*4, mode='reflect')
    padded_depth = F.pad(GA, [padding]*4, mode='reflect')

    device = sparse_residual.device

    # Generate spatial kernel
    y, x = torch.meshgrid(
        torch.arange(-padding, padding + 1, device=device),
        torch.arange(-padding, padding + 1, device=device),
        indexing='ij'
    )
    spatial_kernel = torch.exp(-(x**2 + y**2) / (2 * sigma_spatial**2))

    # Extract sliding local blocks
    residual_patches = padded_residual.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)  # B,C,H,W,k,k
    depth_patches = padded_depth.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)

    # Reshape for computation
    residual_patches = residual_patches.permute(0, 1, 4, 5, 2, 3)  # B,C,k,k,H,W
    depth_patches = depth_patches.permute(0, 1, 4, 5, 2, 3)

    # Compute depth kernel
    depth_diff = (GA.unsqueeze(2).unsqueeze(2) - depth_patches).abs()
    depth_kernel = torch.exp(-(depth_diff ** 2) / (2 * sigma_depth ** 2))

    # Mask valid residual points
    mask_valid = (residual_patches != 0).float()

    # Combine kernels
    combined_kernel = spatial_kernel.view(1, 1, kernel_size, kernel_size, 1, 1) * depth_kernel * mask_valid

    # Compute weighted sum
    weighted_residual = (residual_patches * combined_kernel).sum(dim=(2,3))
    weight_sum = combined_kernel.sum(dim=(2,3)) + 1e-8

    dense_residual = weighted_residual / weight_sum

    return dense_residual



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
        self.max_pred = 25.0
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
        
        for name, param in self.depth_head.named_parameters():
                param.requires_grad = False

        # self.prop_kernel = 3
        # self.refiner = CSPNAccelerate(self.prop_kernel)

        self.scale_map_learner = SMLDeformableAttention(features=64, non_negative=False, channels_last=False, align_corners=True,
        blocks={'expand': True}, min_pred=0.1, max_pred=25.0)
        
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





    def forward(self, x, prompt_depth=None):
        assert prompt_depth is not None, 'prompt_depth is required'

        sparse_mask = (prompt_depth < self.max_pred) * (prompt_depth > self.min_pred)
        sparse_mask = sparse_mask.float()
        sparse_depth_inv = prompt_depth * sparse_mask
        sparse_depth_inv = torch.where(sparse_depth_inv == 0, sparse_depth_inv, 1.0 / sparse_depth_inv)

        h, w = x.shape[-2:]
        features = self.pretrained.get_intermediate_layers(x, self.model_config['layer_idxs'],return_class_token=True)
        
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        rel_depth, dinov2_features_list = self.depth_head(features, patch_h, patch_w, prompt_depth)
        
        estimator = LeastSquaresEstimatorTorch(rel_depth, sparse_depth_inv, sparse_mask)
        estimator.compute_scale() 
        estimator.apply_scale()
        estimator.clamp_min_max(clamp_min=self.min_pred,clamp_max=100.0)
        ga_result = estimator.output
        d = estimator.output.clone()

        ga_mask = (rel_depth >0).float() # where infinte area is, actually using the reltive map is better
        sparse_and_ga_mask  = sparse_mask * ga_mask

        scale_residual = torch.zeros_like(sparse_depth_inv)
        scale_residual[sparse_and_ga_mask.bool()] = sparse_depth_inv[sparse_and_ga_mask.bool()] / ga_result[sparse_and_ga_mask.bool()]

        # dense_residual = joint_bilateral_filter(scale_residual, ga_result, kernel_size=30, sigma_spatial=5.0, sigma_depth=0.1)
        # dense_residual = densify_sparse_residual_gaussian(scale_residual)
        # dense_residual = bilateral_filter(scale_residual, ga_result, kernel_size=15, sigma_spatial=5, sigma_depth=0.1)
        # dense_residual1 = bilateral_filter(scale_residual, ga_result, kernel_size=15, sigma_spatial=5, sigma_depth=0.05)
        dense_residual = bilateral_filter_vector(scale_residual, ga_result, kernel_size=17, sigma_spatial=5, sigma_depth=0.05)
        dense_residual[dense_residual == 0] = 1
        # dense_residual2 = downsampled_joint_bilateral_filter(
        #     scale_residual, ga_result,
        #     downscale_factor=4, kernel_size=9,
        #     sigma_spatial=5.0, sigma_depth=0.1
        #     )
        # dense_residual = kernel_interpolate(scale_residual, sparse_and_ga_mask, kernel_size=30)
        # dense_residual = densify_residual_interpolation(scale_residual, scale_factor=0.25)
        # sparse_mask = (prompt_depth < self.max_pred) * (prompt_depth > self.min_pred)
        # show_images_three_sources(dense_residual, scale_residual, ga_result)
        # show_images_four_sources(ga_result,x, prompt_depth, dense_residual)
        # show_images_three_sources(ga_result, sparse_depth_inv, scale_residual)
        # guide = compute_affinity_map_9(ga_result)

        # sparse_residual_filled = scale_residual.clone()
        # # sparse_residual_filled[~sparse_and_ga_mask.bool()] = 1.0



        # scale = sparse_residual_filled
        # sparse = scale_residual
        # # pred_inter = [pred_init]

        
        # guide_sum = torch.sum(guide.abs(), dim=1, keepdim=True)
        # guide = torch.div(guide, guide_sum)
        
        # for i in range(6):
        #     scale = self.refiner(guide, scale, sparse_residual_filled)
        #     scale = sparse * sparse_and_ga_mask + (1 - sparse_and_ga_mask) * scale
        # show_images_three_sources(ga_result,scale_residual, scale)
        # breakpoint()
        
        # scale_residual_np = scale_residual.cpu().numpy() 
        # sparse_mask_np = sparse_and_ga_mask.bool().cpu().numpy() 
        # int_scale_residual_batch = []
        # for i in range(scale_residual.shape[0]):

        #     ScaleMapInterpolator = AnchorInterpolator2D(
        #         sparse_depth = np.squeeze(scale_residual_np[i], axis = 0),
        #         valid = np.squeeze(sparse_mask_np[i], axis = 0),
        #     )
        #     ScaleMapInterpolator.generate_interpolated_scale_map(
        #         interpolate_method='linear', 
        #         fill_corners=False
        #     )
            
        #     int_scale_residual_batch.append(ScaleMapInterpolator.interpolated_scale_map.astype(np.float32))

        # int_scale_residual_batch = np.stack(int_scale_residual_batch)
        # int_scale_residual = torch.from_numpy(int_scale_residual_batch).float().to(prompt_depth.device) 
        # int_scale_residual = int_scale_residual.unsqueeze(1)
        # show_images_four_sources(dense_residual, int_scale_residual, scale_residual, ga_result)


        # ga_resized = nn.functional.interpolate(
        #     ga_result, (192, 256), mode="bilinear", align_corners=True)
        
        # scale_resized = nn.functional.interpolate(
        #     int_scale_residual, (192, 256), mode="bilinear", align_corners=True)
        
        # prediction is in 336,448
        # print(sparse_and_ga_mask.dtype)
        # scale_and_mask = torch.cat([scale_residual, sparse_and_ga_mask], dim = 1)
        pred, scales = self.scale_map_learner(ga_result, dense_residual, d, dinov2_features_list)
           
        # show_images_two_sources(rel_depth, 1.0/pred)
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
