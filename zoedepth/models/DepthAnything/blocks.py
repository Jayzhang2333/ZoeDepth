import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import numpy as np
from zoedepth.models.layers.fusion_layers import FillConv, PyramidVisionTransformer, conv_bn_relu, BasicBlock, SelfAttnPropagation, mViT, mViT_assemble
import matplotlib.pyplot as plt
from zoedepth.models.layers.global_alignment import LeastSquaresEstimatorTorch, AnchorInterpolator2D

def resize_sparse_depth(depth, new_size):
    """
    Resizes a sparse depth tensor while preserving the sparse measurements.
    
    For each valid measurement in the original tensor (depth > 0), it computes
    its normalized spatial coordinates and maps it to a new output grid.
    If multiple valid measurements land in the same location in the new grid,
    their values are averaged.
    
    Parameters:
        depth (torch.Tensor): Input tensor of shape (B, 1, H, W) with sparse depth measurements.
        new_size (tuple or int): Desired output spatial size as (new_H, new_W) or a single integer for a square output.
    
    Returns:
        torch.Tensor: Resized depth tensor of shape (B, 1, new_H, new_W).
    """
    # Unpack dimensions
    B, C, H, W = depth.shape
    if isinstance(new_size, int):
        new_H, new_W = new_size, new_size
    else:
        new_H, new_W = new_size

    # Prepare tensors to accumulate sums and counts for averaging
    sum_tensor = torch.zeros((B, 1, new_H, new_W), device=depth.device, dtype=depth.dtype)
    count_tensor = torch.zeros((B, 1, new_H, new_W), device=depth.device, dtype=depth.dtype)

    # Create a grid for the original coordinates (for a single image)
    # ys: row indices (0 to H-1) and xs: column indices (0 to W-1)
    ys = torch.arange(H, device=depth.device).view(H, 1).expand(H, W)
    xs = torch.arange(W, device=depth.device).view(1, W).expand(H, W)
    # Flatten the coordinate grids to shape (H*W,)
    ys = ys.flatten()  # row indices
    xs = xs.flatten()  # column indices

    # Map the original pixel indices to normalized coordinates in [0, 1)
    # Then map to the new grid coordinates.
    # Using floor to choose an integer index.
    new_ys = (ys.float() / H * new_H).floor().long()
    new_xs = (xs.float() / W * new_W).floor().long()
    # Clamp to be safe (the last index should be new_H-1 or new_W-1)
    new_ys = new_ys.clamp(0, new_H - 1)
    new_xs = new_xs.clamp(0, new_W - 1)

    # Loop over each item in the batch
    for b in range(B):
        # Get a flattened view of the depth values for this image
        depth_flat = depth[b, 0].flatten()
        # Define a valid measurement as a value > 0
        valid = depth_flat > 0
        valid_depth = depth_flat[valid]
        # Get corresponding new grid indices for valid measurements
        valid_new_ys = new_ys[valid]
        valid_new_xs = new_xs[valid]
        # Compute a flattened index for the new grid (row-major order)
        idx = valid_new_ys * new_W + valid_new_xs

        # Flatten the corresponding slice of the output tensors
        sum_flat = sum_tensor[b, 0].view(-1)
        count_flat = count_tensor[b, 0].view(-1)
        # Scatter-add the depth values to the corresponding positions
        sum_flat.scatter_add_(0, idx, valid_depth)
        # Also count how many values have been added per position
        ones = torch.ones_like(valid_depth)
        count_flat.scatter_add_(0, idx, ones)

    # Compute the final averaged output: where no values were accumulated, output remains 0.
    output = torch.zeros_like(sum_tensor)
    mask = count_tensor > 0
    output[mask] = sum_tensor[mask] / count_tensor[mask]

    return output


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
    
    for idx in range(batch_size):
        # Display source1 image
        axes[idx][0].imshow(images1[idx])
        axes[idx][0].set_title("Source 1")
        axes[idx][0].axis('off')
        
        # Display source2 image
        axes[idx][1].imshow(images2[idx])
        axes[idx][1].set_title("Source 2")
        axes[idx][1].axis('off')
        
        # Display source3 image
        axes[idx][2].imshow(images3[idx])
        axes[idx][2].set_title("Source 3")
        axes[idx][2].axis('off')
        
    
    plt.tight_layout()
    plt.show()


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

def show_images(tensor_images):
    tensor_images = tensor_images.detach().cpu().numpy()  # Convert to numpy if tensor
    tensor_images = np.transpose(tensor_images, (0, 2, 3, 1))  # Change from CHW to HWC
    
    # Display the images
    batch_size = tensor_images.shape[0]
    fig, axes = plt.subplots(1, batch_size, figsize=(15, 5))  # Adjust number of subplots as needed
    if batch_size == 1:  # Handle case where batch size is 1
        axes = [axes]
    
    for idx in range(batch_size):
        axes[idx].imshow(tensor_images[idx])  # Assuming images are normalized [0, 1]
        axes[idx].axis('off')
    
    plt.show()


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )

def _make_fusion_block_with_depth(features, use_bn, input_size = None, patch_size = 1, base_layer = False):
    return FeatureFusionBlock_modified(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        input_size=input_size,
        patch_size = patch_size,
        base_layer = base_layer
    )

def _make_fusion_block_scale_residual_block(features, use_bn):
    return FeatureFusionBlockScaleResidual(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape*2
        out_shape3 = out_shape*4
        if len(in_shape) >= 4:
            out_shape4 = out_shape*8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )

    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features//2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

        self.size = size

    def forward(self, *xs, size=None, prompt_depth=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


class FeatureFusionControlBlock(FeatureFusionBlock):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        """Init.

        Args:
            features (int): number of features
        """
        super.__init__(features, activation, deconv,
                       bn, expand, align_corners, size)
        self.copy_block = FeatureFusionBlock(
            features, activation, deconv, bn, expand, align_corners, size)

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class FeatureFusionDepthBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None, patch_size = 1):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionDepthBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features//2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        
        
        # self.resConfUnit_depth = nn.Sequential(
        #     nn.Conv2d(1, features, kernel_size=3, stride=1,
        #               padding=1, bias=True, groups=1),
        #     activation,
        #     nn.Conv2d(features, features, kernel_size=3,
        #               stride=1, padding=1, bias=True, groups=1),
        #     activation,
        #     zero_module(
        #         nn.Conv2d(features, features, kernel_size=3,
        #                   stride=1, padding=1, bias=True, groups=1)
        #     )
        # )
        self.sparse_embedding_prompt = nn.Sequential(
            nn.Conv2d(1, features, kernel_size=3, stride=1,
                      padding=1, bias=True, groups=1),
            activation)
        
        self.atten_conv_prompt = nn.Sequential(BasicBlock(features*2, features*2, ratio=4), 
                                            #    mViT(features*2, n_query_channels=128, patch_size=patch_size, embedding_dim=128),
                                               activation)

        self.projection_prompt = zero_module(
                nn.Conv2d(features*2, features, kernel_size=3,
                          stride=1, padding=1, bias=True, groups=1)
            )


        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, prompt_depth=None, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]
        
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        
        output = self.resConfUnit2(output)
        print(output.shape)
        if prompt_depth is not None:
            # prompt_depth_resized = F.interpolate(
            #     prompt_depth, output.shape[2:], mode='bilinear', align_corners=False)
            # prompt_depth_mask = (prompt_depth > 0).float()
            prompt_depth_resized = resize_sparse_depth(prompt_depth,  output.shape[2:])
            # min_per_sample = torch.amin(prompt_depth_resized, dim=(1, 2, 3))
            # max_per_sample = torch.amax(prompt_depth_resized, dim=(1, 2, 3))

            # print("Min per sample:", min_per_sample)
            # print("Max per sample:", max_per_sample)
            # show_images_two_sources(prompt_depth, prompt_depth_resized)
            # res = self.resConfUnit_depth(prompt_depth)
           
            res = self.sparse_embedding_prompt(prompt_depth_resized)
            res = torch.cat((res, output), dim=1)
            res = self.atten_conv_prompt(res)
            res = self.projection_prompt(res)
            output = self.skip_add.add(output, res)

        print(size)
        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output
    



class FeatureFusionBlock_modified(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, input_size=None, size = None, patch_size =1, base_layer = False):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_modified, self).__init__()
        self.base_layer = base_layer
        self.max_depth = 18
        self.min_depth = 0.1

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features//2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        
        
        self.rel_depth_prompt = nn.Sequential(
                nn.Conv2d(features, features//2,
                            kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(features//2, 1, kernel_size=1,
                            stride=1, padding=0),
                nn.ReLU(True),
            )
        
        self.local_dense_branch_prompt = nn.Sequential(
                BasicBlock(features+4, features+4, ratio=4)
        )

        self.global_dense_branch_prompt = nn.Sequential(
                mViT_assemble((features+4), patch_size=patch_size, embedding_dim=128, out_channels = (features+4), size = input_size, num_layer = 2),
        )

        self.fuse_conv1_prompt = nn.Sequential(
            nn.Conv2d(((features+4) *2), (features+4), 3, 1, 1),
            nn.ReLU(),
        )
        self.fuse_conv2_prompt = nn.Sequential(
            nn.Conv2d(((features+4) *2), (features+4), 3, 1, 1),
            nn.Sigmoid(),
        )

        self.dense_output_prompt = nn.Sequential(
            nn.Conv2d((features+4), (features+4)//2,
                            kernel_size=3, stride=1, padding=1),
            nn.ReLU(False),
            nn.Conv2d((features+4)//2, 1, kernel_size=1,
                        stride=1, padding=0),
            nn.Identity(),
        )


        self.dense_output_confidence_prompt = nn.Sequential(
            nn.Conv2d((features+4), (features+4)//2,
                            kernel_size=3, stride=1, padding=1),
            nn.ReLU(False),
            nn.Conv2d((features+4)//2, 1, kernel_size=1,
                        stride=1, padding=0),
            nn.Sigmoid(),
        )
        
        
        
        # self.local_affinity_prompt = nn.Sequential(
        #     nn.Conv2d((features+1), (features+1)//2, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d((features+1)//2, 9, kernel_size=1, stride=1, padding=0),
        #     nn.Sigmoid()
        # )

        self.projection_prompt = zero_module(
                nn.Conv2d(1, features, kernel_size=3,
                          stride=1, padding=1, bias=True, groups=1)
            )
        



        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def prompt_sparse_depth(self, decoder_featrues, sparse_depth, previous_depth = None, previous_conf = None):
        
        rel_depth = self.rel_depth_prompt(decoder_featrues)
        max_depth_inv = 1.0/ self.max_depth
        min_deprh_inv = 1.0/ self.min_depth

        # sparse_depth should already be inverse depth at this point
        sparse_mask = (sparse_depth < min_deprh_inv) * (sparse_depth > max_depth_inv)
        estimator = LeastSquaresEstimatorTorch(rel_depth, sparse_depth, sparse_mask)
        estimator.compute_scale_and_shift()  # find best scale & shift
        estimator.apply_scale_and_shift()    # apply them
        # it will do the inverse inside the function
        # show_images(estimator.output)
        estimator.clamp_min_max(clamp_min=self.min_depth, clamp_max=self.max_depth)
        ga_result = estimator.output

        if previous_depth is not None:
            sparse_depth = torch.where(sparse_depth == 0, previous_depth, sparse_depth)
           
        scale_residual = torch.where(sparse_depth>0, sparse_depth / ga_result, torch.ones_like(sparse_depth))

        if previous_conf is not None:
            sparse_mask = torch.where(sparse_mask == 0, previous_conf, sparse_mask)
        input_features = torch.cat([decoder_featrues, scale_residual, sparse_mask, ga_result, rel_depth], dim=1)

        local_features = self.local_dense_branch_prompt(input_features)
        global_features = self.global_dense_branch_prompt(input_features)

        combined_features = torch.cat([local_features, global_features], dim=1)

        combined_features = self.fuse_conv1_prompt(combined_features) * self.fuse_conv2_prompt(combined_features)

        dense_scale_residual = self.dense_output_prompt(combined_features)
        dense_scale_residual_conf = self.dense_output_confidence_prompt(combined_features)
        
        
        # show_images(local_filled)
        scales = F.relu(1.0 + dense_scale_residual)
        pred = scales*ga_result
        # show_images_two_sources(scales, dense_scale_residual_conf)
        return pred, dense_scale_residual_conf



    def forward(self, *xs, prompt_depth=None, size=None, previous_depth = None, previous_conf = None):
        """Forward pass.

        Returns:
            tensor: output
        """
        if self.base_layer == False:
            assert (previous_depth is not None and previous_conf is not None), 'If not base layer, must pass previous depth'
        
        output = xs[0]
        if previous_depth is not None:
            assert (previous_depth.shape[2:] == output.shape[2:] and previous_conf.shape[2:] == output.shape[2:]), 'the depth and conf from previous layer must have correct spatial dimension'
        
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        
        output = self.resConfUnit2(output)
        
        if prompt_depth is not None:
            
            prompt_depth_resized = resize_sparse_depth(prompt_depth,  output.shape[2:])
           
            depth_pred, confidence = self.prompt_sparse_depth(output, prompt_depth_resized, previous_depth, previous_conf)
            res = self.projection_prompt(depth_pred)
            output = self.skip_add.add(output, res)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        depth_pred = nn.functional.interpolate(
            depth_pred, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        confidence = nn.functional.interpolate(
            confidence, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output, depth_pred, confidence
    



class FeatureFusionBlockScaleResidual(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlockScaleResidual, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features//2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()
        

    def forward(self, *xs,  size=None):
        """Forward pass.

        Returns:
            tensor: output
        """

        if (size is None):
            modifier = {"scale_factor": 2}
        else:
            modifier = {"size": size}
        
        output = xs[0]
        
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        
        output = self.resConfUnit2(output)
        
        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)
    

        return output
    

class ScaleResidualBlock(nn.Module):
    def __init__(self, features, out_features, activation, bn, input_size=None, patch_size =1, 
                 base_layer = False, align_corners = True):
        """Init.

        Args:
            features (int): number of features
        """
        super(ScaleResidualBlock, self).__init__()

        self.base_layer = base_layer
        self.align_corners = align_corners
        
        self.residual_embedding = conv_bn_relu(1, 8, kernel=3, stride=1, padding=1,
                                          bn=False)
        
        self.ga_embedding = conv_bn_relu(1, 8, kernel=3, stride=1, padding=1,
                                          bn=False)
        
        self.local_dense_branch = nn.Sequential(
                BasicBlock(features, features, ratio=4)
        )

        self.global_dense_branch = nn.Sequential(
                mViT_assemble(features, patch_size=patch_size, embedding_dim=128, out_channels = features, size = input_size, num_heads=8, num_layer = 2),
        )

        self.fuse_conv1 = nn.Sequential(
            nn.Conv2d((features *2), features, 3, 1, 1),
            nn.ReLU(),
        )
        self.fuse_conv2 = nn.Sequential(
            nn.Conv2d((features *2), features, 3, 1, 1),
            nn.Sigmoid(),
        )

        self.decode_conv = ResidualConvUnit(features, activation, bn)
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.skip_add = nn.quantized.FloatFunctional()
        

    def forward(self, decoder_featrues, ga_result, scale_residaul_map, scale_residual_mask = None, previous_embedding = None, size = None):
        
        scale_residaul_map = torch.where(scale_residaul_map == 0, torch.ones_like(scale_residaul_map), scale_residaul_map)
        # show_images_two_sources(ga_result, scale_residaul_map)
        if scale_residual_mask is not None:
            residual_embedding = self.residual_embedding(torch.cat([scale_residaul_map, scale_residual_mask], dim=1))
        else:
            residual_embedding = self.residual_embedding(scale_residaul_map)

        ga_embedding = self.ga_embedding(ga_result)
        decoder_scale_ga = torch.cat([decoder_featrues, residual_embedding, ga_embedding], dim=1)
       
            
        local_features = self.local_dense_branch(decoder_scale_ga)
        global_features = self.global_dense_branch(decoder_scale_ga)

        combined_features = torch.cat([local_features, global_features], dim=1)
        combined_features = self.fuse_conv1(combined_features) * self.fuse_conv2(combined_features)
       
        if previous_embedding is not None:
            combined_features = self.skip_add.add(previous_embedding, combined_features)

        #need the decoder here
        output = self.decode_conv(combined_features)

        if size is not None:
            output = nn.functional.interpolate(
                output, size, mode="bilinear", align_corners=self.align_corners
            )
        output = self.out_conv(output)
        

        return output
    
class ScaleResidualOutputBlock(nn.Module):
    def __init__(self, features, activation, bn, input_size=None, patch_size =1):
        """Init.

        Args:
            features (int): number of features
        """
        super(ScaleResidualOutputBlock, self).__init__()

        
        
        self.residual_embedding = conv_bn_relu(1, 8, kernel=3, stride=1, padding=1,
                                          bn=False)
        
        self.ga_embedding = conv_bn_relu(1, 8, kernel=3, stride=1, padding=1,
                                          bn=False)
        
        self.local_dense_branch = nn.Sequential(
                BasicBlock(features, features, ratio=4)
        )

        self.global_dense_branch = nn.Sequential(
                mViT_assemble(features, patch_size=patch_size, embedding_dim=128, out_channels = features, size = input_size, num_heads=8, num_layer = 2),
        )

        self.fuse_conv1 = nn.Sequential(
            nn.Conv2d((features *2), features, 3, 1, 1),
            nn.ReLU(),
        )
        self.fuse_conv2 = nn.Sequential(
            nn.Conv2d((features *2), features, 3, 1, 1),
            nn.Sigmoid(),
        )

        self.decode_conv = ResidualConvUnit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()
        

    def forward(self, decoder_featrues, ga_result, scale_residaul_map, scale_residual_mask = None, previous_embedding = None):
        
        scale_residaul_map = torch.where(scale_residaul_map == 0, torch.ones_like(scale_residaul_map), scale_residaul_map)
        # show_images_two_sources(ga_result, scale_residaul_map)
        if scale_residual_mask is not None:
            residual_embedding = self.residual_embedding(torch.cat([scale_residaul_map, scale_residual_mask], dim=1))
        else:
            residual_embedding = self.residual_embedding(scale_residaul_map)

        ga_embedding = self.ga_embedding(ga_result)
        decoder_scale_ga = torch.cat([decoder_featrues, residual_embedding, ga_embedding], dim=1)
       
            
        local_features = self.local_dense_branch(decoder_scale_ga)
        global_features = self.global_dense_branch(decoder_scale_ga)

        combined_features = torch.cat([local_features, global_features], dim=1)
        combined_features = self.fuse_conv1(combined_features) * self.fuse_conv2(combined_features)
       
        if previous_embedding is not None:
            combined_features = self.skip_add.add(previous_embedding, combined_features)

        #need the decoder here
        output = self.decode_conv(combined_features)
        

        return output

