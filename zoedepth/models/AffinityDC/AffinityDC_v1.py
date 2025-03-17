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
from zoedepth.models.DepthAnything.dpt import DPTHead, DPTHead_customised, DPTHeadCustomised2
from zoedepth.models.DepthAnything.dinov2 import DINOv2
from zoedepth.models.DepthAnything.blocks import ScaleResidualBlock, ScaleResidualOutputBlock
from zoedepth.models.depth_model import DepthModel
from zoedepth.models.model_io import load_state_from_resource
import torch.nn.functional as F
import os
import wandb
from zoedepth.models.layers.global_alignment import LeastSquaresEstimatorTorch, AnchorInterpolator2D

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
        self.max_pred = 18.0
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
        # self.depth_head = DPTHead_customised(in_channels=dim,
        #                           features=model_config['features'],
        #                           out_channels=model_config['out_channels'],
        #                           use_bn=self.use_bn,
        #                           use_clstoken=self.use_clstoken,
        #                           output_act=self.output_act,
        #                           with_depth = True)
       
        self.depth_head = DPTHeadCustomised2(in_channels=dim,
                                  features=model_config['features'],
                                  out_channels=model_config['out_channels'],
                                  use_bn=self.use_bn,
                                  use_clstoken=self.use_clstoken,
                                  output_act=self.output_act)
        
        for name, param in self.depth_head.named_parameters():
                param.requires_grad = False

        res_ga_embedding_num = 8
        scaleblock_input_features = model_config['features'] + 2*res_ga_embedding_num
        scaleblock_activation = nn.ReLU(False)
        scaleblock_bn = self.use_bn
        scaleblock_align_corners = True
        final_scaleblock_features = model_config['features']//2 + 2*res_ga_embedding_num

        self.scale_block_4 = ScaleResidualBlock(scaleblock_input_features, scaleblock_input_features, scaleblock_activation, scaleblock_bn, input_size=[24,32],
                                               patch_size =2, base_layer = True, align_corners = scaleblock_align_corners)
        
        self.scale_block_3 = ScaleResidualBlock(scaleblock_input_features, scaleblock_input_features, scaleblock_activation, scaleblock_bn, input_size=[48,64],
                                               patch_size =4, base_layer = True, align_corners = scaleblock_align_corners)
        
        self.scale_block_2 = ScaleResidualBlock(scaleblock_input_features, scaleblock_input_features, scaleblock_activation, scaleblock_bn, input_size=[96,128],
                                               patch_size =8, base_layer = True, align_corners = scaleblock_align_corners)
        
        self.scale_block_1 = ScaleResidualBlock(scaleblock_input_features, final_scaleblock_features, scaleblock_activation, scaleblock_bn, input_size=[192,256],
                                               patch_size =16, base_layer = True, align_corners = scaleblock_align_corners)
        
        # patch size here can actually use 28 to keep the number of token teh same as 192
        self.scale_block_0 = ScaleResidualOutputBlock(final_scaleblock_features, scaleblock_activation, scaleblock_bn, input_size=[336,448],patch_size =32)
        
        self.dense_scale = nn.Sequential(
            nn.Conv2d(final_scaleblock_features, final_scaleblock_features //2,
                            kernel_size=3, stride=1, padding=1),
            nn.ReLU(False),
            nn.Conv2d( final_scaleblock_features //2 , 1, kernel_size=1,
                        stride=1, padding=0),
            nn.Identity(),
        )

        self.scale_conf = nn.Sequential(
            nn.Conv2d(final_scaleblock_features, final_scaleblock_features //2,
                            kernel_size=3, stride=1, padding=1),
            nn.ReLU(False),
            nn.Conv2d( final_scaleblock_features //2 , 1, kernel_size=1,
                        stride=1, padding=0),
            nn.Sigmoid(),
        )
        
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
        rel_depth, decoder_features_list= self.depth_head(features, patch_h, patch_w, prompt_depth)
        
        estimator = LeastSquaresEstimatorTorch(rel_depth, sparse_depth_inv, sparse_mask)
        estimator.compute_scale() 
        estimator.apply_scale()
        estimator.clamp_min_max(clamp_min=self.min_pred,clamp_max=100.0)
        ga_result = estimator.output
        d = estimator.output.clone()

        ga_mask = (ga_result >0).float()
        sparse_and_ga_mask  = sparse_mask * ga_mask

        scale_residual = torch.zeros_like(sparse_depth_inv)
        scale_residual[sparse_and_ga_mask.bool()] = sparse_depth_inv[sparse_and_ga_mask.bool()] / ga_result[sparse_and_ga_mask.bool()]
        # show_images_three_sources(ga_result, sparse_depth_inv, scale_residual)
        
        scale_residual_np = scale_residual.cpu().numpy() 
        sparse_mask_np = sparse_and_ga_mask.bool().cpu().numpy() 
        int_scale_residual_batch = []
        for i in range(scale_residual.shape[0]):

            ScaleMapInterpolator = AnchorInterpolator2D(
                sparse_depth = np.squeeze(scale_residual_np[i], axis = 0),
                valid = np.squeeze(sparse_mask_np[i], axis = 0),
            )
            ScaleMapInterpolator.generate_interpolated_scale_map(
                interpolate_method='linear', 
                fill_corners=False
            )
            
            int_scale_residual_batch.append(ScaleMapInterpolator.interpolated_scale_map.astype(np.float32))

        int_scale_residual_batch = np.stack(int_scale_residual_batch)
        int_scale_residual = torch.from_numpy(int_scale_residual_batch).float().to(prompt_depth.device) 
        int_scale_residual = int_scale_residual.unsqueeze(1)

        # show_images_two_sources(scale_residual, int_scale_residual)
        new_size_list = [(24,32), (48,64), (96,128), (192, 256)]
       
        #this resized sparse scale residual map contains zeros, crrently I fill ones inside the scale residual block
        # resized_sparse_scale = resize_sparse_depth(scale_residual, new_size_list)
        resized_sparse_scale = []
        for new_size in new_size_list:
            scale_resized = nn.functional.interpolate(
            int_scale_residual, new_size, mode="bilinear", align_corners=True)
            resized_sparse_scale.append(scale_resized)
        # show_images_four_sources(resized_sparse_scale[0], resized_sparse_scale[1], resized_sparse_scale[2], resized_sparse_scale[3])

        # resized_sparse_mask = []
        # for sparse_scale in resized_sparse_scale:
        #     sparse_mask_temp = sparse_scale > 0
        #     resized_sparse_mask.append(sparse_mask_temp)

        resized_ga = []
        for new_size in new_size_list:
            ga_resized = nn.functional.interpolate(
            ga_result, new_size, mode="bilinear", align_corners=True)
            resized_ga.append(ga_resized)
        # show_images_four_sources(resized_ga[0], resized_ga[1], resized_ga[2], resized_ga[3])
        
        scale_features_4 = self.scale_block_4(decoder_features_list[0], resized_ga[0], resized_sparse_scale[0], size=[48,64])
        scale_features_3 = self.scale_block_3(decoder_features_list[1], resized_ga[1], resized_sparse_scale[1],  
                                              previous_embedding = scale_features_4, size=[96,128])
        scale_features_2 = self.scale_block_2(decoder_features_list[2], resized_ga[2], resized_sparse_scale[2], 
                                              previous_embedding = scale_features_3, size=[192,256])
        scale_features_1 = self.scale_block_1(decoder_features_list[3], resized_ga[3], resized_sparse_scale[3],
                                              previous_embedding = scale_features_2, size=[336,448])
        scale_output_features = self.scale_block_0(decoder_features_list[4], ga_result, scale_residual, previous_embedding = scale_features_1)

        scale_output = self.dense_scale(scale_output_features)

        scale_conf = self.scale_conf(scale_output_features)

        scale_map = F.relu(1.0 + scale_output)

        pred = d * scale_map

        if self.min_pred is not None:
            min_pred_inv = 1.0/self.min_pred
            pred = torch.where(pred > min_pred_inv, min_pred_inv, pred)
            # intermedian_pred[intermedian_pred > min_pred_inv] = min_pred_inv
        if self.max_pred is not None:
            max_pred_inv = 1.0/self.max_pred
            pred = torch.where(pred < max_pred_inv, max_pred_inv, pred)
            # intermedian_pred[intermedian_pred < max_pred_inv] = max_pred_inv


        # show_images_two_sources(pred, conf)
        output = dict(metric_depth=1.0/pred)
        output['confidence'] = scale_conf
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
