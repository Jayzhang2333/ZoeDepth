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
from zoedepth.models.depth_model import DepthModel
from zoedepth.models.base_models.midas import MidasCore
from zoedepth.models.layers.attractor import AttractorLayer, AttractorLayerUnnormed
from zoedepth.models.layers.dist_layers import ConditionalLogBinomial
from zoedepth.models.layers.localbins_layers import (Projector, SeedBinRegressor,
                                            SeedBinRegressorUnnormed, PriorEmbeddingLayer)

from zoedepth.models.layers.global_alignment import LeastSquaresEstimatorTorch
from zoedepth.models.layers.fusion_layers import ScaleAndPlaceModule, GateConv
from zoedepth.models.midas.midas_net_custom import MidasNet_small_videpth_original
from zoedepth.models.model_io import load_state_from_resource
import torch.nn.functional as F
from zoedepth.models.layers.global_alignment import LeastSquaresEstimator, Interpolator2D, Interpolator2D_original

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

    for idx in range(batch_size):
        # Display source1 image
        axes[idx][0].imshow(images1[idx])
        axes[idx][0].set_title("Source 1")
        axes[idx][0].axis('off')

        # Display source2 image
        axes[idx][1].imshow(images2[idx])
        axes[idx][1].set_title("Source 2")
        axes[idx][1].axis('off')

    plt.tight_layout()
    plt.show()

def adapt_pool(m, dep):
    dep = F.avg_pool2d(dep, 3, 2, 1)
    m = F.avg_pool2d(m.float(), 3, 2, 1)
    dep = (m > 0) * dep / (m + 1e-8)

    return dep


def get_sparse_pool(dep, num):
    masks, depths = [], []
    m = dep > 0
    masks.append(m)
    depths.append(dep)
    for _ in range(num):
        dep = adapt_pool(m, dep)
        m = dep > 0
        masks.append(m)
        depths.append(dep)
    
    # for i in range(num+1):
    #     depths[i][~masks[i]] = 1

    return masks, depths


class ZoeDepth_videpth(DepthModel):
    def __init__(self, core,  prior_channel = 2, n_bins=64, bin_centers_type="softplus", bin_embedding_dim=128, min_depth=1e-3, max_depth=10,
                 n_attractors=[16, 8, 4, 1], attractor_alpha=300, attractor_gamma=2, attractor_kind='sum', attractor_type='exp', min_temp=5, max_temp=50, train_midas=True,
                 midas_lr_factor=10, encoder_lr_factor=10, pos_enc_lr_factor=10, inverse_midas=False, **kwargs):
        """ZoeDepth model. This is the version of ZoeDepth that has a single metric head

        Args:
            core (models.base_models.midas.MidasCore): The base midas model that is used for extraction of "relative" features
            n_bins (int, optional): Number of bin centers. Defaults to 64.
            bin_centers_type (str, optional): "normed" or "softplus". Activation type used for bin centers. For "normed" bin centers, linear normalization trick is applied. This results in bounded bin centers.
                                               For "softplus", softplus activation is used and thus are unbounded. Defaults to "softplus".
            bin_embedding_dim (int, optional): bin embedding dimension. Defaults to 128.
            min_depth (float, optional): Lower bound for normed bin centers. Defaults to 1e-3.
            max_depth (float, optional): Upper bound for normed bin centers. Defaults to 10.
            n_attractors (List[int], optional): Number of bin attractors at decoder layers. Defaults to [16, 8, 4, 1].
            attractor_alpha (int, optional): Proportional attractor strength. Refer to models.layers.attractor for more details. Defaults to 300.
            attractor_gamma (int, optional): Exponential attractor strength. Refer to models.layers.attractor for more details. Defaults to 2.
            attractor_kind (str, optional): Attraction aggregation "sum" or "mean". Defaults to 'sum'.
            attractor_type (str, optional): Type of attractor to use; "inv" (Inverse attractor) or "exp" (Exponential attractor). Defaults to 'exp'.
            min_temp (int, optional): Lower bound for temperature of output probability distribution. Defaults to 5.
            max_temp (int, optional): Upper bound for temperature of output probability distribution. Defaults to 50.
            train_midas (bool, optional): Whether to train "core", the base midas model. Defaults to True.
            midas_lr_factor (int, optional): Learning rate reduction factor for base midas model except its encoder and positional encodings. Defaults to 10.
            encoder_lr_factor (int, optional): Learning rate reduction factor for the encoder in midas model. Defaults to 10.
            pos_enc_lr_factor (int, optional): Learning rate reduction factor for positional encodings in the base midas model. Defaults to 10.
        """
        super().__init__()

        self.core = core
        self.max_depth = 18.0
        self.min_depth = 0.1
        self.min_temp = min_temp
        self.bin_centers_type = bin_centers_type

        self.midas_lr_factor = midas_lr_factor
        self.encoder_lr_factor = encoder_lr_factor
        self.pos_enc_lr_factor = pos_enc_lr_factor
        self.train_midas = train_midas
        self.inverse_midas = inverse_midas

        if self.encoder_lr_factor <= 0:
            self.core.freeze_encoder(
                freeze_rel_pos=self.pos_enc_lr_factor <= 0)

        N_MIDAS_OUT = 32
        btlnck_features = self.core.output_channels[0]
        num_out_features = self.core.output_channels[1:]
        print(f"other decoder channel number: {num_out_features}")

        self.ScaleMapLearner = MidasNet_small_videpth_original(in_channels = 2, min_pred=self.min_depth, max_pred=self.max_depth)

     

    def forward(self, x, sparse_feature = None, denorm=False, **kwargs):
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W)
            return_final_centers (bool, optional): Whether to return the final bin centers. Defaults to False.
            denorm (bool, optional): Whether to denormalize the input image. This reverses ImageNet normalization as midas normalization is different. Defaults to False.
            return_probs (bool, optional): Whether to return the output probability distribution. Defaults to False.
        
        Returns:
            dict: Dictionary containing the following keys:
                - rel_depth (torch.Tensor): Relative depth map of shape (B, H, W)
                - metric_depth (torch.Tensor): Metric depth map of shape (B, 1, H, W)
                - bin_centers (torch.Tensor): Bin centers of shape (B, n_bins). Present only if return_final_centers is True
                - probs (torch.Tensor): Output probability distribution of shape (B, n_bins, H, W). Present only if return_probs is True

        """
        b, c, h, w = x.shape
        # print("input shape ", x.shape)
        self.orig_input_width = w
        self.orig_input_height = h
        # print(f" image shape is: {x.shape}")
        rel_depth, out = self.core(x, denorm=denorm, return_rel_depth=True)

        # print(f'relative depth shape is {rel_depth.shape}')
        # print(f'sparse feature shape is {sparse_feature.shape}')
        rel_depth_np = rel_depth.unsqueeze(1).cpu().numpy()
       

        sparse_feature_np = sparse_feature.cpu().numpy() 

        batch_size = rel_depth_np.shape[0]
        int_depth_batch = []
        int_scaffolding_batch = []

        for i in range(batch_size):
        #     # Extract individual depth maps and priors
            rel_depth_single = rel_depth_np[i]
            rel_depth_single = np.squeeze(rel_depth_single, axis=0)
           
            sparse_feature_single = np.squeeze(sparse_feature_np[i], axis=0)

            input_sparse_depth_valid = (sparse_feature_single < self.max_depth) * (sparse_feature_single > self.min_depth)
            input_sparse_depth_valid = input_sparse_depth_valid.astype(bool)
            valid_pixel_count = input_sparse_depth_valid.sum()
            
            sparse_feature_single_inv = sparse_feature_single.copy()
            sparse_feature_single_inv[~input_sparse_depth_valid] = np.inf
            sparse_feature_single_inv = 1.0/sparse_feature_single_inv
            # sparse_feature_single[~input_sparse_depth_valid] = np.inf
            # sparse_feature_single = 1.0/sparse_feature_single
            GlobalAlignment = LeastSquaresEstimator(
                estimate=rel_depth_single,
                target=sparse_feature_single_inv,
                valid=input_sparse_depth_valid
            )
            GlobalAlignment.compute_scale_and_shift()
            # GlobalAlignment.compute_scale()
            GlobalAlignment.apply_scale_and_shift()
            GlobalAlignment.clamp_min_max(clamp_min=self.min_depth, clamp_max=self.max_depth)
            
            # Store the output depth map
            int_depth_batch.append(GlobalAlignment.output.astype(np.float32))

            ScaleMapInterpolator = Interpolator2D_original(
                pred_inv = int_depth_batch[-1],
                sparse_depth_inv = sparse_feature_single_inv,
                valid = input_sparse_depth_valid,
            )
            ScaleMapInterpolator.generate_interpolated_scale_map(
                interpolate_method='linear', 
                fill_corners=False
            )
            
            int_scaffolding_batch.append(ScaleMapInterpolator.interpolated_scale_map.astype(np.float32))
            

            
      
        int_depth_batch = np.stack(int_depth_batch)
        int_scaffolding_batch = np.stack(int_scaffolding_batch)
      

        int_depth = torch.from_numpy(int_depth_batch).float().to(sparse_feature.device)
        int_scaffolding = torch.from_numpy(int_scaffolding_batch).float().to(sparse_feature.device) 
      
        int_depth = int_depth.unsqueeze(1)
        int_scaffolding = int_scaffolding.unsqueeze(1)

        # print(f'ga shape is {int_depth.shape}')
        # print(f'scale residual shape is {int_scaffolding.shape}')
        # show_images_three_sources(rel_depth.unsqueeze(1), int_depth, int_scaffolding)
        
        prior_map = torch.cat([int_depth,int_scaffolding ], dim = 1)
        pred, scales = self.ScaleMapLearner(prior_map, int_depth)


        output = dict(metric_depth=1.0/pred)
        # show_images(1.0/pred)
        # if return_final_centers or return_probs:
        #     output['bin_centers'] = b_centers

        # if return_probs:
        #     output['probs'] = probabilities

        return output

    def get_lr_params(self, lr):
        """
        Learning rate configuration for different layers of the model
        Args:
            lr (float) : Base learning rate
        Returns:
            list : list of parameters to optimize and their learning rates, in the format required by torch optimizers.
        """
        param_conf = []
        if self.train_midas:
            if self.encoder_lr_factor > 0:
                param_conf.append({'params': self.core.get_enc_params_except_rel_pos(
                ), 'lr': lr / self.encoder_lr_factor})

            if self.pos_enc_lr_factor > 0:
                param_conf.append(
                    {'params': self.core.get_rel_pos_params(), 'lr': lr / self.pos_enc_lr_factor})

            midas_params = self.core.core.scratch.parameters()
            midas_lr_factor = self.midas_lr_factor
            param_conf.append(
                {'params': midas_params, 'lr': lr / midas_lr_factor})

        remaining_modules = []
        for name, child in self.named_children():
            if name != 'core':
                remaining_modules.append(child)
        remaining_params = itertools.chain(
            *[child.parameters() for child in remaining_modules])

        param_conf.append({'params': remaining_params, 'lr': lr})

        return param_conf

    @staticmethod
    def build(midas_model_type="DPT_BEiT_L_384", pretrained_resource=None, use_pretrained_midas=False, train_midas=False, freeze_midas_bn=True, **kwargs):
        core = MidasCore.build(midas_model_type=midas_model_type, use_pretrained_midas=use_pretrained_midas,
                               train_midas=train_midas, fetch_features=True, freeze_bn=freeze_midas_bn, **kwargs)
        model = ZoeDepth_videpth(core, **kwargs)
        if pretrained_resource:
            assert isinstance(pretrained_resource, str), "pretrained_resource must be a string"
            model = load_state_from_resource(model, pretrained_resource)
        return model

    @staticmethod
    def build_from_config(config):
        return ZoeDepth_videpth.build(**config)
