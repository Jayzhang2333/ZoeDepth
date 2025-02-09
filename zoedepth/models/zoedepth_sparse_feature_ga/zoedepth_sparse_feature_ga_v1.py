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

from zoedepth.models.layers.global_alignment import LeastSquaresEstimator, Interpolator2D
from zoedepth.models.model_io import load_state_from_resource
from zoedepth.data.data_mono import get_depth_prior_from_features
from scipy import ndimage
from scipy.interpolate import RBFInterpolator

def interpolate_depth_map(sparse_depth_map):
    # Get the indices of the known depth pixels
    known_points = np.argwhere(sparse_depth_map > 0)
    known_values = sparse_depth_map[sparse_depth_map > 0]

    if known_points.shape[0] == 0:
        raise ValueError("No known depth values to interpolate from.")

    # Create a grid for the interpolation
    grid_x, grid_y = np.meshgrid(np.arange(sparse_depth_map.shape[1]),
                                 np.arange(sparse_depth_map.shape[0]))

    # Reshape the grid for interpolation
    grid_points = np.vstack((grid_y.ravel(), grid_x.ravel())).T

    # Perform RBF interpolation
    rbf_interpolator = RBFInterpolator(known_points, known_values, kernel='cubic')
    interpolated_values = rbf_interpolator(grid_points)

    # Reshape the interpolated values back to the grid shape
    filled_depth_map = interpolated_values.reshape(sparse_depth_map.shape)

    return filled_depth_map

def fill_sparse_depth_map(sparse_depth_map):
    # Create a mask where depth is zero (these are the pixels to fill)
    mask = sparse_depth_map == 0
    
    # Compute the distance transform and get indices of nearest non-zero pixels
    distance, indices = ndimage.distance_transform_edt(mask, 
                                                       return_distances=True, 
                                                       return_indices=True)
    
    # Use the indices to map the nearest non-zero depth values to the zero-depth pixels
    filled_depth_map = sparse_depth_map[tuple(indices)]
    
    return filled_depth_map


def show_images(tensor_images):
    tensor_images = tensor_images.detach().cpu().numpy()  # Convert to numpy if tensor
    tensor_images = np.transpose(tensor_images, (0, 2, 3, 1))  # Change from CHW to HWC
    
    # Display the images
    batch_size = tensor_images.shape[0]
    fig, axes = plt.subplots(1, batch_size, figsize=(15, 5))  # Adjust number of subplots as needed
    if batch_size == 1:  # Handle case where batch size is 1
        axes = [axes]
    
    for idx in range(batch_size):
        axes[idx].imshow(np.clip(tensor_images[idx], 0, 1))  # Assuming images are normalized [0, 1]
        axes[idx].axis('off')
    
    plt.show()


class ZoeDepth_sparse_feature_ga(DepthModel):
    def __init__(self, core,  n_bins=64, bin_centers_type="softplus", bin_embedding_dim=128, min_depth=1e-3, max_depth=10,
                 n_attractors=[16, 8, 4, 1], attractor_alpha=300, attractor_gamma=2, attractor_kind='sum', attractor_type='exp', prior_channels = 0, min_temp=5, max_temp=50, train_midas=True,
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
        self.max_depth = max_depth
        self.min_depth = min_depth
        # print(self.max_depth)
        # print(self.min_depth)
        self.min_temp = min_temp
        self.bin_centers_type = bin_centers_type

        print(f"prior channels are: {prior_channels}")
        self.prior_channels = prior_channels

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
        # print(f"bottleneck channel number: {btlnck_features}")
        num_out_features = self.core.output_channels[1:]
        print(f"other decoder channel number: {num_out_features}")

        # add the prior channels
        self.conv2 = nn.Conv2d(btlnck_features  , btlnck_features ,
                               kernel_size=1, stride=1, padding=0)  # btlnck conv

        if bin_centers_type == "normed":
            SeedBinRegressorLayer = SeedBinRegressor
            Attractor = AttractorLayer
        elif bin_centers_type == "softplus":
            SeedBinRegressorLayer = SeedBinRegressorUnnormed
            Attractor = AttractorLayerUnnormed
        elif bin_centers_type == "hybrid1":
            SeedBinRegressorLayer = SeedBinRegressor
            Attractor = AttractorLayerUnnormed
        elif bin_centers_type == "hybrid2":
            SeedBinRegressorLayer = SeedBinRegressorUnnormed
            Attractor = AttractorLayer
        else:
            raise ValueError(
                "bin_centers_type should be one of 'normed', 'softplus', 'hybrid1', 'hybrid2'")

        # add prior channels
        self.seed_bin_regressor = SeedBinRegressorLayer(
            bin_embedding_dim, n_bins=n_bins, min_depth=min_depth, max_depth=max_depth)
        
        # self.seed_projector = Projector(btlnck_features, bin_embedding_dim-64)
        self.seed_projector = Projector(btlnck_features, bin_embedding_dim//2)

        # self.first_prior_embedding_layer = PriorEmbeddingLayer(in_features = prior_channels, embedding_dim = 4)
        self.first_prior_embedding_layer = PriorEmbeddingLayer(in_features =  prior_channels, embedding_dim = bin_embedding_dim//2)
        dimension_list = [16,64,64,64]
        pre_dimension_list = [4, 16,64,64]
        pre_dimension_list_inverse = [64, 64, 16, 4]
        # self.priorembeddinglayers = nn.ModuleList([
        #     PriorEmbeddingLayer(in_features = pre_dimension_list[i], embedding_dim = dimension_list[i])
        #     for i in range(len(num_out_features))
        # ])
        self.priorembeddinglayers = nn.ModuleList([
            PriorEmbeddingLayer(in_features = bin_embedding_dim//2, embedding_dim = bin_embedding_dim//2)
            for i in range(len(num_out_features))
        ])

        # self.projectors = nn.ModuleList([
        #     Projector(num_out_features[i], bin_embedding_dim - pre_dimension_list_inverse[i] )
        #     for i in range(len(num_out_features))
        # ])

        self.projectors = nn.ModuleList([
            Projector(num_out_features[i], bin_embedding_dim//2 )
            for i in range(len(num_out_features))
        ])

        self.attractors = nn.ModuleList([
            Attractor(bin_embedding_dim , n_bins, n_attractors=n_attractors[i], min_depth=min_depth, max_depth=max_depth,
                      alpha=attractor_alpha, gamma=attractor_gamma, kind=attractor_kind, attractor_type=attractor_type)
            for i in range(len(num_out_features))
        ])

        

        last_in = N_MIDAS_OUT + 1  # +1 for relative depth

        # use log binomial instead of softmax
        self.conditional_log_binomial = ConditionalLogBinomial(
            last_in, bin_embedding_dim, n_classes=n_bins, min_temp=min_temp, max_temp=max_temp)

    def forward(self, x, sparse_feature = None, return_final_centers=False, denorm=False, return_probs=False, input_height = 480, input_width = 640, **kwargs):
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
        rel_depth, out = self.core(x, denorm=denorm, return_rel_depth=True)

        # show_images(x)

        # print("output shapes", out[0].shape)

        # rel_depth_np = rel_depth.cpu().numpy()
        rel_depth_np = torch.nn.functional.interpolate(
                    rel_depth.unsqueeze(1),
                    size=(input_height, input_width),
                    mode="bilinear",
                    align_corners=True,
                ).cpu().numpy()
        # print(torch.max(sparse_feature[0]))
        # print(torch.min(sparse_feature[0][sparse_feature[0] > 0]))

        # print("torch sparse map shape:",sparse_feature.shape)
        sparse_feature_np = sparse_feature.cpu().numpy() 
        # print(np.shape(sparse_feature_np))
        # print(np.max(sparse_feature_np))

        # sparse_feature_np[~input_sparse_depth_valid] = np.inf # set invalid depth

        batch_size = rel_depth_np.shape[0]
        int_depth_batch = []
        int_depth_no_shift_batch = []
        int_scaffolding_batch = []
        nearest_prior_batch = []
        interpolation_prior_batch = []
        int_scaffolding_no_shift_batch = []
        # prior_map_batch = []

        for i in range(batch_size):
        #     # Extract individual depth maps and priors
            rel_depth_single = rel_depth_np[i]
            rel_depth_single = np.squeeze(rel_depth_single, axis=0)
            # plt.imshow(rel_depth_single, cmap='viridis')  # Use 'viridis' or any other colormap you prefer
            # plt.colorbar(label='Depth')  # Optional: Adds a color bar for reference
            # plt.title("Int Depth Visualization")
            # plt.show()
            sparse_feature_single = np.squeeze(sparse_feature_np[i], axis=0)

            # nearest_pior = fill_sparse_depth_map(sparse_feature_single)
            # nearest_prior_batch.append(nearest_pior.astype(np.float32))
            # interpolation = interpolate_depth_map(sparse_feature_single)
            # interpolation_prior_batch.append(interpolation.astype(np.float32))
            
            
            # sparse_feature_single = sparse_feature_single/1.44
        #     print(np.shape(sparse_feature_single))
        #     print(np.max(sparse_feature_single))
        #     print(np.min(sparse_feature_single[sparse_feature_single>0.5]))

            input_sparse_depth_valid = (sparse_feature_single < self.max_depth) * (sparse_feature_single > self.min_depth)
            input_sparse_depth_valid = input_sparse_depth_valid.astype(bool)
            valid_pixel_count = input_sparse_depth_valid.sum()
            # print(f"Number of True elements in input_sparse_depth_valid: {valid_pixel_count}")


            
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
            int_depth_batch.append(1.0/GlobalAlignment.output.astype(np.float32))

            GlobalAlignment_scale = LeastSquaresEstimator(
                estimate=rel_depth_single,
                target=sparse_feature_single_inv,
                valid=input_sparse_depth_valid
            )
            GlobalAlignment_scale.compute_scale()
            # GlobalAlignment.compute_scale()
            GlobalAlignment_scale.apply_scale()
            GlobalAlignment_scale.clamp_min_max(clamp_min=self.min_depth, clamp_max=self.max_depth)
            ga_no_shift = np.full_like(GlobalAlignment_scale.output, fill_value=self.max_depth)

            # Perform division where the denominator is not zero
            np.divide(1.0, GlobalAlignment_scale.output, out=ga_no_shift, where=GlobalAlignment.output_no_shift != 0)
            # Store the output depth map
            int_depth_no_shift_batch.append(ga_no_shift.astype(np.float32))



            # Initialize the result array with the desired value
            # ga_no_shift = np.full_like(GlobalAlignment.output_no_shift, fill_value=self.max_depth)

            # Perform division where the denominator is not zero
            # np.divide(1.0, GlobalAlignment.output_no_shift, out=ga_no_shift, where=GlobalAlignment.output_no_shift != 0)
            # int_depth_no_shift_batch.append(ga_no_shift.astype(np.float32))

            # plt.imshow(test, cmap='viridis')  # Use 'viridis' or any other colormap you prefer
            # plt.colorbar(label='Depth')  # Optional: Adds a color bar for reference
            # plt.title("sparse depth")
            # plt.show()

            # assert (np.sum(input_sparse_depth_valid) >= 4), "not enough valid sparse points"
            # # print(np.min(sparse_feature_single[sparse_feature_single>0]))
            # # print(np.max(sparse_feature_single))
            ResidualMapInterpolator = Interpolator2D(
                pred = int_depth_batch[-1], #ga result
                sparse_depth = sparse_feature_single,
                valid = input_sparse_depth_valid,
            )
            ResidualMapInterpolator.generate_interpolated_residual_map(
                interpolate_method='linear', 
                fill_corners=False
            )
            
            int_scaffolding_batch.append(ResidualMapInterpolator.interpolated_residual_map.astype(np.float32))
            

            ResidualMapInterpolator_no_shift = Interpolator2D(
                pred = int_depth_no_shift_batch[-1], #ga result
                sparse_depth = sparse_feature_single,
                valid = input_sparse_depth_valid,
            )
            ResidualMapInterpolator_no_shift.generate_interpolated_residual_map(
                interpolate_method='linear', 
                fill_corners=False
            )
            
            int_scaffolding_no_shift_batch.append(ResidualMapInterpolator_no_shift.interpolated_residual_map.astype(np.float32))
            
            # fig, axes = plt.subplots(1, 4, figsize=(12, 6))  # Adjust figsize as needed

            # # Plot the first image on the first subplot
            # im1 = axes[0].imshow(nearest_prior_batch[-1], cmap='viridis')
            # axes[0].set_title("Nearest Neighbor")
            # fig.colorbar(im1, ax=axes[0], label='Depth')  # Adds a color bar to the first subplot

            # # Plot the second image on the second subplot
            # im2 = axes[1].imshow(interpolation_prior_batch[-1], cmap='viridis')  # Replace 'second_image' with your second image
            # axes[1].set_title("Interpolation")
            # fig.colorbar(im2, ax=axes[1], label='Depth')  # Adds a color bar to the second subplot

            # # Plot the second image on the second subplot
            # im3 = axes[2].imshow(int_depth_batch[-1], cmap='viridis')  # Replace 'second_image' with your second image
            # axes[2].set_title("GA")
            # fig.colorbar(im3, ax=axes[2], label='Depth')  # Adds a color bar to the second subplot

            # # Plot the second image on the second subplot
            # im4 = axes[3].imshow(int_depth_no_shift_batch[-1], cmap='viridis')  # Replace 'second_image' with your second image
            # axes[3].set_title("GA no shift")
            # fig.colorbar(im4, ax=axes[3], label='Depth')  # Adds a color bar to the second subplot

            # # Optional: Adjust layout to prevent overlap
            # plt.tight_layout()

            # # Display the figure
            # plt.show()

            # plt.imshow(int_depth_batch[-1], cmap='viridis')  # Use 'viridis' or any other colormap you prefer
            # plt.colorbar(label='Depth')  # Optional: Adds a color bar for reference
            # plt.title("Int Depth Visualization")
            # plt.show()
            # fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize as needed

            # # Plot the first image on the first subplot
            # im1 = axes[0].imshow(int_depth_batch[-1], cmap='viridis')
            # axes[0].set_title("GA scale and shift")
            # fig.colorbar(im1, ax=axes[0], label='Depth')  # Adds a color bar to the first subplot

            # # Plot the second image on the second subplot
            # im2 = axes[1].imshow(1.0/GlobalAlignment_scale.output.astype(np.float32), cmap='viridis')  # Replace 'second_image' with your second image
            # axes[1].set_title("GA scale")
            # fig.colorbar(im2, ax=axes[1], label='Depth')  # Adds a color bar to the second subplot

            # # Optional: Adjust layout to prevent overlap
            # plt.tight_layout()

            # # Display the figure
            # plt.show()

        # print("ga results before stack", np.shape(int_depth_batch))
        int_depth_batch = np.stack(int_depth_batch)
        int_depth_no_shift_batch = np.stack(int_depth_no_shift_batch)
        int_scaffolding_batch = np.stack(int_scaffolding_batch)
        int_scaffolding_no_shift_batch = np.stack(int_scaffolding_no_shift_batch)
        # nearest_prior_batch = np.stack(nearest_prior_batch)
        # interpolation_prior_batch = np.stack(interpolation_prior_batch)
        # prior_map_batch = np.stack(prior_map_batch)
        # print("ga result after stack", np.shape(int_depth_batch))

        int_depth = torch.from_numpy(int_depth_batch).float().to(sparse_feature.device)  # Ensure same device
        int_depth_no_shift = torch.from_numpy(int_depth_no_shift_batch).float().to(sparse_feature.device) 
        int_scaffolding = torch.from_numpy(int_scaffolding_batch).float().to(sparse_feature.device) 
        int_scaffolding_no_shift = torch.from_numpy(int_scaffolding_no_shift_batch).float().to(sparse_feature.device) 
        # nearest_prior = torch.from_numpy(nearest_prior_batch).float().to(sparse_feature.device) 
        # interpolation_prior = torch.from_numpy(interpolation_prior_batch).float().to(sparse_feature.device) 
        # prior_map = torch.from_numpy(prior_map_batch).float().to(sparse_feature.device) 
        # prior_map = prior_map.permute(0, 3, 1, 2)
        # print(prior_map.shape)
        # print("ga result after to troch", int_depth.shape)
        int_depth = int_depth.unsqueeze(1)
        int_depth_no_shift = int_depth_no_shift.unsqueeze(1)
        int_scaffolding = int_scaffolding.unsqueeze(1)
        int_scaffolding_no_shift = int_scaffolding_no_shift.unsqueeze(1)
        # nearest_prior = nearest_prior.unsqueeze(1)
        # interpolation_prior = interpolation_prior.unsqueeze(1)
        # print("ga result after unsquezze", int_depth.shape)
        


        # rel_depth_unsqueeze = rel_depth.unsqueeze(1)
        # int_depth = torch.cat([int_depth, rel_depth_unsqueeze], dim = 1)
        int_depth = torch.nn.functional.interpolate(
                    int_depth,
                    size=(self.orig_input_height, self.orig_input_width),
                    mode="bilinear",
                    align_corners=True,
                )
        
        int_depth_no_shift = torch.nn.functional.interpolate(
                    int_depth_no_shift,
                    size=(self.orig_input_height, self.orig_input_width),
                    mode="bilinear",
                    align_corners=True,
                )
        
        int_scaffolding = torch.nn.functional.interpolate(
                    int_scaffolding,
                    size=(self.orig_input_height, self.orig_input_width),
                    mode="bilinear",
                    align_corners=True,
                )
        
        int_scaffolding_no_shift = torch.nn.functional.interpolate(
                    int_scaffolding_no_shift,
                    size=(self.orig_input_height, self.orig_input_width),
                    mode="bilinear",
                    align_corners=True,
                )
        
        # nearest_prior = torch.nn.functional.interpolate(
        #             nearest_prior,
        #             size=(self.orig_input_height, self.orig_input_width),
        #             mode="bilinear",
        #             align_corners=True,
        #         )
        # interpolation_prior = torch.nn.functional.interpolate(
        #             interpolation_prior,
        #             size=(self.orig_input_height, self.orig_input_width),
        #             mode="bilinear",
        #             align_corners=True,
        #         )
        
        # prior_map = torch.nn.functional.interpolate(
        #             prior_map,
        #             size=(self.orig_input_height, self.orig_input_width),
        #             mode="bilinear",
        #             align_corners=True,
        #         )
        # print("ga result after interpolation", int_depth.shape)
        # print()

        # prior_map = torch.cat([int_depth, int_scaffolding], dim = 1)
        # print(int_depth.shape)
        # print(int_scaffolding.shape)
        # prior_map = torch.cat([int_depth,int_scaffolding,int_depth_no_shift, int_scaffolding_no_shift ], dim = 1)
        prior_map = torch.cat([int_depth,int_scaffolding ], dim = 1)
        # prior_map = torch.cat([prior_map, int_depth, int_scaffolding], dim = 1)
        # print(prior_map.shape)


        prior_embeddings = []
        # print(sparse_feature.shape)
        # print(rel_depth.shape)
        # rel_depth_unsqueeze = rel_depth.unsqueeze(1)
        # sparse_feature = torch.cat([sparse_feature, rel_depth_unsqueeze], dim = 1)
        prior_embeddings.append(self.first_prior_embedding_layer(prior_map))
        for prior_embedding_layer in self.priorembeddinglayers:
            prior_embeddings.insert(0, prior_embedding_layer(prior_embeddings[0]))

        outconv_activation = out[0]
        btlnck = out[1]
        # print(f"bottleneck dimension is: {btlnck.shape}")
        x_blocks = out[2:]
        # x_d0 = btlnck
        x_d0 = self.conv2(btlnck)
        

        #interpolate the sparse prior and concat with the bottle neck
        # if sparse_feature is not None:
        #     sparse_feature_scaled = nn.functional.interpolate(
        #             sparse_feature,
        #             size=[x_d0.size(2), x_d0.size(3)],
        #             mode="bilinear",
        #             align_corners=True,
        #         )
            
        #     x_d0 = torch.cat([x_d0, sparse_feature_scaled], dim = 1)
            # x = torch.cat([x_d0, sparse_feature_scaled], dim = 1)
        # else:
            # x = x_d0
        # x_d0 = self.conv2(x_d0)
        # x = x_d0
            
        b_embedding = self.seed_projector(x_d0)
        b_embedding = torch.cat([b_embedding, prior_embeddings[0]], dim = 1)
        _, seed_b_centers = self.seed_bin_regressor(b_embedding)

        if self.bin_centers_type == 'normed' or self.bin_centers_type == 'hybrid2':
            b_prev = (seed_b_centers - self.min_depth) / \
                (self.max_depth - self.min_depth)
        else:
            b_prev = seed_b_centers

        # prev_b_embedding = self.seed_projector(x)
        # ADD PRIOR TO THE EMBEDDING RAW
        prev_b_embedding = b_embedding
        # sparse_feature_scaled = nn.functional.interpolate(
        #             sparse_feature,
        #             size=[prev_b_embedding.size(2), prev_b_embedding.size(3)],
        #             mode="bilinear",
        #             align_corners=True,
        #         )
        # prev_b_embedding = torch.cat([prev_b_embedding, sparse_feature_scaled], dim = 1)

        # unroll this loop for better performance
        for idx, (projector, attractor, x) in enumerate(zip(self.projectors, self.attractors, x_blocks)):

            # if sparse_feature is not None:
            #     sparse_feature_scaled = nn.functional.interpolate(
            #             sparse_feature,
            #             size=[x.size(2), x.size(3)],
            #             mode="bilinear",
            #             align_corners=True,
            #         )
            
            #     x = torch.cat([x, sparse_feature_scaled], dim = 1)
            # print(f"decoder dimension is: {x.shape}")
            b_embedding = projector(x)
            b_embedding = torch.cat([b_embedding, prior_embeddings[idx+1]], dim = 1)
            # INSTEAD OF CREATING THE EMBEDDING WITH THE SPARSE PRIOR, 
            # THIS ADD THE PRIOR TO THE EMBEDDING
            # THE ATTRACTOR LAYER GENERATES ATTRACTOR WITH THIS PRIOR
            # if sparse_feature is not None:
            #     sparse_feature_scaled = nn.functional.interpolate(
            #             sparse_feature,
            #             size=[b_embedding.size(2), b_embedding.size(3)],
            #             mode="bilinear",
            #             align_corners=True,
            #         )
            
                # b_embedding = torch.cat([b_embedding, sparse_feature_scaled], dim = 1)

            b, b_centers = attractor(
                b_embedding, b_prev, prev_b_embedding = prev_b_embedding, interpolate=True)
            b_prev = b.clone()
            prev_b_embedding = b_embedding.clone()

        last = outconv_activation

        if self.inverse_midas:
            # invert depth followed by normalization
            rel_depth = 1.0 / (rel_depth + 1e-6)
            rel_depth = (rel_depth - rel_depth.min()) / \
                (rel_depth.max() - rel_depth.min())
        # concat rel depth with last. First interpolate rel depth to last size
        rel_cond = rel_depth.unsqueeze(1)
        rel_cond = nn.functional.interpolate(
            rel_cond, size=last.shape[2:], mode='bilinear', align_corners=True)
        last = torch.cat([last, rel_cond], dim=1)

        b_embedding = nn.functional.interpolate(
            b_embedding, last.shape[-2:], mode='bilinear', align_corners=True)
        x = self.conditional_log_binomial(last, b_embedding)

        # Now depth value is Sum px * cx , where cx are bin_centers from the last bin tensor
        # print(x.shape, b_centers.shape)
        b_centers = nn.functional.interpolate(
            b_centers, x.shape[-2:], mode='bilinear', align_corners=True)
        out = torch.sum(x * b_centers, dim=1, keepdim=True)
        # print(out.shape)
        # print(int_depth.shape)
        # print(self.max_depth)
        # mask = int_depth == self.max_depth
        # out[mask] = self.max_depth


        # Structure output dict
        # this is also a way to define a dictionary
        output = dict(metric_depth=out)
        if return_final_centers or return_probs:
            output['bin_centers'] = b_centers

        if return_probs:
            output['probs'] = x

        # output['rel'] = rel_depth

        
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
    def build(midas_model_type="DPT_BEiT_L_512", pretrained_resource=None, use_pretrained_midas=False, train_midas=False, freeze_midas_bn=True, **kwargs):
        core = MidasCore.build(midas_model_type=midas_model_type, use_pretrained_midas=use_pretrained_midas,
                               train_midas=train_midas, fetch_features=True, freeze_bn=freeze_midas_bn, **kwargs)
        model = ZoeDepth_sparse_feature_ga(core, **kwargs)
        if pretrained_resource:
            assert isinstance(pretrained_resource, str), "pretrained_resource must be a string"
            model = load_state_from_resource(model, pretrained_resource)
        return model

    @staticmethod
    def build_from_config(config):
        return ZoeDepth_sparse_feature_ga.build(**config)
