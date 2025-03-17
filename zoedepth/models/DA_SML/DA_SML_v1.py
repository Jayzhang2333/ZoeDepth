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
from zoedepth.models.DepthAnything.dpt import DPTHead
from zoedepth.models.DepthAnything.dinov2 import DINOv2
from zoedepth.models.depth_model import DepthModel
from zoedepth.models.model_io import load_state_from_resource
import torch.nn.functional as F
import os
import wandb
from zoedepth.models.midas.midas_net_custom import ScaleMapLearnerDA
from zoedepth.models.layers.global_alignment import LeastSquaresEstimator, Interpolator2D_original

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], 'layer_idxs': [2, 5, 8, 11]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768], 'layer_idxs': [2, 5, 8, 11]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'layer_idxs': [4, 11, 17, 23]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536], 'layer_idxs': [9, 19, 29, 39]}
}

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

class DA_SML(DepthModel):
    patch_size = 14  # patch size of the pretrained dinov2 model
    use_bn = False
    use_clstoken = False
    output_act = 'identity'

    def __init__(self,
                 da_pretrained_resource,
                 **kwargs):
        super().__init__()

        self.min_pred = 0.1
        self.max_pred = 20.0
        self.min_pred_inv = 1.0/self.min_pred
        self.max_pred_inv = 1.0/self.max_pred
        self.train_dino = kwargs['train_dino']
        self.train_DPTHead = kwargs['train_DPTHead']
        self.dino_lr_factor = kwargs['dino_lr_factor']
        model_config = model_configs[kwargs['da_model_type']]

        self.encoder = kwargs['da_model_type']
        self.model_config = model_config
        
        self.pretrained = DINOv2(model_name=kwargs['da_model_type'])
        if not self.train_dino:
            for param in self.pretrained.parameters():
                param.requires_grad = False
        
        dim = self.pretrained.blocks[0].attn.qkv.in_features
        self.depth_head = DPTHead(in_channels=dim,
                                  features=model_config['features'],
                                  out_channels=model_config['out_channels'],
                                  use_bn=self.use_bn,
                                  use_clstoken=self.use_clstoken,
                                  output_act=self.output_act,
                                  with_depth= False)
        if not self.train_DPTHead:
            for param in self.depth_head.parameters():
                param.requires_grad = False
        
        self.ScaleMapLearner = ScaleMapLearnerDA(in_channels = 2, min_pred=self.min_pred, max_pred=self.max_pred)
        
        

        # mean and std of the pretrained dinov2 model
        # self.register_buffer('_mean', torch.tensor(
        #     [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # self.register_buffer('_std', torch.tensor(
        #     [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

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

        # prompt_depth, min_val, max_val = self.normalize(prompt_depth)
        # prompt_depth_valid = (prompt_depth < self.min_pred_inv) & (prompt_depth > self.max_pred_inv)
        # prompt_depth[~prompt_depth_valid] = 0


        h, w = x.shape[-2:]
        features = self.pretrained.get_intermediate_layers(x, self.model_config['layer_idxs'],return_class_token=True)
        
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        rel_depth = self.depth_head(features, patch_h, patch_w)
        # depth = self.depth_head(features, patch_h, patch_w)
        # depth = self.denormalize(depth, min_val, max_val)
        # print(rel_depth.shape)

        
        # print(f'relative depth shape is {rel_depth.shape}')
        # print(f'sparse feature shape is {sparse_feature.shape}')
        rel_depth_np = rel_depth.cpu().numpy()
       

        sparse_feature_np = prompt_depth.cpu().numpy() 

        batch_size = rel_depth_np.shape[0]
        int_depth_batch = []
        int_scaffolding_batch = []

        for i in range(batch_size):
        #     # Extract individual depth maps and priors
            rel_depth_single = rel_depth_np[i]
            rel_depth_single = np.squeeze(rel_depth_single, axis=0)
           
            sparse_feature_single = np.squeeze(sparse_feature_np[i], axis=0)

            input_sparse_depth_valid = (sparse_feature_single < self.max_pred) * (sparse_feature_single > self.min_pred)
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
            GlobalAlignment.clamp_min_max(clamp_min=self.min_pred, clamp_max=self.max_pred)
            
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
      

        int_depth = torch.from_numpy(int_depth_batch).float().to(prompt_depth.device)
        int_scaffolding = torch.from_numpy(int_scaffolding_batch).float().to(prompt_depth.device) 
      
        int_depth = int_depth.unsqueeze(1)
        int_scaffolding = int_scaffolding.unsqueeze(1)

        # print(f'ga shape is {int_depth.shape}')
        # print(f'scale residual shape is {int_scaffolding.shape}')
        # show_images_three_sources(rel_depth.unsqueeze(1), int_depth, int_scaffolding)
        
        prior_map = torch.cat([int_depth,int_scaffolding ], dim = 1)
        pred, scales = self.ScaleMapLearner(prior_map, int_depth)

        # show_images_two_sources(rel_depth, pred)
        output = dict(metric_depth=1.0/pred)
        output['inverse_depth'] = pred
        # show_images(1.0/pred)
        # if return_final_centers or return_probs:
        #     output['bin_centers'] = b_centers

        # if return_probs:
        #     output['probs'] = probabilities

        return output
        

    @torch.no_grad()
    def predict(self,
                image: torch.Tensor,
                prompt_depth: torch.Tensor):
        return self.forward(image, prompt_depth)

    def normalize(self,
                  prompt_depth: torch.Tensor):
        B, C, H, W = prompt_depth.shape
        min_val = torch.quantile(
            prompt_depth.reshape(B, -1), 0., dim=1, keepdim=True)[:, :, None, None]
        max_val = torch.quantile(
            prompt_depth.reshape(B, -1), 1., dim=1, keepdim=True)[:, :, None, None]
        prompt_depth = (prompt_depth - min_val) / (max_val - min_val)
        return prompt_depth, min_val, max_val

    def denormalize(self,
                    depth: torch.Tensor,
                    min_val: torch.Tensor,
                    max_val: torch.Tensor):
        return depth * (max_val - min_val) + min_val


    # def get_lr_params(self, lr):
    #     """
    #     Learning rate configuration for different layers of the model
    #     Args:
    #         lr (float) : Base learning rate
    #     Returns:
    #         list : list of parameters to optimize and their learning rates, in the format required by torch optimizers.
    #     """
    #     param_conf = []
    #     if self.train_dino:

    #         dino_params = self.pretrained.parameters()
    #         dino_lr_factor = self.dino_lr_factor
    #         param_conf.append(
    #             {'params': dino_params, 'lr': lr / dino_lr_factor})

    #     remaining_modules = []
    #     for name, child in self.named_children():
    #         if name != 'pretrained':
    #             remaining_modules.append(child)
    #     remaining_params = itertools.chain(
    #         *[child.parameters() for child in remaining_modules])

    #     param_conf.append({'params': remaining_params, 'lr': lr})

    #     return param_conf

    # def get_lr_params(self, lr):
    #     """
    #     Learning rate configuration for different layers of the model.
    #     Args:
    #         lr (float): Base learning rate.
    #     Returns:
    #         list: List of parameter groups with their learning rates and names.
    #     """
    #     param_conf = []
        
    #     if self.train_dino:
    #         dino_params = self.pretrained.parameters()
    #         dino_lr_factor = self.dino_lr_factor
    #         param_conf.append({
    #             'params': dino_params,
    #             'lr': lr / dino_lr_factor,
    #             'name': 'pretrained'
    #         })
        
    #     # Instead of chaining all remaining modules together, add each module with its name.
    #     for name, child in self.named_children():
    #         if name != 'pretrained':
    #             param_conf.append({
    #                 'params': child.parameters(),
    #                 'lr': lr,
    #                 'name': name
    #             })
        
    #     return param_conf
    
    def get_lr_params(self, lr):
        param_conf = []
        
        # If using a pretrained branch with a different LR
        if self.train_dino:
            dino_params = self.pretrained.parameters()
            dino_lr_factor = self.dino_lr_factor
            param_conf.append({
                'params': dino_params,
                'lr': lr / dino_lr_factor,
                'name': 'pretrained'
            })

        if self.train_DPTHead:
            dino_params = self.depth_head.parameters()
            dino_lr_factor = self.dino_lr_factor
            param_conf.append({
                'params': dino_params,
                'lr': lr / dino_lr_factor,
                'name': 'DPTHead'
            })
        
        SML_params = self.ScaleMapLearner.parameters()
        param_conf.append({
            'params': SML_params,
            'lr': lr,
            'name': 'SML'
        })
        
        return param_conf

    @staticmethod
    def build(da_pretrained_recource, pretrained_resource = None, **kwargs):
        
        model = DA_SML(da_pretrained_recource,  **kwargs)
        if pretrained_resource:
            assert isinstance(pretrained_resource, str), "pretrained_resource must be a string"
            model = load_state_from_resource(model, pretrained_resource)
        return model

    @staticmethod
    def build_from_config(config):
        return DA_SML.build(**config)
