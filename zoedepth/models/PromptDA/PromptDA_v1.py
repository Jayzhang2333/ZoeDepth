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

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], 'layer_idxs': [2, 5, 8, 11]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768], 'layer_idxs': [2, 5, 8, 11]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'layer_idxs': [4, 11, 17, 23]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536], 'layer_idxs': [9, 19, 29, 39]}
}


class PromptDA(DepthModel):
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
        self.depth_head = DPTHead(in_channels=dim,
                                  features=model_config['features'],
                                  out_channels=model_config['out_channels'],
                                  use_bn=self.use_bn,
                                  use_clstoken=self.use_clstoken,
                                  output_act=self.output_act)

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
        prompt_depth_valid = (prompt_depth < self.min_pred_inv) & (prompt_depth > self.max_pred_inv)
        prompt_depth[~prompt_depth_valid] = 0


        h, w = x.shape[-2:]
        features = self.pretrained.get_intermediate_layers(x, self.model_config['layer_idxs'],return_class_token=True)
        
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        depth = self.depth_head(features, patch_h, patch_w, prompt_depth)
        # depth = self.depth_head(features, patch_h, patch_w)
        # depth = self.denormalize(depth, min_val, max_val)
        
        if self.min_pred is not None:
            min_pred_inv = 1.0 / self.min_pred
            depth = torch.where(depth > min_pred_inv, min_pred_inv, depth)

        if self.max_pred is not None:
            max_pred_inv = 1.0 / self.max_pred
            depth = torch.where(depth < max_pred_inv, max_pred_inv, depth)

        depth = 1.0 / depth
        return depth

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

    def get_lr_params(self, lr):
        """
        Learning rate configuration for different layers of the model.
        Args:
            lr (float): Base learning rate.
        Returns:
            list: List of parameter groups with their learning rates and names.
        """
        param_conf = []
        
        if self.train_dino:
            dino_params = self.pretrained.parameters()
            dino_lr_factor = self.dino_lr_factor
            param_conf.append({
                'params': dino_params,
                'lr': lr / dino_lr_factor,
                'name': 'pretrained'
            })
        
        # Instead of chaining all remaining modules together, add each module with its name.
        for name, child in self.named_children():
            if name != 'pretrained':
                param_conf.append({
                    'params': child.parameters(),
                    'lr': lr,
                    'name': name
                })
        
        return param_conf
    
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
        
        prompt_depth_params = []
        other_params = []
        for name, param in self.named_parameters():
            # Check if the parameter belongs to the "scratch.refinenet1" subtree
            if 'sparse_embedding_prompt' in name or 'atten_conv_prompt' in name or 'projection_prompt' in name:
                prompt_depth_params.append(param)
            else:
                other_params.append(param)
        
        if other_params:
            param_conf.append({
                'params': other_params,
                'lr': lr,
                'name': 'other'
            })
        if prompt_depth_params:
            param_conf.append({
                'params': prompt_depth_params,
                'lr': lr * 10,  # 10x higher learning rate
                'name': 'prompt modules'
            })
        
        return param_conf

    @staticmethod
    def build(da_pretrained_recource, pretrained_resource = None, **kwargs):
        
        model = PromptDA(da_pretrained_recource,  **kwargs)
        if pretrained_resource:
            assert isinstance(pretrained_resource, str), "pretrained_resource must be a string"
            model = load_state_from_resource(model, pretrained_resource)
        return model

    @staticmethod
    def build_from_config(config):
        return PromptDA.build(**config)
