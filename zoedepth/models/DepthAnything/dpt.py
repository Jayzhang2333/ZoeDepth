# Copyright (c) 2024, Depth Anything V2
# https://github.com/DepthAnything/Depth-Anything-V2/blob/main/depth_anything_v2/dpt.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import _make_scratch, _make_fusion_block, _make_fusion_block_with_depth, _make_fusion_block_scale_residual_block, ScaleResidualBlock, ResidualConvUnit
from zoedepth.models.layers.fusion_layers import BasicBlock, mViT_assemble, conv_bn_relu
from zoedepth.models.layers.global_alignment import LeastSquaresEstimatorTorch, AnchorInterpolator2D
import numpy as np
import matplotlib.pyplot as plt
# def adapt_pool(m, dep):
#     dep = F.avg_pool2d(dep, 3, 2, 1)
#     m = F.avg_pool2d(m.float(), 3, 2, 1)
#     dep = (m > 0) * dep / (m + 1e-8)

#     return dep

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

# def get_sparse_pool(dep, num):
#     masks, depths = [], []
#     m = dep > 0
#     masks.append(m)
#     depths.append(dep)
#     for _ in range(num):
#         dep = adapt_pool(m, dep)
#         m = dep > 0
#         masks.append(m)
#         depths.append(dep)
    
#     for i in range(num+1):
#         depths[i][~masks[i]] = 1

#     return masks, depths

class DPTHead(nn.Module):
    def __init__(self,
                 in_channels,
                 features=256,
                 out_channels=[256, 512, 1024, 1024],
                 use_bn=False,
                 use_clstoken=False,
                 output_act='sigmoid',
                 with_depth = False):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        if with_depth:
            self.scratch.refinenet1 = _make_fusion_block_with_depth(
                features, use_bn, patch_size = 8, size = [96,128])
            self.scratch.refinenet2 = _make_fusion_block_with_depth(
                features, use_bn, patch_size = 4, size = [48,64])
            self.scratch.refinenet3 = _make_fusion_block_with_depth(
                features, use_bn, patch_size = 2, size = [24,32])
            self.scratch.refinenet4 = _make_fusion_block_with_depth(
                features, use_bn, patch_size = 1, size = [12,16])
        else:
            self.scratch.refinenet1 = _make_fusion_block(
                features, use_bn)
            self.scratch.refinenet2 = _make_fusion_block(
                features, use_bn)
            self.scratch.refinenet3 = _make_fusion_block(
                features, use_bn)
            self.scratch.refinenet4 = _make_fusion_block(
                features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        act_func = nn.Sigmoid() if output_act == 'sigmoid' else nn.Identity()

      
        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)

        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2,
                        kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1,
                        stride=1, padding=0),
            nn.ReLU(True),
            act_func,
        )

    def forward(self, out_features, patch_h, patch_w, prompt_depth=None):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape(
                (x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

       
        path_4 = self.scratch.refinenet4(
            layer_4_rn, size=layer_3_rn.shape[2:], prompt_depth=prompt_depth)
        
        path_3 = self.scratch.refinenet3(
            path_4, layer_3_rn, size=layer_2_rn.shape[2:], prompt_depth=prompt_depth)
        
        path_2 = self.scratch.refinenet2(
            path_3, layer_2_rn, size=layer_1_rn.shape[2:], prompt_depth=prompt_depth)
        
        path_1 = self.scratch.refinenet1(
            path_2, layer_1_rn, prompt_depth=prompt_depth)
        
        out = self.scratch.output_conv1(path_1)
        out_feat = F.interpolate(
            out, (int(patch_h * 14), int(patch_w * 14)),
            mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out_feat)
        return out
    

class DPTHead_customised(nn.Module):
    def __init__(self,
                 in_channels,
                 features=256,
                 out_channels=[256, 512, 1024, 1024],
                 use_bn=False,
                 use_clstoken=False,
                 output_act='sigmoid',
                 with_depth = False):
        super(DPTHead_customised, self).__init__()

        self.max_depth = 18.0
        self.min_depth = 0.1

        self.use_clstoken = use_clstoken

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        if with_depth:
            self.scratch.refinenet1 = _make_fusion_block_with_depth(
                features, use_bn, patch_size = 8, input_size = [96,128], base_layer = False)
            self.scratch.refinenet2 = _make_fusion_block_with_depth(
                features, use_bn, patch_size = 4, input_size = [48,64], base_layer = False)
            self.scratch.refinenet3 = _make_fusion_block_with_depth(
                features, use_bn, patch_size = 2, input_size = [24,32], base_layer = False)
            self.scratch.refinenet4 = _make_fusion_block_with_depth(
                features, use_bn, patch_size = 1, input_size = [12,16], base_layer = True)
        else:
            self.scratch.refinenet1 = _make_fusion_block(
                features, use_bn)
            self.scratch.refinenet2 = _make_fusion_block(
                features, use_bn)
            self.scratch.refinenet3 = _make_fusion_block(
                features, use_bn)
            self.scratch.refinenet4 = _make_fusion_block(
                features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        act_func = nn.Sigmoid() if output_act == 'sigmoid' else nn.Identity()

      
        # the provides output featrue map
        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)

        # this is the output relative depth
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2,
                        kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1,
                        stride=1, padding=0),
            nn.ReLU(True),
            act_func,
        )


        self.local_dense_branch_prompt = nn.Sequential(
                BasicBlock( (head_features_1 // 2 )+4, (head_features_1 // 2 )+4, ratio=4)
        )

        self.global_dense_branch_prompt = nn.Sequential(
                mViT_assemble((head_features_1 // 2 )+4, patch_size=32, embedding_dim=128, out_channels = (head_features_1 // 2 )+4, size = [336,448], num_layer = 2),
        )

        self.fuse_conv1_prompt = nn.Sequential(
            nn.Conv2d((((head_features_1 // 2 )+4) *2), (head_features_1 // 2 )+4, 3, 1, 1),
            nn.ReLU(),
        )
        self.fuse_conv2_prompt = nn.Sequential(
            nn.Conv2d((((head_features_1 // 2 )+4) *2), (head_features_1 // 2 )+4, 3, 1, 1),
            nn.Sigmoid(),
        )

        self.dense_output_prompt = nn.Sequential(
            nn.Conv2d((head_features_1 // 2 )+4, ((head_features_1 // 2 )+4) //2,
                            kernel_size=3, stride=1, padding=1),
            nn.ReLU(False),
            nn.Conv2d( ((head_features_1 // 2 )+4) //2 , 1, kernel_size=1,
                        stride=1, padding=0),
            nn.Identity(),
        )


        self.dense_output_confidence_prompt = nn.Sequential(
            nn.Conv2d((head_features_1 // 2 )+4, ((head_features_1 // 2 )+4) //2,
                            kernel_size=3, stride=1, padding=1),
            nn.ReLU(False),
            nn.Conv2d(((head_features_1 // 2 )+4) //2 , 1, kernel_size=1,
                        stride=1, padding=0),
            nn.Sigmoid(),
        )
        
        



    def forward(self, out_features, patch_h, patch_w, prompt_depth=None):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape(
                (x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        

        path_4, depth_4, conf_4 = self.scratch.refinenet4(
            layer_4_rn, size=layer_3_rn.shape[2:], prompt_depth=prompt_depth)
       
        path_3, depth_3, conf_3 = self.scratch.refinenet3(
            path_4, layer_3_rn, size=layer_2_rn.shape[2:], prompt_depth=prompt_depth, previous_depth = depth_4, previous_conf = conf_4)
        
        path_2, depth_2, conf_2 = self.scratch.refinenet2(
            path_3, layer_2_rn, size=layer_1_rn.shape[2:], prompt_depth=prompt_depth, previous_depth = depth_3, previous_conf = conf_3)
        
        path_1, depth_1, conf_1 = self.scratch.refinenet1(
            path_2, layer_1_rn, prompt_depth=prompt_depth, previous_depth = depth_2, previous_conf = conf_2)
        
        
        out = self.scratch.output_conv1(path_1)
        out_feat = F.interpolate(
            out, (int(patch_h * 14), int(patch_w * 14)),
            mode="bilinear", align_corners=True)
        
        depth_1 = F.interpolate(
            depth_1, (int(patch_h * 14), int(patch_w * 14)),
            mode="bilinear", align_corners=True)
        
        conf_1 = F.interpolate(
            conf_1, (int(patch_h * 14), int(patch_w * 14)),
            mode="bilinear", align_corners=True)
        

        rel_depth = self.scratch.output_conv2(out_feat)

        max_depth_inv = 1.0/ self.max_depth
        min_deprh_inv = 1.0/ self.min_depth

        # sparse_depth should already be inverse depth at this point
        sparse_mask = (prompt_depth < min_deprh_inv) * (prompt_depth > max_depth_inv)
        estimator = LeastSquaresEstimatorTorch(rel_depth, prompt_depth, sparse_mask)
        estimator.compute_scale_and_shift()  # find best scale & shift
        estimator.apply_scale_and_shift()    # apply them
        # it will do the inverse inside the function
        estimator.clamp_min_max(clamp_min=self.min_depth, clamp_max=self.max_depth)
        ga_result = estimator.output

        prompt_depth = torch.where(prompt_depth == 0, depth_1, prompt_depth)
        scale_residual = torch.where(prompt_depth>0, prompt_depth / ga_result, torch.ones_like(prompt_depth))
        sparse_mask = torch.where(sparse_mask == 0, conf_1, sparse_mask)
        input_features = torch.cat([out_feat, scale_residual, sparse_mask, ga_result, rel_depth], dim=1)
        # print(input_features.shape)
        local_features = self.local_dense_branch_prompt(input_features)
        global_features = self.global_dense_branch_prompt(input_features)

        # print(local_features.shape)
        # print(global_features.shape)
        combined_features = torch.cat([local_features, global_features], dim=1)

        combined_features = self.fuse_conv1_prompt(combined_features) * self.fuse_conv2_prompt(combined_features)

        dense_scale_residual = self.dense_output_prompt(combined_features)
        dense_scale_residual_conf = self.dense_output_confidence_prompt(combined_features)
        
        
        # show_images(local_filled)
        scales = F.relu(1.0 + dense_scale_residual)
        pred = scales*ga_result

       
        if self.min_depth is not None:
            min_pred_inv = 1.0/self.min_depth
            pred[pred > min_pred_inv] = min_pred_inv
            # intermedian_pred[intermedian_pred > min_pred_inv] = min_pred_inv
        if self.max_depth is not None:
            max_pred_inv = 1.0/self.max_depth
            # max_pred_inv = 0
            pred[pred < max_pred_inv] = max_pred_inv
            # intermedian_pred[intermedian_pred < max_pred_inv] = max_pred_inv

        return pred, dense_scale_residual_conf
    



class DPTHeadCustomised2(nn.Module):
    def __init__(self,
                 in_channels,
                 features=256,
                 out_channels=[256, 512, 1024, 1024],
                 use_bn=False,
                 use_clstoken=False,
                 output_act='sigmoid'):
        super(DPTHeadCustomised2, self).__init__()

        self.max_depth = 18.0
        self.min_depth = 0.1

        self.use_clstoken = use_clstoken

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        head_features_1 = features
        head_features_2 = 32

        self.scratch.refinenet1 = _make_fusion_block_scale_residual_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block_scale_residual_block(features,  use_bn)
        self.scratch.refinenet3 = _make_fusion_block_scale_residual_block(features,  use_bn)
        self.scratch.refinenet4 = _make_fusion_block_scale_residual_block(features,  use_bn)

        act_func = nn.Sigmoid() if output_act == 'sigmoid' else nn.Identity()

      
        # the provides output featrue map
        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)

        # this is the output relative depth
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2,
                        kernel_size=3, stride=1, padding=1),
            nn.ReLU(False),
            nn.Conv2d(head_features_2, 1, kernel_size=1,
                        stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )

        # self.scale_residual_embedding = conv_bn_relu(2, 8, kernel=3, stride=1, padding=1,
        #                                   bn=False)
        
        # self.ga_embedding_scale = conv_bn_relu(1, 8, kernel=3, stride=1, padding=1,
        #                                   bn=False)

        # self.local_scale_branch = nn.Sequential(
        #         BasicBlock((head_features_1 // 2 )+16, (head_features_1 // 2 )+16, ratio=4)
        # )

        # self.global_scale_branch = nn.Sequential(
        #         mViT_assemble((head_features_1 // 2 )+16, patch_size=32, embedding_dim=128, out_channels = (head_features_1 // 2 )+16, size = [336,448], num_heads=8, num_layer = 2),
        # )

        # self.scale_fuse_conv1 = nn.Sequential(
        #     nn.Conv2d((((head_features_1 // 2 )+16) *2), ((head_features_1 // 2 )+16), 3, 1, 1),
        #     nn.ReLU(),
        # )
        # self.scale_fuse_conv2 = nn.Sequential(
        #     nn.Conv2d((((head_features_1 // 2 )+16) *2), ((head_features_1 // 2 )+16), 3, 1, 1),
        #     nn.Sigmoid(),
        # )

        # self.scale_decode_conv = ResidualConvUnit(((head_features_1 // 2 )+16), nn.ReLU(False), use_bn)
        
        # self.dense_scale = nn.Sequential(
        #     nn.Conv2d((head_features_1 // 2 )+16, ((head_features_1 // 2 )+16) //2,
        #                     kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(False),
        #     nn.Conv2d( ((head_features_1 // 2 )+16) //2 , 1, kernel_size=1,
        #                 stride=1, padding=0),
        #     nn.Identity(),
        # )

        # self.scale_conf = nn.Sequential(
        #     nn.Conv2d((head_features_1 // 2 )+16, ((head_features_1 // 2 )+16) //2,
        #                     kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(False),
        #     nn.Conv2d( ((head_features_1 // 2 )+16) //2 , 1, kernel_size=1,
        #                 stride=1, padding=0),
        #     nn.Sigmoid(),
        # )

        self.skip_add = nn.quantized.FloatFunctional()


        
        
        



    def forward(self, out_features, patch_h, patch_w, prompt_depth=None):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape(
                (x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out_feat = F.interpolate(
            out, (int(patch_h * 14), int(patch_w * 14)),
            mode="bilinear", align_corners=True)

        rel_depth = self.scratch.output_conv2(out_feat)

        decoder_featrues = [path_4, path_3, path_2, path_1, out_feat]
        return rel_depth, decoder_featrues

        # max_depth_inv = 1.0/ self.max_depth
        # min_deprh_inv = 1.0/ self.min_depth

        # sparse_depth should already be inverse depth at this point
        # sparse_mask = (prompt_depth < min_deprh_inv) * (prompt_depth > max_depth_inv)
        # estimator = LeastSquaresEstimatorTorch(rel_depth, prompt_depth, sparse_mask)
        # estimator.compute_scale()  # find best scale & shift
        # estimator.apply_scale()    # apply them
        
        # print(f'scale is {estimator.scale}')
        # print(f'scale is negative {estimator.scale<0}')
        # print(f'shift is {estimator.shift}')
        # print(f'shift is negative {estimator.shift<0}')
        # show_images_two_sources(estimator.output, 1.0/gt)
        # show_images_three_sources(rel_depth, prompt_depth, estimator.output)

        # estimator.clamp_min_max(clamp_min=self.min_depth)
        # ga_result = estimator.output
        # d = ga_result.clone()

        # scale_residual = torch.where(prompt_depth>0, prompt_depth / (ga_result + 1e-6), torch.ones_like(prompt_depth))

        # scale_residual_np = scale_residual.detach().cpu().numpy() 
        # sparse_mask_np = sparse_mask.cpu().numpy() 
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
       
        # residual_embedding = self.scale_residual_embedding(torch.cat([scale_residual, sparse_mask], dim=1))
        # ga_embedding = self.ga_embedding_scale(ga_result)
        # input_features = torch.cat([out_feat, residual_embedding, ga_embedding], dim=1)


        # local_features = self.local_scale_branch(input_features)
        # global_features = self.global_scale_branch(input_features)

        # combined_features = torch.cat([local_features, global_features], dim=1)

        # combined_features = self.scale_fuse_conv1(combined_features) * self.scale_fuse_conv2(combined_features)

        # combined_features = self.skip_add.add(embedding_1, combined_features)

        # scale_output_feature = self.scale_decode_conv(combined_features)

        # scale_output = self.dense_scale(scale_output_feature)

        # scale_conf = self.scale_conf(scale_output_feature)

        # scale_map = F.relu(1.0 + scale_output)

        # scale_map =  torch.mul(scale_map, scale_map_1)
        
        # pred = d * scale_map

        # show_images_four_sources(rel_depth, d, scale_map, pred)
        # show_images_two_sources(ga_result, scale_map)

        

        
        # if self.min_depth is not None:
        #     min_pred_inv = 1.0/self.min_depth
        #     pred = torch.where(pred > min_pred_inv, min_pred_inv, pred)
        #     # intermedian_pred[intermedian_pred > min_pred_inv] = min_pred_inv
        # if self.max_depth is not None:
        #     max_pred_inv = 1.0/self.max_depth
        #     pred = torch.where(pred < max_pred_inv, max_pred_inv, pred)
        #     # intermedian_pred[intermedian_pred < max_pred_inv] = max_pred_inv
        # scale_show = torch.where(scale_residual == 1, 0, scale_residual)
        # show_images_two_sources(scale_show, scale_map)
        # show_images_four_sources(scale_residual, scale_map, 1.0/d, 1.0/pred)

        # return pred, scale_conf




class DPTHeadCustomised3(nn.Module):
    def __init__(self,
                 in_channels,
                 features=256,
                 out_channels=[256, 512, 1024, 1024],
                 use_bn=False,
                 use_clstoken=False,
                 output_act='sigmoid'):
        super(DPTHeadCustomised3, self).__init__()

        self.use_clstoken = use_clstoken

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        head_features_1 = features
        head_features_2 = 32

        self.scratch.refinenet1 = _make_fusion_block_scale_residual_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block_scale_residual_block(features,  use_bn)
        self.scratch.refinenet3 = _make_fusion_block_scale_residual_block(features,  use_bn)
        self.scratch.refinenet4 = _make_fusion_block_scale_residual_block(features,  use_bn)
      
        # the provides output featrue map
        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)

        # this is the output relative depth
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2,
                        kernel_size=3, stride=1, padding=1),
            nn.ReLU(False),
            nn.Conv2d(head_features_2, 1, kernel_size=1,
                        stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, out_features, patch_h, patch_w, prompt_depth=None):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape(
                (x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out
        # featrues = [layer_1, layer_2, layer_3, layer_4]
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out_feat = F.interpolate(
            out, (int(patch_h * 14), int(patch_w * 14)),
            mode="bilinear", align_corners=True)

        rel_depth = self.scratch.output_conv2(out_feat)

        featrues = [path_1, path_2, path_3, path_4]
       
        return rel_depth, featrues
