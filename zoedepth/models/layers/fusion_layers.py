import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torchvision
import math
# from mmseg.utils import get_root_logger
model_path = {
    'resnet18': 'resnet18.pth',
    'resnet34': 'resnet34.pth'
}



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


# def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True, relu=True):
#     assert (kernel % 2) == 1, "only odd kernel is supported but kernel = {}".format(
#         kernel
#     )

#     layers = []
#     layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding, bias=not bn))
#     if bn:
#         layers.append(nn.BatchNorm2d(ch_out))
#     if relu:
#         layers.append(nn.LeakyReLU(0.2, inplace=True))

#     layers = nn.Sequential(*layers)

#     return layers

class GateConv(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(FillConv, self).__init__()
        
        self.fuse_conv1 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fuse_conv2 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 3, 1, 1),
            nn.Sigmoid(),
        )
      

        self.apply(self._init_weights)

    def _init_weights(self, m):
        for m in self.modules():
            # check if it is a cov2D layer
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                # check is it is using bias
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.3)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0.1)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fuse_conv1(x) * self.fuse_conv2(x)
        
        return x

# class FillConv(nn.Module):
#     def __init__(self, branch_1_channel, branch_2_channel):
#         super(FillConv, self).__init__()
#         # self.channel = channel
#         # self.conv_branch1 = conv_bn_relu(branch_1_channel, 48, 3, 1, 1, bn=False)
#         # self.conv_branch2 = conv_bn_relu(branch_2_channel, 16, 3, 1, 1, bn=False)

#         # self._trans = nn.Conv2d(48, 16, 1, 1, 0)
#         self.fuse_conv1 = nn.Sequential(
#             nn.Conv2d(32, 16, 3, 1, 1),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#         self.fuse_conv2 = nn.Sequential(
#             nn.Conv2d(32, 16, 3, 1, 1),
#             nn.Sigmoid(),
#         )
#         # self.fuse_conv3 = nn.Sequential(
#         #     nn.Conv2d(64, 64, 3, 1, 1),
#         #     nn.LeakyReLU(0.2, inplace=True),
#         # )
#         # self.fuse_conv4 = nn.Sequential(
#         #     nn.Conv2d(64, 64, 3, 1, 1),
#         #     nn.Sigmoid(),
#         # )

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         for m in self.modules():
#             # check if it is a cov2D layer
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.xavier_normal_(m.weight.data)
#                 # check is it is using bias
#                 if m.bias is not None:
#                     torch.nn.init.constant_(m.bias.data, 0.3)
#             elif isinstance(m, nn.Linear):
#                 torch.nn.init.normal_(m.weight.data, 0.1)
#                 if m.bias is not None:
#                     torch.nn.init.zeros_(m.bias.data)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def forward(self, branch1_input, branch2_input):
#         f_branch1 = self.conv_branch1(branch1_input)
#         f_branch2 = self.conv_branch2(branch2_input)
#         f_branch2 = torch.cat([self._trans(f_branch1), f_branch2], dim=1)
#         f_branch2 = self.fuse_conv1(f_branch2) * self.fuse_conv2(f_branch2)
#         f = torch.cat([f_branch1, f_branch2], dim=1)
#         f = f + self.fuse_conv3(f) * self.fuse_conv4(f)

#         return f

class FillConv(nn.Module):
    def __init__(self, channel):
        super(FillConv, self).__init__()
        self.channel = channel
        self.conv_rgb = conv_bn_relu(1, 4, 3, 1, 1, bn=False)
        self.conv_dep = conv_bn_relu(2, 4, 3, 1, 1, bn=False)

        # self._trans = nn.Conv2d(33, 16, 1, 1, 0)
        self.fuse_conv1 = nn.Sequential(
            nn.Conv2d(8, 4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fuse_conv2 = nn.Sequential(
            nn.Conv2d(8, 4, 3, 1, 1),
            nn.Sigmoid(),
        )
        # self.fuse_conv3 = nn.Sequential(
        #     nn.Conv2d(8, 8, 3, 1, 1),
        #     nn.LeakyReLU(0.2, inplace=True),
        # )
        # self.fuse_conv4 = nn.Sequential(
        #     nn.Conv2d(8, 8, 3, 1, 1),
        #     nn.Sigmoid(),
        # )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        for m in self.modules():
            # 判断是否属于Conv2d
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                # 判断是否有偏置
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.3)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0.1)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, rgb, dep):
        f_rgb = self.conv_rgb(rgb)
        # f_rgb = rgb
        f_dep = self.conv_dep(dep)
        # f_dep = torch.cat([self._trans(f_rgb), f_dep], dim=1)
        f_dep = torch.cat([f_rgb, f_dep], dim=1)
        f_dep = self.fuse_conv1(f_dep) * self.fuse_conv2(f_dep)
        # f = torch.cat([f_rgb, f_dep], dim=1)
        # f = f + self.fuse_conv3(f) * self.fuse_conv4(f)

        return f_dep
    


class EstimateAndPlaceModule(nn.Module):

    def __init__(self, in_channels):
        """
        Args:
            in_channels (int): Number of input feature channels.
                              E.g. if you cat backbone feats + prev_depth + prev_conf, 
                              set this accordingly.
            intermediate_channels (int): #channels in the stem block before heads.
        """
        super().__init__()

        self.scale_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1, groups=1),
            nn.Conv2d(in_channels//2, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace = False),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Identity(),
        )

        self.conf_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1, groups=1),
            nn.Conv2d(in_channels//2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(False),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        for m in self.modules():
            # 判断是否属于Conv2d
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                # 判断是否有偏置
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.3)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0.1)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, 
                x, 
                sparse_prior):

        # p stands for prediction, it is prediction of depth
        
        scale = self.scale_head(x)
        scale = F.relu(1.0 + scale)
        # c stands for confidence
        
        

        B, _, H, W = scale.shape
        valid_mask = (sparse_prior > 0).float()  # (B,1,H,W)

        scale = scale * (1 - valid_mask) + sparse_prior * valid_mask
        # return scale
        c = self.conf_head(x)
        c = 0.8 * c + 0.1   # make confident between 0.1 to 0.9

        # 5) Confidence = 1 for valid pixels
        cs = c * (1 - valid_mask) + valid_mask

        # show_images_four_sources(scale, cs, sparse_prior, valid_mask)
        
        output = torch.cat([scale, cs], dim=1)

        return output

        
    

class ScaleAndPlaceModule(nn.Module):
    """
    Scale & Place (S&P) module:
      - Optionally concatenates features with prev_depth & prev_conf.
      - Predicts an 'up-to-scale' depth and its confidence.
      - Scales the depth via differentiable weighted linear regression w.r.t. sparse depth.
      - Places the known sparse depths (and assigns maximum confidence to them).
    """

    def __init__(self, in_channels, min_depth = 0.1, max_depth = 10):
        """
        Args:
            in_channels (int): Number of input feature channels.
                              E.g. if you cat backbone feats + prev_depth + prev_conf, 
                              set this accordingly.
            intermediate_channels (int): #channels in the stem block before heads.
        """
        super().__init__()

        # --- Stem block: two conv layers, each with Conv2d → BN → LeakyReLU ---
        # self.conv_input = conv_bn_relu(in_channels, intermediate_channels, kernel=3, stride=1, padding=1)
        self.min_depth_inverse = 1.0/min_depth
        self.max_depth_inverse = 1.0/max_depth
        self.depth_head = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
        )

        self.conf_head = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self._init_weights()

        # self.depth_dead = conv_bn_relu(in_channels, 1, kernel=3, stride=1, padding=1)
        # self.conf_head = conv_bn_relu(in_channels, 1, kernel=3, stride=1, padding=1)

        
        # --- Heads: 1) depth head, 2) confidence head ---
        # Depth head outputs one channel (predicted depth).
        # self.depth_head = nn.Conv2d(branch_channels, 1, kernel_size=3, stride=1, padding=1)
        
        # Confidence head outputs one channel in (0.1..0.9). 
        # We will post-process the raw prediction with a sigmoid-based transformation.
        # self.conf_head = nn.Conv2d(branch_channels, 1, kernel_size=3, stride=1, padding=1)
    def _init_weights(self):
        """
        Initialize Conv2d layers based on the activation that follows them:
          - If followed by ReLU or LeakyReLU => Kaiming
          - If followed by Sigmoid => Xavier
          - Otherwise => default to Kaiming
        """
        # You can define a helper if you prefer to avoid duplication
        def init_layer_pair(seq):
            """
            Given a sequential container (seq),
            look at (module, next_module) pairs and init accordingly.
            """
            for i, m in enumerate(seq):
                if isinstance(m, nn.Conv2d):
                    # Check the next module to decide
                    next_m = seq[i+1] if (i + 1) < len(seq) else None

                    # If next is ReLU/LeakyReLU => Kaiming
                    if isinstance(next_m, nn.ReLU) or isinstance(next_m, nn.LeakyReLU):
                        nn.init.kaiming_normal_(
                            m.weight,
                            mode='fan_in',
                            nonlinearity='relu'  # or 'leaky_relu' if next_m is LeakyReLU with known slope
                        )
                    
                    # If next is Sigmoid => Xavier
                    elif isinstance(next_m, nn.Sigmoid):
                        nn.init.xavier_normal_(m.weight)
                    
                    # Fallback
                    else:
                        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

        # Apply init to both heads
        init_layer_pair(self.depth_head)
        init_layer_pair(self.conf_head)

    def forward(self, 
                x, 
                sparse_depth=None,
                input_depth = None,
                output_depth = False):
        """
        Forward pass of the S&P module.
        
        Args:
            feats (Tensor): Backbone features, shape (B, C, H, W).
            prev_depth (Tensor, optional): Previous scale's dense depth map, (B, 1, H, W).
            prev_conf  (Tensor, optional): Previous scale's confidence map, (B, 1, H, W).
            sparse_depth (Tensor, optional): Sparse depth map, (B, 1, H, W), 
                                             zero where no measurement is available.
        
        Returns:
            (scaled_depth, updated_conf) each with shape (B, 1, H, W).
        """

        

        # p stands for prediction, it is prediction of depth
        if input_depth is None:
            p = self.depth_head(x)
        else:
            p = input_depth

        # c stands for confidence
        c = self.conf_head(x)
        c = 0.8 * c + 0.1   # make confident between 0.1 to 0.9
        
        
        # this is an implementation looping each data in a batch
        ##########################################################################################3
        # B, _, H, W = p.shape
        # Ds_list = []
        # Cs_list = []
        # resid_list   = [] # scale residual list
        
        # if sparse_depth is not None:
        #     for b in range(B):
        #         # Extract single image from batch
        #         p_b = p[b, 0]          # (H, W)
        #         c_b = c[b, 0]          # (H, W)
        #         s_b = sparse_depth[b, 0]  # (H, W)

        #         valid_mask = (s_b > 0).float()  # 1 where we have valid sparse depth
                
        #         # Weighted sums
        #         sum_c = (c_b * valid_mask).sum()   # denominator for p̄, s̄
        #         if sum_c > 0:
        #             p_bar = (c_b * p_b * valid_mask).sum() / sum_c
        #             s_bar = (c_b * s_b * valid_mask).sum() / sum_c

        #             numerator = (c_b * (p_b - p_bar) * (s_b - s_bar) * valid_mask).sum()
        #             denominator = (c_b * (p_b - p_bar)**2 * valid_mask).sum()

        #             # Protect from zero denominator
        #             if denominator.abs() < 1e-9:
        #                 beta = 1.0
        #                 # assert False, "Weighted variance denominator is < 1e-9. Cannot safely compute beta."
        #             else:
        #                 beta = numerator / denominator
                        
        #             alpha = s_bar - beta * p_bar

        #             # Scale the entire predicted depth
        #             ds_b = alpha + beta * p_b
                    
        #             # Place step: replace known depths + set confidence=1 there
        #             ds_b = ds_b * (1 - valid_mask) + s_b * valid_mask
        #             cs_b = c_b * (1 - valid_mask) + valid_mask  # 1.0 for replaced points
        #         else:
        #             # No valid point => just pass predictions through
        #             ds_b = p_b
        #             cs_b = c_b

        #         Ds_list.append(ds_b.unsqueeze(0))  # shape (1, H, W)
        #         Cs_list.append(cs_b.unsqueeze(0))  # shape (1, H, W)

        #         with torch.no_grad():
        #             # Avoid division by zero
        #             ds_b_clamped = torch.clamp(ds_b, min=1e-9)
        #             ratio_b = torch.ones_like(ds_b_clamped)
        #             ratio_b[valid_mask.bool()] = s_b[valid_mask.bool()] / ds_b_clamped[valid_mask.bool()]

        #         resid_list.append(ratio_b.unsqueeze(0))


        #     # Stack back into (B, 1, H, W)
        #     Ds = torch.stack(Ds_list, dim=0)
        #     Cs = torch.stack(Cs_list, dim=0)
        #     scale_residual = torch.stack(resid_list,  dim=0)
        # else:
        #     # If no sparse depth is provided at all, skip the scale/place steps
        #     Ds = p
        #     Cs = c
        #     scale_residual = torch.ones_like(p)
        #########################################################################################################

        B, _, H, W = p.shape

        # 1) valid_mask = 1 where sparse depth is available, guaranteed > 0
        valid_mask = (sparse_depth > 0).float()  # (B,1,H,W)

        # 2) Weighted sums across spatial dims
        sum_c = (c * valid_mask).sum(dim=[2,3], keepdim=True)  # (B,1,1,1), guaranteed > 0

        # Weighted means
        p_bar = ((p * c * valid_mask).sum(dim=[2,3], keepdim=True)) / sum_c
        s_bar = ((sparse_depth * c * valid_mask).sum(dim=[2,3], keepdim=True)) / sum_c

        # Weighted covariance and variance
        p_shifted = p - p_bar
        s_shifted = sparse_depth - s_bar

        numerator   = (c * valid_mask * p_shifted * s_shifted).sum(dim=[2,3], keepdim=True)  # (B,1,1,1)
        denominator = (c * valid_mask * p_shifted * p_shifted).sum(dim=[2,3], keepdim=True)  # (B,1,1,1)

        # Because it's guaranteed at least one valid point, we expect denominator != 0
        # Nonetheless, we clamp to avoid numeric instability
        denominator_clamped = torch.clamp_min(denominator, 1e-9)

        beta  = numerator / denominator_clamped  # (B,1,1,1)
        alpha = s_bar - beta * p_bar             # (B,1,1,1)
        # if (beta < 0).any():
        #     print("There are negative values in the beta.")
        # else:
        #     pass

        # if (alpha < 0).any():
        #     print("There are negative values in the alpha.")
        # else:
        #     pass

        # 3) Scale p => alpha + beta*p
        #    No fallback needed, since each sample has valid points
        ds = alpha + beta * p  # (B,1,H,W)
        # print(f'alpha: {alpha}, beta: {beta}')
        show_images_four_sources(p,c,ds,sparse_depth )
        # 4) Place step: for valid pixels, overwrite ds with sparse_depth
        ds = ds * (1 - valid_mask) + sparse_depth * valid_mask
        ds_clamped = torch.clamp(ds, min=self.max_depth_inverse, max = self.min_depth_inverse )
        if (ds_clamped < 0).any():
            print("There are negative values in the ds after place depth point.")
        else:
            pass

        # 5) Confidence = 1 for valid pixels
        cs = c * (1 - valid_mask) + valid_mask

        # 6) Scale residual = s / ds for valid, else 1
        # ds_clamped = torch.clamp(ds, min=1e-9)
        ratio = torch.ones_like(ds)
        ratio[valid_mask.bool()] = sparse_depth[valid_mask.bool()] / ds_clamped[valid_mask.bool()]
        
        output = torch.cat([x,p, cs,ratio], dim=1)

        

        if output_depth:
            return output, ds_clamped
        else:
            return output



class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output
    


class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )
        
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn==True:
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
        if self.bn==True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn==True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)
    


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.resblock = BasicBlock(dim, dim, ratio=16)
        self.concat_conv = nn.Conv2d(dim*2, dim, kernel_size=(3, 3), padding=(1, 1), bias=False)


    def forward(self, x, H, W):
        input = x

        # Transformer branch
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # CNN branch
        B, N, C = input.shape
        _, _, Cx = x.shape
        input = input.transpose(1, 2).view(B, C, H, W)
        input = self.resblock(input)

        # fusion
        x = x.transpose(1, 2).view(B, Cx, H, W)
        x = self.concat_conv(torch.cat([x, input], dim=1))
        x = x.flatten(2).transpose(1, 2)


        return x
    

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, ratio=16):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes, ratio=ratio)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=(stride, stride),
                     padding=(1, 1), bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, (1, 1), bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, (1, 1), bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, (kernel_size, kernel_size), padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


def get_resnet34(pretrained=True):
    net = torchvision.models.resnet34(pretrained=False)
    if pretrained:
        state_dict = torch.load(model_path['resnet34'])
        net.load_state_dict(state_dict)

    return net


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=(288,384), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)
    

def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True,
                 relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding,
                            bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    layers = nn.Sequential(*layers)

    return layers

class PyramidVisionTransformer(nn.Module):
    def __init__(self, img_size=(288,384), patch_size=1, in_chans=3, num_classes=1000, embed_dims=[64],
                 num_heads=[1], mlp_ratios=[4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3],
                 sr_ratios=[1], num_stages=1, pretrained=None):
        super().__init__()
        self.depths = depths
        self.num_stages = num_stages

        net = get_resnet34(pretrained=True)
        setattr(self, "embed_layer1", net.layer1)
        setattr(self, "embed_layer2", net.layer2)
        del net
        in_chans = 128

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = PatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                     patch_size= patch_size if i == 0 else 2,
                                     in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                     embed_dim=embed_dims[i])
            num_patches = patch_embed.num_patches if i != num_stages - 1 else patch_embed.num_patches + 1
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=norm_layer, sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)

            trunc_normal_(pos_embed, std=.02)

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x):
        outs = []

        B = x.shape[0]

        x = getattr(self, 'embed_layer1')(x)
        outs.append(x)
        x = getattr(self, 'embed_layer2')(x)
        outs.append(x)


        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, (H, W) = patch_embed(x)
            if i == self.num_stages - 1:
                pos_embed = self._get_pos_embed(pos_embed[:, 1:], patch_embed, H, W)
            else:
                pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)

            x = pos_drop(x + pos_embed)
            for blk in block:
                x = blk(x, H, W)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)

        return x
    


class SelfAttnPropagation(nn.Module):
    """
    Global self-attention propagation on an image feature map.
    
    This module propagates the image feature map using a self-attention mechanism.
    The feature map is first flattened, then linear layers are used to compute the
    query, key, and value. Finally, a scaled dot-product attention is computed to 
    obtain the output, which is then combined with the original input via a residual
    connection.
    
    Args:
        in_channels (int): Number of channels in the input feature map.
    """
    def __init__(self, in_channels):
        super(SelfAttnPropagation, self).__init__()
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Initialize parameters with Xavier uniform initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, feature):
        """
        Forward pass of the self-attention layer.
        
        Args:
            feature (torch.Tensor): Input feature map of shape [B, C, H, W].
            
        Returns:
            out (torch.Tensor): Output feature map after applying self-attention, 
                                of shape [B, C, H, W].
            attention (torch.Tensor): Attention map of shape [B, N, N] where N = H * W.
        """
        B, C, H, W = feature.size()
        
        # Flatten spatial dimensions: [B, C, H, W] -> [B, H*W, C]
        x = feature.view(B, C, H * W).permute(0, 2, 1)  # [B, N, C], where N = H * W
        
        # Compute query, key, and value via linear projections
        query = self.q_proj(x)  # [B, N, C]
        key   = self.k_proj(x)  # [B, N, C]
        value = self.v_proj(x)  # [B, N, C]
        
        # Compute scaled dot-product attention scores: [B, N, N]
        scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(C)
        attention = torch.softmax(scores, dim=-1)
        
        # Propagate the features using the attention map: [B, N, C]
        out = torch.matmul(attention, value)
        
        # Reshape the output back to the original spatial dimensions: [B, C, H, W]
        out = out.permute(0, 2, 1).view(B, C, H, W)
        
        # Apply the residual connection with a learnable scaling parameter
        out = self.gamma * out + feature
        
        return out, attention
