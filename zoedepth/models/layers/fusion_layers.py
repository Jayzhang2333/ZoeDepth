import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F



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


def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True, relu=True):
    assert (kernel % 2) == 1, "only odd kernel is supported but kernel = {}".format(
        kernel
    )

    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding, bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    return layers

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
        self.conv_rgb = conv_bn_relu(33, 16, 3, 1, 1, bn=False)
        self.conv_dep = conv_bn_relu(1, 16, 3, 1, 1, bn=False)

        # self._trans = nn.Conv2d(33, 16, 1, 1, 0)
        self.fuse_conv1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fuse_conv2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.Sigmoid(),
        )
        self.fuse_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fuse_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.Sigmoid(),
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

    def forward(self, rgb, dep):
        f_rgb = self.conv_rgb(rgb)
        # f_rgb = rgb
        f_dep = self.conv_dep(dep)
        # f_dep = torch.cat([self._trans(f_rgb), f_dep], dim=1)
        f_dep = torch.cat([f_rgb, f_dep], dim=1)
        f_dep = self.fuse_conv1(f_dep) * self.fuse_conv2(f_dep)
        f = torch.cat([f_rgb, f_dep], dim=1)
        f = f + self.fuse_conv3(f) * self.fuse_conv4(f)

        return f
    


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
        c = self.conf_head(x)
        c = 0.8 * c + 0.1   # make confident between 0.1 to 0.9
        

        B, _, H, W = scale.shape
        valid_mask = (sparse_prior > 0).float()  # (B,1,H,W)

        scale = scale * (1 - valid_mask) + sparse_prior * valid_mask

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