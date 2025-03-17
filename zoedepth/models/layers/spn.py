import torch
import torch.nn as nn
# from zoedepth.plugins.modulated_deform_conv_func import ModulatedDeformConvFunction
import torch.nn.functional as F

def compute_affinity_map_9(depth_map, sigma=0.1):
    B, _, H, W = depth_map.shape
    pad_depth = F.pad(depth_map, (1,1,1,1), mode='replicate')

    affinity_maps = []
    # shifts including the center pixel (0,0)
    shifts = [(-1,-1), (-1,0), (-1,1),
              (0,-1),  (0,0),  (0,1),
              (1,-1),  (1,0),  (1,1)]

    for dy, dx in shifts:
        neighbor = pad_depth[:, :, 1+dy:H+1+dy, 1+dx:W+1+dx]
        diff = torch.abs(depth_map - neighbor)
        affinity = torch.exp(-diff / sigma)
        affinity_maps.append(affinity)

    affinity_map = torch.cat(affinity_maps, dim=1)  # shape: (B, 9, H, W)
    return affinity_map

class DepthGradientAffinity9(nn.Module):
    def __init__(self):
        super().__init__()
        # Adjusted output channels from 8 → 9
        self.conv = nn.Conv2d(3, 9, kernel_size=3, padding=1)

        sobel_x = torch.tensor([[[[-1,0,1], [-2,0,2], [-1,0,1]]]])
        sobel_y = torch.tensor([[[[-1,-2,-1], [0,0,0], [1,2,1]]]])
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, depth):
        grad_x = F.conv2d(depth, self.sobel_x, padding=1)
        grad_y = F.conv2d(depth, self.sobel_y, padding=1)

        feat = torch.cat([depth, grad_x, grad_y], dim=1)  # (B, 3, H, W)
        affinity = self.conv(feat)  # (B, 9, H, W)

        return affinity

class LightweightAffinityNet9(nn.Module):
    def __init__(self):
        super().__init__()
        self.dwconv = nn.Conv2d(1, 1, kernel_size=3, padding=1, groups=1, bias=False)
        self.pwconv = nn.Conv2d(1, 9, kernel_size=1, bias=True)  # from 8→9

    def forward(self, depth):
        x = self.dwconv(depth)
        x = torch.relu(x)
        affinity = self.pwconv(x)
        return affinity


class CSPNAccelerate(nn.Module):
    def __init__(self, kernel_size):
        super(CSPNAccelerate, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, kernel, input, input0):  
        bs = input.size()[0]
        h, w = input.size()[2], input.size()[3]

        input_im2col = F.unfold(input, self.kernel_size, 1, self.kernel_size//2, 1)

        kernel = kernel.reshape(bs, self.kernel_size * self.kernel_size, h * w)

        input0 = input0.view(bs, 1, h * w)
        mid_index = int((self.kernel_size * self.kernel_size - 1) / 2)
        input_im2col[:, mid_index:mid_index + 1, :] = input0

        output = (input_im2col * kernel).sum(dim=1)
        return output.view(bs, 1, h, w)
    
# class NLSPN(nn.Module):
#     def __init__(self, ch_g, ch_f, k_g, k_f, prop_time = 6, affinity = 'ASS', affinity_gamma = 0.5, conf_prop = False, legacy = False, preserve_input = True):
#         super(NLSPN, self).__init__()

#         # Guidance : [B x ch_g x H x W]
#         # Feature : [B x ch_f x H x W]

#         # Currently only support ch_f == 1
#         assert ch_f == 1, 'only tested with ch_f == 1 but {}'.format(ch_f)

#         assert (k_g % 2) == 1, \
#             'only odd kernel is supported but k_g = {}'.format(k_g)
#         pad_g = int((k_g - 1) / 2)
#         assert (k_f % 2) == 1, \
#             'only odd kernel is supported but k_f = {}'.format(k_f)
#         pad_f = int((k_f - 1) / 2)

#         self.prop_time = prop_time
#         self.affinity = affinity
#         self.affinity_gamma = affinity_gamma
#         self.conf_prop = conf_prop
#         self.legacy = legacy
#         self.preserve_input = preserve_input

#         self.ch_g = ch_g
#         self.ch_f = ch_f
#         self.k_g = k_g
#         self.k_f = k_f
#         # Assume zero offset for center pixels
#         self.num = self.k_f * self.k_f - 1
#         self.idx_ref = self.num // 2

#         if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
#             self.conv_offset_aff = nn.Conv2d(
#                 self.ch_g, 3 * self.num, kernel_size=self.k_g, stride=1,
#                 padding=pad_g, bias=True
#             )
#             self.conv_offset_aff.weight.data.zero_()
#             self.conv_offset_aff.bias.data.zero_()

#             if self.affinity == 'TC':
#                 self.aff_scale_const = nn.Parameter(self.num * torch.ones(1))
#                 self.aff_scale_const.requires_grad = False
#             elif self.affinity == 'TGASS':
#                 self.aff_scale_const = nn.Parameter(
#                     self.affinity_gamma * self.num * torch.ones(1))
#             else:
#                 self.aff_scale_const = nn.Parameter(torch.ones(1))
#                 self.aff_scale_const.requires_grad = False
#         else:
#             raise NotImplementedError

#         # Dummy parameters for gathering
#         self.w = nn.Parameter(torch.ones((self.ch_f, 1, self.k_f, self.k_f)))
#         self.b = nn.Parameter(torch.zeros(self.ch_f))

#         self.w.requires_grad = False
#         self.b.requires_grad = False

#         self.w_conf = nn.Parameter(torch.ones((1, 1, 1, 1)))
#         self.w_conf.requires_grad = False

#         self.stride = 1
#         self.padding = pad_f
#         self.dilation = 1
#         self.groups = self.ch_f
#         self.deformable_groups = 1
#         self.im2col_step = 64

#     def _get_offset_affinity(self, guidance, confidence=None, rgb=None):
#         B, _, H, W = guidance.shape

#         if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
#             offset_aff = self.conv_offset_aff(guidance)
#             o1, o2, aff = torch.chunk(offset_aff, 3, dim=1)

#             # Add zero reference offset
#             offset = torch.cat((o1, o2), dim=1).view(B, self.num, 2, H, W)
#             list_offset = list(torch.chunk(offset, self.num, dim=1))
#             list_offset.insert(self.idx_ref,
#                                torch.zeros((B, 1, 2, H, W)).type_as(offset))
#             offset = torch.cat(list_offset, dim=1).view(B, -1, H, W)

#             if self.affinity in ['AS', 'ASS']:
#                 pass
#             elif self.affinity == 'TC':
#                 aff = torch.tanh(aff/100) / self.aff_scale_const
#             elif self.affinity == 'TGASS':
#                 aff = torch.tanh(aff/100) / (self.aff_scale_const + 1e-8)
#             else:
#                 raise NotImplementedError
#         else:
#             raise NotImplementedError

#         # Apply confidence
#         # TODO : Need more efficient way
#         if self.conf_prop:
#             list_conf = []
#             offset_each = torch.chunk(offset, self.num + 1, dim=1)

#             modulation_dummy = torch.ones((B, 1, H, W)).type_as(offset).detach()

#             for idx_off in range(0, self.num + 1):
#                 ww = idx_off % self.k_f
#                 hh = idx_off // self.k_f

#                 if ww == (self.k_f - 1) / 2 and hh == (self.k_f - 1) / 2:
#                     continue

#                 offset_tmp = offset_each[idx_off].detach()

#                 # NOTE : Use --legacy option ONLY for the pre-trained models
#                 # for ECCV20 results.
#                 if self.legacy:
#                     offset_tmp[:, 0, :, :] = \
#                         offset_tmp[:, 0, :, :] + hh - (self.k_f - 1) / 2
#                     offset_tmp[:, 1, :, :] = \
#                         offset_tmp[:, 1, :, :] + ww - (self.k_f - 1) / 2

#                 conf_tmp = ModulatedDeformConvFunction.apply(
#                     confidence, offset_tmp, modulation_dummy, self.w_conf,
#                     self.b, self.stride, 0, self.dilation, self.groups,
#                     self.deformable_groups, self.im2col_step)
#                 list_conf.append(conf_tmp)

#             conf_aff = torch.cat(list_conf, dim=1)
#             aff = aff * conf_aff.contiguous()

#         # Affinity normalization
#         aff_abs = torch.abs(aff)
#         aff_abs_sum = torch.sum(aff_abs, dim=1, keepdim=True) + 1e-4

#         if self.affinity in ['ASS', 'TGASS']:
#             aff_abs_sum[aff_abs_sum < 1.0] = 1.0

#         if self.affinity in ['AS', 'ASS', 'TGASS']:
#             aff = aff / aff_abs_sum

#         aff_sum = torch.sum(aff, dim=1, keepdim=True)
#         aff_ref = 1.0 - aff_sum

#         list_aff = list(torch.chunk(aff, self.num, dim=1))
#         list_aff.insert(self.idx_ref, aff_ref)
#         aff = torch.cat(list_aff, dim=1)

#         return offset, aff

#     def _propagate_once(self, feat, offset, aff):
#         feat = ModulatedDeformConvFunction.apply(
#             feat, offset, aff, self.w, self.b, self.stride, self.padding,
#             self.dilation, self.groups, self.deformable_groups, self.im2col_step
#         )

#         return feat

#     def forward(self, feat_init, guidance, confidence=None, feat_fix=None,
#                 rgb=None):
#         assert self.ch_g == guidance.shape[1]
#         assert self.ch_f == feat_init.shape[1]

#         if self.conf_prop:
#             assert confidence is not None

#         if self.conf_prop:
#             offset, aff = self._get_offset_affinity(guidance, confidence, rgb)
#         else:
#             offset, aff = self._get_offset_affinity(guidance, None, rgb)

#         # Propagation
#         if self.preserve_input:
#             assert feat_init.shape == feat_fix.shape
#             mask_fix = torch.sum(feat_fix > 0.0, dim=1, keepdim=True).detach()
#             mask_fix = (mask_fix > 0.0).type_as(feat_fix)

#         feat_result = feat_init

#         list_feat = []

#         for k in range(1, self.prop_time + 1):
#             # Input preservation for each iteration
#             if self.preserve_input:
#                 feat_result = (1.0 - mask_fix) * feat_result \
#                               + mask_fix * feat_fix

#             feat_result = self._propagate_once(feat_result, offset, aff)

#             list_feat.append(feat_result)

#         return feat_result, list_feat, offset, aff, self.aff_scale_const.data