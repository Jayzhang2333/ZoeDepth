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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np


KEY_OUTPUT = 'metric_depth'


def extract_key(prediction, key):
    if isinstance(prediction, dict):
        return prediction[key]
    return prediction

import torch
import torch.nn as nn
import torch.nn.functional as F


class VNL_Loss(nn.Module):
    """
    Virtual Normal Loss Function.
    """
    def __init__(self, focal_x, focal_y, input_size,
                 delta_cos=0.867, delta_diff_x=0.6,
                 delta_diff_y=0.6, delta_diff_z=0.6,
                 delta_z=0.0001, sample_ratio=0.15):
        super(VNL_Loss, self).__init__()
        self.fx = torch.tensor([focal_x], dtype=torch.float32).cuda()
        self.fy = torch.tensor([focal_y], dtype=torch.float32).cuda()
        self.input_size = input_size
        self.u0 = torch.tensor(input_size[1] // 2, dtype=torch.float32).cuda()
        self.v0 = torch.tensor(input_size[0] // 2, dtype=torch.float32).cuda()
        self.init_image_coor()
        self.delta_cos = delta_cos
        self.delta_diff_x = delta_diff_x
        self.delta_diff_y = delta_diff_y
        self.delta_diff_z = delta_diff_z
        self.delta_z = delta_z
        self.sample_ratio = sample_ratio

    def init_image_coor(self):
        x_row = np.arange(0, self.input_size[1])
        x = np.tile(x_row, (self.input_size[0], 1))
        x = x[np.newaxis, :, :].astype(np.float32)
        x = torch.from_numpy(x.copy()).cuda()
        self.u_u0 = x - self.u0

        y_col = np.arange(0, self.input_size[0])
        y = np.tile(y_col, (self.input_size[1], 1)).T
        y = y[np.newaxis, :, :].astype(np.float32)
        y = torch.from_numpy(y.copy()).cuda()
        self.v_v0 = y - self.v0

    def transfer_xyz(self, depth):
        x = self.u_u0 * torch.abs(depth) / self.fx
        y = self.v_v0 * torch.abs(depth) / self.fy
        z = depth
        # [B, H, W, 3]
        pw = torch.cat([x, y, z], 1).permute(0, 2, 3, 1)
        return pw

    def select_index(self, mask=None):
        valid_width = self.input_size[1]
        valid_height = self.input_size[0]

        if mask is not None:
            # Here mask is expected to be a numpy array of shape (H, W)
            mask_flat = mask.flatten()
            valid_indices = np.where(mask_flat)[0]
            num_valid = len(valid_indices)
            # Sample indices only from the valid ones
            p1 = np.random.choice(valid_indices, int(num_valid * self.sample_ratio), replace=True)
            p2 = np.random.choice(valid_indices, int(num_valid * self.sample_ratio), replace=True)
            p3 = np.random.choice(valid_indices, int(num_valid * self.sample_ratio), replace=True)
        else:
            # Sample from all pixels
            num = valid_width * valid_height
            p1 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
            p2 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
            p3 = np.random.choice(num, int(num * self.sample_ratio), replace=True)

        # Convert flat indices to x, y coordinates
        p1_x = p1 % valid_width
        p1_y = (p1 // valid_width).astype(np.int_)
        
        p2_x = p2 % valid_width
        p2_y = (p2 // valid_width).astype(np.int_)
        
        p3_x = p3 % valid_width
        p3_y = (p3 // valid_width).astype(np.int_)

        p123 = {
            'p1_x': p1_x, 'p1_y': p1_y,
            'p2_x': p2_x, 'p2_y': p2_y,
            'p3_x': p3_x, 'p3_y': p3_y
        }
        return p123

    def form_pw_groups(self, p123, pw):
        """
        Form 3D point groups with 3 points in each group.
        :param p123: dictionary containing indices for p1, p2, and p3.
        :param pw: 3D points [B, H, W, 3]
        :return: Tensor of shape [B, N, 3, 3]
        """
        p1_x = p123['p1_x']
        p1_y = p123['p1_y']
        p2_x = p123['p2_x']
        p2_y = p123['p2_y']
        p3_x = p123['p3_x']
        p3_y = p123['p3_y']

        pw1 = pw[:, p1_y, p1_x, :]
        pw2 = pw[:, p2_y, p2_x, :]
        pw3 = pw[:, p3_y, p3_x, :]
        # Concatenate along a new axis for the three points
        pw_groups = torch.cat([pw1[:, :, :, np.newaxis],
                               pw2[:, :, :, np.newaxis],
                               pw3[:, :, :, np.newaxis]], 3)
        return pw_groups

    def filter_mask(self, p123, gt_xyz, delta_cos=0.867,
                    delta_diff_x=0.005,
                    delta_diff_y=0.005,
                    delta_diff_z=0.005):
        """
        Compute a mask to filter out groups that are too linear or invalid.
        :param p123: sampled point indices
        :param gt_xyz: 3D points from the ground truth depth [B, H, W, 3]
        :return: (mask, pw_groups) where mask is a boolean tensor and pw_groups are the 3D groups.
        """
        pw = self.form_pw_groups(p123, gt_xyz)
        pw12 = pw[:, :, :, 1] - pw[:, :, :, 0]
        pw13 = pw[:, :, :, 2] - pw[:, :, :, 0]
        pw23 = pw[:, :, :, 2] - pw[:, :, :, 1]
        pw_diff = torch.cat([pw12[:, :, :, np.newaxis],
                             pw13[:, :, :, np.newaxis],
                             pw23[:, :, :, np.newaxis]], 3)  # [B, N, 3, 3]
        m_batchsize, groups, coords, index = pw_diff.shape
        proj_query = pw_diff.view(m_batchsize * groups, -1, index).permute(0, 2, 1)
        proj_key = pw_diff.view(m_batchsize * groups, -1, index)
        q_norm = proj_query.norm(2, dim=2)
        nm = torch.bmm(q_norm.view(m_batchsize * groups, index, 1), q_norm.view(m_batchsize * groups, 1, index))
        energy = torch.bmm(proj_query, proj_key)
        norm_energy = energy / (nm + 1e-8)
        norm_energy = norm_energy.view(m_batchsize, groups, -1)
        mask_cos = torch.sum((norm_energy > delta_cos) + (norm_energy < -delta_cos), dim=2) > 3
        mask_pad = torch.sum(pw[:, :, 2, :] > self.delta_z, dim=2) == 3
        mask_x = torch.sum(torch.abs(pw_diff[:, :, 0, :]) < delta_diff_x, dim=2) > 0
        mask_y = torch.sum(torch.abs(pw_diff[:, :, 1, :]) < delta_diff_y, dim=2) > 0
        mask_z = torch.sum(torch.abs(pw_diff[:, :, 2, :]) < delta_diff_z, dim=2) > 0
        mask_ignore = (mask_x & mask_y & mask_z) | mask_cos
        mask_near = ~mask_ignore
        mask = mask_pad & mask_near

        return mask, pw

    def select_points_groups(self, gt_depth, pred_depth, mask=None):
        """
        For each image in the batch, select valid 3D point groups based on its own mask.
        :param gt_depth: ground truth depth [B, C, H, W]
        :param pred_depth: predicted depth [B, C, H, W]
        :param mask: optional binary mask [B, C, H, W] indicating valid pixels.
        :return: (all_pw_groups_gt, all_pw_groups_pred) concatenated over the batch.
        """
        pw_gt = self.transfer_xyz(gt_depth)    # [B, H, W, 3]
        pw_pred = self.transfer_xyz(pred_depth)  # [B, H, W, 3]
        B, C, H, W = gt_depth.shape

        groups_gt_list = []
        groups_pred_list = []
        for i in range(B):
            # Use the mask for the i-th image if provided (assumed to be in channel 0)
            if mask is not None:
                mask_np = mask[i, 0].detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask[i, 0]
            else:
                mask_np = None

            # Sample indices for this image based on its mask
            p123 = self.select_index(mask=mask_np)
            # Process ground truth groups for this image (using unsqueezed batch dimension)
            mask_valid, pw_groups_gt = self.filter_mask(p123, pw_gt[i:i+1],
                                                        delta_cos=0.867,
                                                        delta_diff_x=0.005,
                                                        delta_diff_y=0.005,
                                                        delta_diff_z=0.005)
            pw_groups_pred = self.form_pw_groups(p123, pw_pred[i:i+1])
            # Prevent division-by-zero in later calculations
            pw_groups_pred[pw_groups_pred[:, :, 2, :] == 0] = 0.0001

            # Broadcast the filter mask to select valid point groups
            mask_broadcast = mask_valid.repeat(1, 9).reshape(1, 3, 3, -1).permute(0, 3, 1, 2)
            # Use boolean indexing to filter and then reshape to [1, N, 3, 3]
            pw_groups_pred_not_ignore = pw_groups_pred[mask_broadcast].reshape(1, -1, 3, 3)
            pw_groups_gt_not_ignore = pw_groups_gt[mask_broadcast].reshape(1, -1, 3, 3)
            groups_gt_list.append(pw_groups_gt_not_ignore)
            groups_pred_list.append(pw_groups_pred_not_ignore)

        # Concatenate groups across the batch (along the group dimension)
        all_pw_groups_gt = torch.cat(groups_gt_list, dim=1)
        all_pw_groups_pred = torch.cat(groups_pred_list, dim=1)
        return all_pw_groups_gt, all_pw_groups_pred

    def forward(self, pred_depth, gt_depth, mask=None, select=True, interpolate = True):
        """
        Compute the virtual normal loss.
        :param gt_depth: ground truth depth map [B, C, H, W]
        :param pred_depth: predicted depth map [B, C, H, W]
        :param mask: optional binary mask [B, C, H, W] indicating valid pixels.
        :param select: whether to select a subset of points for loss computation.
        :return: computed loss.
        """

        if pred_depth.shape[-1] != gt_depth.shape[-1]:
            pred_depth = F.interpolate(pred_depth, gt_depth.shape[-2:], mode = 'bilinear', align_corners = True)
        gt_points, dt_points = self.select_points_groups(gt_depth, pred_depth, mask=mask)

        gt_p12 = gt_points[:, :, :, 1] - gt_points[:, :, :, 0]
        gt_p13 = gt_points[:, :, :, 2] - gt_points[:, :, :, 0]
        dt_p12 = dt_points[:, :, :, 1] - dt_points[:, :, :, 0]
        dt_p13 = dt_points[:, :, :, 2] - dt_points[:, :, :, 0]

        gt_normal = torch.cross(gt_p12, gt_p13, dim=2)
        dt_normal = torch.cross(dt_p12, dt_p13, dim=2)
        dt_norm = torch.norm(dt_normal, 2, dim=2, keepdim=True)
        gt_norm = torch.norm(gt_normal, 2, dim=2, keepdim=True)
        dt_mask = (dt_norm == 0.0).to(torch.float32) * 0.01
        gt_mask = (gt_norm == 0.0).to(torch.float32) * 0.01
        gt_norm = gt_norm + gt_mask
        dt_norm = dt_norm + dt_mask
        gt_normal = gt_normal / gt_norm
        dt_normal = dt_normal / dt_norm

        loss = torch.abs(gt_normal - dt_normal)
        loss = torch.sum(torch.sum(loss, dim=2), dim=0)
        if select:
            loss, _ = torch.sort(loss, dim=0, descending=False)
            loss = loss[int(loss.size(0) * 0.25):]
        loss = torch.mean(loss)
        return loss*5

class NegativeLogLikelihoodLoss(nn.Module):
    """
    Negative Log Likelihood Loss for a regression task with uncertainty estimation.
    
    This loss function assumes that the network outputs two separate predictions:
        - depth_pred: the predicted depth (mean), μ.
        - uncertainty_pred: the predicted log variance, s = log(σ^2).
    
    The per-pixel loss is computed as:
    
        loss = 0.5 * exp(-s) * (y - μ)^2 + 0.5 * s
        
    Optionally, a binary mask can be provided to select valid pixels only.
    """
    def __init__(self) -> None:
        super(NegativeLogLikelihoodLoss, self).__init__()
        self.name = "NegativeLogLikelihoodLoss"

    def forward(self, depth_pred, uncertainty_pred, target, mask=None, interpolate=True):
        """
        Parameters:
            depth_pred (torch.Tensor): Predicted depth (mean), shape (N, 1, H_pred, W_pred).
            uncertainty_pred (torch.Tensor): Predicted log variance, shape (N, 1, H_pred, W_pred).
            target (torch.Tensor): Ground truth depth, shape (N, 1, H, W).
            mask (torch.Tensor, optional): Binary mask to specify valid pixels. Shape (N, 1, H, W).
            interpolate (bool, optional): Whether to interpolate predictions to the target size if needed.
        Returns:
            torch.Tensor: The computed loss (scalar).
        """
        # Interpolate depth prediction if its spatial dimensions don't match the target.
        if interpolate and depth_pred.shape[-1] != target.shape[-1]:
            depth_pred = F.interpolate(depth_pred, size=target.shape[-2:], mode='bilinear', align_corners=True)
        
        # Interpolate uncertainty prediction if necessary.
        if interpolate and uncertainty_pred.shape[-1] != target.shape[-1]:
            uncertainty_pred = F.interpolate(uncertainty_pred, size=target.shape[-2:], mode='bilinear', align_corners=True)
        
        # Compute the per-pixel negative log likelihood loss.
        loss = 0.5 * torch.exp(-uncertainty_pred) * (target - depth_pred) ** 2 + 0.5 * uncertainty_pred

        # If a mask is provided, compute the masked average loss.
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss



class InvNegativeLogLikelihoodLoss(nn.Module):
    """
    Negative Log Likelihood Loss for a regression task with uncertainty estimation.
    
    This loss function assumes that the network outputs two separate predictions:
        - depth_pred: the predicted depth (mean), μ.
        - uncertainty_pred: the predicted log variance, s = log(σ^2).
    
    The per-pixel loss is computed as:
    
        loss = 0.5 * exp(-s) * (y - μ)^2 + 0.5 * s
        
    Optionally, a binary mask can be provided to select valid pixels only.
    """
    def __init__(self) -> None:
        super(InvNegativeLogLikelihoodLoss, self).__init__()
        self.name = "NegativeLogLikelihoodLoss"

    def forward(self, depth_pred, uncertainty_pred, target, mask=None, interpolate=True):
        """
        Parameters:
            depth_pred (torch.Tensor): Predicted depth (mean), shape (N, 1, H_pred, W_pred).
            uncertainty_pred (torch.Tensor): Predicted log variance, shape (N, 1, H_pred, W_pred).
            target (torch.Tensor): Ground truth depth, shape (N, 1, H, W).
            mask (torch.Tensor, optional): Binary mask to specify valid pixels. Shape (N, 1, H, W).
            interpolate (bool, optional): Whether to interpolate predictions to the target size if needed.
        Returns:
            torch.Tensor: The computed loss (scalar).
        """
        # Interpolate depth prediction if its spatial dimensions don't match the target.
        if interpolate and depth_pred.shape[-1] != target.shape[-1]:
            depth_pred = F.interpolate(depth_pred, size=target.shape[-2:], mode='bilinear', align_corners=True)
        
        # Interpolate uncertainty prediction if necessary.
        if interpolate and uncertainty_pred.shape[-1] != target.shape[-1]:
            uncertainty_pred = F.interpolate(uncertainty_pred, size=target.shape[-2:], mode='bilinear', align_corners=True)
        
        # Compute the per-pixel negative log likelihood loss.
        if mask is not None:
            # Convert mask to boolean if it's not already.
            valid = mask.bool() if mask.dtype != torch.bool else mask

            # Select only the valid pixels.
            target_valid = target[valid]
            target_valid = 1.0 / target_valid
            depth_pred_valid = depth_pred[valid]
            uncertainty_pred_valid = uncertainty_pred[valid]

            # Compute the loss only on valid pixels.
            loss = 0.5 * torch.exp(-uncertainty_pred_valid) * (target_valid - depth_pred_valid) ** 2 + 0.5 * uncertainty_pred_valid
            loss = loss.mean() *100 # Average loss over the valid pixels.
        else:
            loss = 0.5 * torch.exp(-uncertainty_pred) * (target - depth_pred) ** 2 + 0.5 * uncertainty_pred
            loss = loss.mean() * 100
        
        return loss



class L1SmoothLoss(nn.Module):
    """Root Mean Squared Error (RMSE)"""

    def __init__(self) -> None:
        super(L1SmoothLoss, self).__init__()

        self.name = "L1SmoothLoss"

        self.l1smooth_loss = nn.SmoothL1Loss()

    def forward(self, input, target, mask=None, interpolate=True):

        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        if target.ndim == 3:
            target = target.unsqueeze(1)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            input = input[mask]
            target = target[mask]

        with amp.autocast(enabled=False):
            loss = self.l1smooth_loss(input, target)

        return loss

class RMSELoss(nn.Module):
    """Root Mean Squared Error (RMSE)"""

    def __init__(self) -> None:
        super(RMSELoss, self).__init__()

        self.name = "RMSELoss"

        self.mse_loss = nn.MSELoss()

    def forward(self, input, target, mask=None, interpolate=True):

        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        if target.ndim == 3:
            target = target.unsqueeze(1)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            input = input[mask]
            target = target[mask]

        with amp.autocast(enabled=False):
            loss = torch.sqrt(self.mse_loss(input, target))

        return loss
    
class InvRMSELoss(nn.Module):
    """Root Mean Squared Error (RMSE)"""

    def __init__(self) -> None:
        super(InvRMSELoss, self).__init__()

        self.name = "iRMSELoss"

        self.mse_loss = nn.MSELoss()

    def forward(self, input, target, mask=None, interpolate=True):

        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        if target.ndim == 3:
            target = target.unsqueeze(1)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            input = input[mask]
            input = input*100
            target = target[mask]
            target = 1.0/(target+ 1e-6)
            target = target*100

        with amp.autocast(enabled=False):
            loss = torch.sqrt(self.mse_loss(input, target))

        return loss
    
class InvInfRMSELoss(nn.Module):
    """Inverse Infinite-region RMSE Loss"""

    def __init__(self) -> None:
        super(InvInfRMSELoss, self).__init__()
        self.name = "InvInfRMSELoss"
        self.mse_loss = nn.MSELoss()

    def forward(self, input, target, max_depth=20, inf_mask=None, interpolate=True):
        # Extract the output tensor
        input = extract_key(input, KEY_OUTPUT)

        # Upsample if dimensions differ
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)

        # If no mask provided or no infinite pixels, return zero loss
        if inf_mask is None:
            return torch.tensor(0.0, device=input.device, dtype=input.dtype)

        # Ensure mask has channel dimension
        if inf_mask.ndim == 3:
            inf_mask = inf_mask.unsqueeze(1)

        # If mask contains no True values, return zero loss
        if not inf_mask.any():
            return torch.tensor(0.0, device=input.device, dtype=input.dtype)

        # Select infinite-region pixels and scale
        # show_images_two_sources(input,inf_mask)
        vals = input[inf_mask]
        vals = vals * 100.0
        inf_tensor = torch.full_like(vals, 100.0 / max_depth)

        # Compute RMSE without AMP autocast
        with amp.autocast(enabled=False):
            loss = torch.sqrt(self.mse_loss(vals, inf_tensor))

        return loss
    
class MultiScaleGradientLoss(nn.Module):
    """
    Multi-scale gradient loss on metric depths:

        L = (1/K) * sum_{k=0..K-1} [ 
                mean(|∂_x d_pred^k - ∂_x d_gt^k|) 
              + mean(|∂_y d_pred^k - ∂_y d_gt^k|) 
            ]

    where d_pred^k, d_gt^k, and mask^k are downsampled by 2×2 avg-pool at each scale.
    Loss is only computed where mask^k==True for both pixels involved in each gradient.
    """

    def __init__(self, num_scales: int = 3, eps: float = 1e-6):
        super().__init__()
        self.num_scales = num_scales
        self.eps        = eps

    def forward(self,
                inv_pred: torch.Tensor,
                gt:   torch.Tensor,
                mask: torch.Tensor = None,
                interpolate: bool = True) -> torch.Tensor:
        """
        Parameters
        ----------
        pred, gt : (B,1,H,W) or (B,H,W)
            metric depth maps (in metres).
        mask : (B,H,W) or (B,1,H,W), optional
            boolean mask of valid GT pixels.
        interpolate : whether to upsample pred to gt resolution.
        """

        # 1) Optionally resize pred → gt
        if inv_pred.shape[-1] != gt.shape[-1] and interpolate:
            inv_pred = nn.functional.interpolate(
                inv_pred, gt.shape[-2:], mode='bilinear', align_corners=True)
            
        if inv_pred.dim() == 3:
            inv_pred = inv_pred.unsqueeze(1)
        if gt.dim() == 3:
            gt   = gt.unsqueeze(1)

        # 2) inverse
        # inv_pred = 1.0 / (pred + self.eps)
        inv_pred = inv_pred*100
        inv_gt   = 1.0 / (gt   + self.eps) * 100

        # 3) Prepare mask at full resolution
        if mask is not None:
            if mask.dim()==3: mask = mask.unsqueeze(1)
            mask_down = mask.bool()
        else:
            mask_down = None

        total_loss = 0.0
        depth_pred = inv_pred
        depth_gt   = inv_gt


        for k in range(self.num_scales):
            # compute gradients along x and y
            dx_pred = depth_pred[:, :, :, :-1] - depth_pred[:, :, :, 1:]
            dx_gt   = depth_gt  [:, :, :, :-1] - depth_gt  [:, :, :, 1:]
            d_dx    = dx_pred - dx_gt

            dy_pred = depth_pred[:, :, :-1, :] - depth_pred[:, :, 1:, :]
            dy_gt   = depth_gt  [:, :, :-1, :] - depth_gt  [:, :, 1:, :]
            d_dy    = dy_pred - dy_gt

            if mask_down is not None:
                # only keep gradients where both pixels are valid
                mx = mask_down[:, :, :, :-1] & mask_down[:, :, :, 1:]
                my = mask_down[:, :, :-1, :] & mask_down[:, :, 1:, :]

                loss_x = (d_dx.abs() * mx).sum() / (mx.sum() + self.eps)
                loss_y = (d_dy.abs() * my).sum() / (my.sum() + self.eps)
            else:
                loss_x = d_dx.abs().mean()
                loss_y = d_dy.abs().mean()

            total_loss += (loss_x + loss_y)

            # downsample for next scale
            if k < self.num_scales - 1:
                # show_images_two_sources(depth_pred, mask_down)
                depth_pred = F.avg_pool2d(depth_pred, kernel_size=2, stride=2)
                depth_gt   = F.avg_pool2d(depth_gt,   kernel_size=2, stride=2)
                if mask_down is not None:
                    # any pixel in the 2×2 patch is valid → keep it
                    mask_down = (F.avg_pool2d(mask_down.float(),
                                              kernel_size=2,
                                              stride=2) > 0)

        return total_loss / self.num_scales

# class MultiScaleGradientLoss(nn.Module):
#     """
#     Multi-scale gradient loss on inverse-depth residuals:
#         R = 1/(gt + eps) - 1/(pred + eps)
#     L = (1/K) * sum_{k=0..K-1} [ mean(|∂_x R^k|) + mean(|∂_y R^k|) ]
#     where R^k and mask are downsampled by 2×2 avg-pool at each scale.
#     """

#     def __init__(self, num_scales: int = 3, eps: float = 1e-6):
#         super().__init__()
#         self.num_scales = num_scales
#         self.eps        = eps

#     def forward(self,
#                 inv_pred: torch.Tensor,
#                 gt:   torch.Tensor,
#                 mask: torch.Tensor = None,
#                 interpolate=True) -> torch.Tensor:
#         """
#         Parameters
#         ----------
#         pred, gt : (B,1,H,W) or (B,H,W)
#             metric depth maps
#         mask : (B,H,W) or (B,1,H,W), optional
#             boolean mask of valid GT pixels
#         """
#         if inv_pred.shape[-1] != gt.shape[-1] and interpolate:
#             inv_pred = nn.functional.interpolate(
#                 inv_pred, gt.shape[-2:], mode='bilinear', align_corners=True)

#         # 1) ensure channel dim
#         if inv_pred.dim() == 3:
#             inv_pred = inv_pred.unsqueeze(1)
#         if gt.dim() == 3:
#             gt   = gt.unsqueeze(1)

#         # 2) inverse
#         # inv_pred = 1.0 / (pred + self.eps)
#         inv_pred = inv_pred*100
#         inv_gt   = 1.0 / (gt   + self.eps) * 100

#         # 3) prepare mask_down to track at each scale
#         if mask is not None:
#             if mask.dim() == 3:
#                 mask = mask.unsqueeze(1)
#             mask_down = mask.bool()
#         else:
#             mask_down = None

#         # 4) residual at full res
#         R = inv_gt - inv_pred
#         total_loss = 0.0

#         for k in range(self.num_scales):
#             B, C, H, W = R.shape
#             # breakpoint()
#             # gradients of R^k
#             grad_x = R[:, :, :, :-1] - R[:, :, :, 1:]
#             grad_y = R[:, :, :-1, :] - R[:, :, 1:, :]

#             if mask_down is not None:
#                 # use the **current** mask_down for this scale
#                 mask_k = mask_down

#                 # only keep grads where both pixels valid
#                 mask_x = mask_k[:, :, :, :-1] & mask_k[:, :, :, 1:]
#                 mask_y = mask_k[:, :, :-1, :] & mask_k[:, :, 1:, :]

#                 loss_x = (grad_x.abs() * mask_x).sum() / (mask_x.sum() + self.eps)
#                 loss_y = (grad_y.abs() * mask_y).sum() / (mask_y.sum() + self.eps)
#             else:
#                 loss_x = grad_x.abs().mean()
#                 loss_y = grad_y.abs().mean()

#             total_loss += (loss_x + loss_y)
#             show_images_two_sources(R, mask_down)
#             # downsample R—and mask_down—by 2×2 avg-pool for the next scale
#             if k < self.num_scales - 1:
#                 R = F.avg_pool2d(R, kernel_size=2, stride=2)
#                 if mask_down is not None:
#                     # avg_pool2d(mask_down.float()) > 0  matches exactly R’s receptive field
#                     mask_down = (F.avg_pool2d(mask_down.float(),
#                                               kernel_size=2,
#                                               stride=2) > 0)

#         return  (total_loss / self.num_scales)



# Main loss function used for ZoeDepth. Copy/paste from AdaBins repo (https://github.com/shariqfarooq123/AdaBins/blob/0952d91e9e762be310bb4cd055cbfe2448c0ce20/loss.py#L7)
class SILogLoss(nn.Module):
    """SILog loss (pixel-wise)"""
    def __init__(self, beta=0.15):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'
        self.beta = beta

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        if target.ndim == 3:
            target = target.unsqueeze(1)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            input = input[mask]
            target = target[mask]

        with amp.autocast(enabled=False):  # amp causes NaNs in this loss function
            alpha = 1e-7
            g = torch.log(input + alpha) - torch.log(target + alpha)

            # n, c, h, w = g.shape
            # norm = 1/(h*w)
            # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

            Dg = torch.var(g) + self.beta * torch.pow(torch.mean(g), 2)

            loss = 10 * torch.sqrt(Dg)

        if torch.isnan(loss):
            print("Nan SILog loss")
            print("input:", input.shape)
            print("target:", target.shape)
            print(mask)
            print("G", torch.sum(torch.isnan(g)))
            print("Input min max", torch.min(input), torch.max(input))
            print("Target min max", torch.min(target), torch.max(target))
            print("Dg", torch.isnan(Dg))
            print("loss", torch.isnan(loss))

        if not return_interpolated:
            return loss

        return loss, intr_input
    
class InvSILogLoss(nn.Module):
    """SILog loss (pixel-wise)"""
    def __init__(self, beta=0.85):
        super(InvSILogLoss, self).__init__()
        self.name = 'iSILog'
        self.beta = beta

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        if target.ndim == 3:
            target = target.unsqueeze(1)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            input = input[mask]
            input = input*100
            target = target[mask]
            target = 1.0 / target
            target = target*100

        with amp.autocast(enabled=False):  # amp causes NaNs in this loss function
            alpha = 1e-7
            g = torch.log(input + alpha) - torch.log(target + alpha)

            # n, c, h, w = g.shape
            # norm = 1/(h*w)
            # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

            Dg = torch.var(g) + self.beta * torch.pow(torch.mean(g), 2)

            loss = 10 * torch.sqrt(Dg)

        if torch.isnan(loss):
            print("Nan SILog loss")
            print("input:", input.shape)
            print("target:", target.shape)
            print(mask)
            print("G", torch.sum(torch.isnan(g)))
            print("Input min max", torch.min(input), torch.max(input))
            print("Target min max", torch.min(target), torch.max(target))
            print("Dg", torch.isnan(Dg))
            print("loss", torch.isnan(loss))

        if not return_interpolated:
            return loss

        return loss, intr_input


def grad(x):
    # x.shape : n, c, h, w
    diff_x = x[..., 1:, 1:] - x[..., 1:, :-1]
    diff_y = x[..., 1:, 1:] - x[..., :-1, 1:]
    mag = torch.abs(diff_x) + torch.abs(diff_y)
    # angle_ratio
    angle = torch.atan(diff_y / (diff_x + 1e-10))
    return mag, angle


def grad_mask(mask):
    return mask[..., 1:, 1:] & mask[..., 1:, :-1] & mask[..., :-1, 1:]

def sobel_grad_mask(mask):
    """
    Given a binary mask of shape [B, 1, H, W], returns a mask of shape [B, 1, H-2, W-2]
    where each output pixel is True only if its entire 3x3 neighborhood in the input mask is True.
    """
    return (mask[..., 1:-1, 1:-1] &  # center
            mask[..., :-2, 1:-1] &   # top center
            mask[..., 2:, 1:-1] &    # bottom center
            mask[..., 1:-1, :-2] &   # center left
            mask[..., 1:-1, 2:] &    # center right
            mask[..., :-2, :-2] &    # top left
            mask[..., :-2, 2:] &     # top right
            mask[..., 2:, :-2] &     # bottom left
            mask[..., 2:, 2:])       # bottom right

class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))
        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)
        
        # Freeze parameters.
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x) 
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))
        return out

# Combined loss: gradient loss and surface normal loss.
# Combined loss: gradient loss and surface normal loss.
import matplotlib.pyplot as plt
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
        axes[idx][0].set_title("Source1")
        axes[idx][0].axis('off')
        plt.colorbar(im1, ax=axes[idx][0], fraction=0.046, pad=0.04)
    
        # Display source2 image
        im2 = axes[idx][1].imshow(images2[idx])
        axes[idx][1].set_title("Source2")
        axes[idx][1].axis('off')
        plt.colorbar(im2, ax=axes[idx][1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


class SobelGradAndNormalLoss(nn.Module):
    """
    Computes two losses:
      1. Gradient Loss: L1 loss between the Sobel gradients (x & y) of input and target.
      2. Surface Normal Loss: 
         - Normals are computed as:
              depth_normal  = cat( -grad_gt_x, -grad_gt_y, ones )
              output_normal = cat( -grad_pred_x, -grad_pred_y, ones )
         - Then the cosine similarity is computed per pixel and the loss is
              mean( |1 - cosine_similarity| )
    
    The provided mask (shape [B,1,H,W]) is first converted to a valid-region mask via grad_mask,
    and then applied (after indexing the inner region [1:,1:]) to both losses.
    """
    def __init__(self):
        super(SobelGradAndNormalLoss, self).__init__()
        self.name = 'SobelGradAndNormal'
        self.sobel = Sobel().cuda()  # Sobel operator for gradient computation.
        self.cos = nn.CosineSimilarity(dim=1, eps=0)

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        # Assume extract_key and KEY_OUTPUT are defined elsewhere.
        input = extract_key(input, KEY_OUTPUT)
        
        # If the spatial dimensions don't match, interpolate input to match target.
        if input.shape[-1] != target.shape[-1]:
            input = F.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        # if mask is not None:
        #     target = target.clone()
        #     target[mask] = 1.0 / target[mask]
            # target[~mask] = 0
        
        # Compute Sobel gradients for input and target.
        grad_pred = self.sobel(input)   # Shape: [B, 2, H, W]
        grad_gt   = self.sobel(target)    # Shape: [B, 2, H, W]
        # show_images_two_sources(grad_pred[:, 0, :, :].unsqueeze(1), grad_gt[:, 0, :, :].unsqueeze(1))
        # Compute the valid-region mask.
        if mask is not None:
            mask_g = sobel_grad_mask(mask)  # Shape: [B, 1, H-1, W-1]
        else:
            B, _, H, W = input.shape
            mask_g = torch.ones(B, 1, H-1, W-1, dtype=torch.bool, device=input.device)
        
        # --- Gradient Loss ---
        # We apply the mask to the inner region [1:,1:] of the gradients.
        grad_pred_x = grad_pred[:, 0, 1:-1, 1:-1]
        grad_gt_x   = grad_gt[:, 0, 1:-1, 1:-1]
        grad_pred_y = grad_pred[:, 1, 1:-1, 1:-1]
        grad_gt_y   = grad_gt[:, 1, 1:-1, 1:-1]
        
        grad_loss = F.l1_loss(grad_pred_x[mask_g.squeeze(1)], grad_gt_x[mask_g.squeeze(1)]) + \
                    F.l1_loss(grad_pred_y[mask_g.squeeze(1)], grad_gt_y[mask_g.squeeze(1)])
        
        # --- Surface Normal Loss ---
        # Compute normals using the same inner region without an extra cropping step.
        # Create a ones tensor for the z-component.
        # ones = torch.ones_like(grad_pred_x)
        # depth_normal  = torch.cat((-grad_gt_x.unsqueeze(1), -grad_gt_y.unsqueeze(1), ones.unsqueeze(1)), dim=1)
        # output_normal = torch.cat((-grad_pred_x.unsqueeze(1), -grad_pred_y.unsqueeze(1), ones.unsqueeze(1)), dim=1)
        
        # # Compute cosine similarity per pixel; result has shape [B, H-1, W-1].
        # cos_sim = self.cos(output_normal, depth_normal)
        # loss_normal = torch.abs(1 - cos_sim)[mask_g.squeeze(1)].mean()
        
        if not return_interpolated:
            return grad_loss#, loss_normal*10
        return grad_loss, intr_input#, loss_normal*10, intr_input


class GradL1Loss(nn.Module):
    """Gradient loss"""
    def __init__(self):
        super(GradL1Loss, self).__init__()
        self.name = 'GradL1'

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        grad_gt = grad(target)
        grad_pred = grad(input)
        mask_g = grad_mask(mask)

        loss = nn.functional.l1_loss(grad_pred[0][mask_g], grad_gt[0][mask_g])
        loss = loss + \
            nn.functional.l1_loss(grad_pred[1][mask_g], grad_gt[1][mask_g])
        if not return_interpolated:
            return loss
        return loss, intr_input
    
class MultiScaleInverseGradL1Loss(nn.Module):
    """Multi-scale Gradient L1 loss using inverse ground truth depth."""

    def __init__(self, num_scales=1):
        super(MultiScaleInverseGradL1Loss, self).__init__()
        self.name = 'MultiScaleInverseGradL1'
        self.num_scales = num_scales

    def forward(self, input, target, mask=None, interpolate=True):
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
           

        if mask is not None:
            target = target.clone()
            target[mask] = 1.0 / target[mask]

        loss = 0.0
        pred, gt, mask_scale = input, target, mask

        for k in range(self.num_scales):
            grad_gt = grad(gt)
            grad_pred = grad(pred)

            if mask_scale is not None:
                mask_g = grad_mask(mask_scale)
                valid = mask_g.sum()
                if valid > 0:
                    loss += F.l1_loss(grad_pred[0][mask_g], grad_gt[0][mask_g])
                    # print(loss)
                    # loss += F.l1_loss(grad_pred[1][mask_g], grad_gt[1][mask_g])
                    # print(loss)
            else:
                loss += F.l1_loss(grad_pred[0], grad_gt[0])
                # loss += F.l1_loss(grad_pred[1], grad_gt[1])

            if k < self.num_scales - 1:
                pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
                gt = F.avg_pool2d(gt, kernel_size=2, stride=2)
                if mask_scale is not None:
                    mask_scale = F.max_pool2d(mask_scale.float(), kernel_size=2, stride=2).bool()

        loss /= self.num_scales
        # breakpoint()
        return loss*100
        


class OrdinalRegressionLoss(object):

    def __init__(self, ord_num, beta, discretization="SID"):
        self.ord_num = ord_num
        self.beta = beta
        self.discretization = discretization

    def _create_ord_label(self, gt):
        N,one, H, W = gt.shape
        # print("gt shape:", gt.shape)

        ord_c0 = torch.ones(N, self.ord_num, H, W).to(gt.device)
        if self.discretization == "SID":
            label = self.ord_num * torch.log(gt) / np.log(self.beta)
        else:
            label = self.ord_num * (gt - 1.0) / (self.beta - 1.0)
        label = label.long()
        mask = torch.linspace(0, self.ord_num - 1, self.ord_num, requires_grad=False) \
            .view(1, self.ord_num, 1, 1).to(gt.device)
        mask = mask.repeat(N, 1, H, W).contiguous().long()
        mask = (mask > label)
        ord_c0[mask] = 0
        ord_c1 = 1 - ord_c0
        # implementation according to the paper.
        # ord_label = torch.ones(N, self.ord_num * 2, H, W).to(gt.device)
        # ord_label[:, 0::2, :, :] = ord_c0
        # ord_label[:, 1::2, :, :] = ord_c1
        # reimplementation for fast speed.
        ord_label = torch.cat((ord_c0, ord_c1), dim=1)
        return ord_label, mask

    def __call__(self, prob, gt):
        """
        :param prob: ordinal regression probability, N x 2*Ord Num x H x W, torch.Tensor
        :param gt: depth ground truth, NXHxW, torch.Tensor
        :return: loss: loss value, torch.float
        """
        # N, C, H, W = prob.shape
        valid_mask = gt > 0.
        ord_label, mask = self._create_ord_label(gt)
        # print("prob shape: {}, ord label shape: {}".format(prob.shape, ord_label.shape))
        entropy = -prob * ord_label
        loss = torch.sum(entropy, dim=1)[valid_mask.squeeze(1)]
        return loss.mean()


class DiscreteNLLLoss(nn.Module):
    """Cross entropy loss"""
    def __init__(self, min_depth=1e-3, max_depth=10, depth_bins=64):
        super(DiscreteNLLLoss, self).__init__()
        self.name = 'CrossEntropy'
        self.ignore_index = -(depth_bins + 1)
        # self._loss_func = nn.NLLLoss(ignore_index=self.ignore_index)
        self._loss_func = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_bins = depth_bins
        self.alpha = 1
        self.zeta = 1 - min_depth
        self.beta = max_depth + self.zeta

    def quantize_depth(self, depth):
        # depth : N1HW
        # output : NCHW

        # Quantize depth log-uniformly on [1, self.beta] into self.depth_bins bins
        depth = torch.log(depth / self.alpha) / np.log(self.beta / self.alpha)
        depth = depth * (self.depth_bins - 1)
        depth = torch.round(depth) 
        depth = depth.long()
        return depth
        

    
    def _dequantize_depth(self, depth):
        """
        Inverse of quantization
        depth : NCHW -> N1HW
        """
        # Get the center of the bin




    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        # assert torch.all(input <= 0), "Input should be negative"

        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        # assert torch.all(input)<=1)
        if target.ndim == 3:
            target = target.unsqueeze(1)

        target = self.quantize_depth(target)
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            # Set the mask to ignore_index
            mask = mask.long()
            input = input * mask + (1 - mask) * self.ignore_index
            target = target * mask + (1 - mask) * self.ignore_index

        

        input = input.flatten(2)  # N, nbins, H*W
        target = target.flatten(1)  # N, H*W
        loss = self._loss_func(input, target)

        if not return_interpolated:
            return loss
        return loss, intr_input
    



def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1
class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "SSILoss"

    def forward(self, prediction, target, mask, interpolate=True, return_interpolated=False):
        
        if prediction.shape[-1] != target.shape[-1] and interpolate:
            prediction = nn.functional.interpolate(prediction, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = prediction
        else:
            intr_input = prediction


        prediction, target, mask = prediction.squeeze(), target.squeeze(), mask.squeeze()
        assert prediction.shape == target.shape, f"Shape mismatch: Expected same shape but got {prediction.shape} and {target.shape}."

        scale, shift = compute_scale_and_shift(prediction, target, mask)

        scaled_prediction = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        loss = nn.functional.l1_loss(scaled_prediction[mask], target[mask])
        if not return_interpolated:
            return loss
        return loss, intr_input




if __name__ == '__main__':
    # Tests for DiscreteNLLLoss
    celoss = DiscreteNLLLoss()
    print(celoss(torch.rand(4, 64, 26, 32)*10, torch.rand(4, 1, 26, 32)*10, ))

    d = torch.Tensor([6.59, 3.8, 10.0])
    print(celoss.dequantize_depth(celoss.quantize_depth(d)))
