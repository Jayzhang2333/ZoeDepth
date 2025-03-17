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

"""Miscellaneous utility functions."""

from scipy import ndimage

import base64
import math
import re
from io import BytesIO

import matplotlib
import matplotlib.cm
import numpy as np
import requests
import torch
import torch.distributed as dist
import torch.nn
import torch.nn as nn
import torch.utils.data.distributed
from PIL import Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

class NormalAverage:
    """A class that computes the normal (simple) average."""
    def __init__(self):
        self.total = 0
        self.count = 0

    def append(self, value):
        self.total += value
        self.count += 1

    def get_value(self):
        if self.count == 0:
            return 0  # Return 0 or None if no values have been added
        return self.total / self.count


class NormalAverageDict:
    """A dictionary of normal averages."""
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if new_dict is None:
            return

        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = NormalAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        if self._dict is None:
            return None
        return {key: value.get_value() for key, value in self._dict.items()}


class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


def denormalize(x):
    """Reverses the imagenet normalization applied to the input.

    Args:
        x (torch.Tensor - shape(N,3,H,W)): input tensor

    Returns:
        torch.Tensor - shape(N,3,H,W): Denormalized input
    """
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    return x * std + mean


class RunningAverageDict:
    """A dictionary of running averages."""
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if new_dict is None:
            return

        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        if self._dict is None:
            return None
        return {key: value.get_value() for key, value in self._dict.items()}


def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img


def count_parameters(model, include_all=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or include_all)

# import numpy as np

def rmse_silog(predicted, ground_truth):
    """
    Computes the RMSE_silog as described in the provided formula.

    Parameters:
    - predicted: np.array, predicted depth values (N,)
    - ground_truth: np.array, ground truth depth values (N,)

    Returns:
    - rmse_silog: scalar, RMSE_silog error
    """
    # Ensure no division by zero or log of zero
    epsilon = 1e-8
    predicted = np.maximum(predicted, epsilon)
    ground_truth = np.maximum(ground_truth, epsilon)

    # Compute log values
    log_predicted = np.log(predicted)
    log_ground_truth = np.log(ground_truth)

    # Compute alpha
    N = len(predicted)
    alpha = np.sum(log_ground_truth - log_predicted) / N

    # Compute RMSE_silog
    rmse_silog = np.sqrt(np.mean((log_predicted - log_ground_truth + alpha) ** 2))

    return rmse_silog

# def compute_errors(gt, pred):
#     """Compute metrics for 'pred' compared to 'gt'

#     Args:
#         gt (numpy.ndarray): Ground truth values
#         pred (numpy.ndarray): Predicted values

#         gt.shape should be equal to pred.shape

#     Returns:
#         dict: Dictionary containing the following metrics:
#             'a1': Delta1 accuracy: Fraction of pixels that are within a scale factor of 1.25
#             'a2': Delta2 accuracy: Fraction of pixels that are within a scale factor of 1.25^2
#             'a3': Delta3 accuracy: Fraction of pixels that are within a scale factor of 1.25^3
#             'abs_rel': Absolute relative error
#             'rmse': Root mean squared error
#             'log_10': Absolute log10 error
#             'sq_rel': Squared relative error
#             'rmse_log': Root mean squared error on the log scale
#             'silog': Scale invariant log error
#     """
#     if gt.size == 0:
#         return None
#         # raise ValueError("Ground truth (gt) is empty, cannot compute errors.")
#     thresh = np.maximum((gt / pred), (pred / gt))
#     a1 = (thresh < 1.25).mean()
#     a2 = (thresh < 1.25 ** 2).mean()
#     a3 = (thresh < 1.25 ** 3).mean()

#     abs_rel = np.mean(np.abs(gt - pred) / gt)
#     sq_rel = np.mean(((gt - pred) ** 2) / gt)

#     rmse = (gt - pred) ** 2
#     rmse = np.sqrt(rmse.mean())

#     rmse_log = (np.log(gt) - np.log(pred)) ** 2
#     rmse_log = np.sqrt(rmse_log.mean())

#     # err = np.log(pred) - np.log(gt)
#     # silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

#     silog = rmse_silog(pred, gt)



#     log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
#     return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
#                 silog=silog, sq_rel=sq_rel)
def compute_errors(gt, pred ):
    """Compute metrics for 'pred' compared to 'gt'

    Args:
        gt (numpy.ndarray): Ground truth values.
        pred (numpy.ndarray): Predicted values.
            Both should have the same shape.

    Returns:
        dict: Dictionary containing the following metrics:
            'a1': Delta1 accuracy: Fraction of pixels that are within a scale factor of 1.25.
            'a2': Delta2 accuracy: Fraction of pixels that are within a scale factor of 1.25^2.
            'a3': Delta3 accuracy: Fraction of pixels that are within a scale factor of 1.25^3.
            'abs_rel': Absolute relative error.
            'sq_rel': Squared relative error.
            'rmse': Root mean squared error.
            'rmse_log': Root mean squared error on the log scale.
            'log_10': Mean absolute log10 error.
            'silog': Scale invariant log error.
            'mae': Mean absolute error.
            'iRMSE': Inverse RMSE (computed on inverse depths).
            'iMAE': Inverse MAE (computed on inverse depths).
            'iAbsRel': Inverse absolute relative error (computed on inverse depths).
    """
    if gt.size == 0:
        return None
    
    # if sparse_gt.size == 0:
    #     return None

    # Threshold-based accuracy metrics
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    # Standard error metrics
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt((np.log(gt) - np.log(pred)) ** 2).mean()  # Ensuring computation consistency

    # Alternatively, you might have a function for silog; here we assume it exists.
    silog = rmse_silog(pred, gt)

    log_10 = np.abs(np.log10(gt) - np.log10(pred)).mean()

    # Additional metrics
    mae = np.mean(np.abs(gt - pred))

    # Compute inverse depths. Assuming no zero values in gt or pred.
    inv_gt = 1.0 / gt
    inv_pred = 1.0 / pred

    i_rmse = np.sqrt(np.mean((inv_gt - inv_pred) ** 2))
    i_mae = np.mean(np.abs(inv_gt - inv_pred))
    i_abs_rel = np.mean(np.abs(inv_gt - inv_pred) / inv_gt)

    # sparse_abs_rel = np.mean(np.abs(sparse_gt - sparse_pred) / sparse_gt)
    # sparse_rmse = np.sqrt(((sparse_gt - sparse_pred) ** 2).mean())
    # sparse_mae = np.mean(np.abs(sparse_gt - sparse_pred))
    # inv_sparse_gt = 1.0 / sparse_gt
    # inv_sparse_pred = 1.0 / sparse_pred
    # sparse_i_rmse = np.sqrt(np.mean((inv_sparse_gt - inv_sparse_pred) ** 2))
    # sparse_i_mae = np.mean(np.abs(inv_sparse_gt - inv_sparse_pred))
    # sparse_i_abs_rel = np.mean(np.abs(inv_sparse_gt - inv_sparse_pred) / inv_sparse_gt)

    return dict(
        a1=a1,
        a2=a2,
        a3=a3,
        abs_rel=abs_rel,
        rmse=rmse,
        log_10=log_10,
        rmse_log=rmse_log,
        silog=silog,
        sq_rel=sq_rel,
        mae=mae,
        iRMSE=i_rmse,
        iMAE=i_mae,
        iAbsRel=i_abs_rel,
        # sparse_abs_rel = sparse_abs_rel,
        # sparse_rmse = sparse_rmse,
        # sparse_mae=sparse_mae,
        # sparse_iRMSE=sparse_i_rmse,
        # sparse_iMAE=sparse_i_mae,
        # sparse_iAbsRel=sparse_i_abs_rel,
    )


def compute_metrics(gt, pred, interpolate=True, garg_crop=False, eigen_crop=True, dataset='nyu', min_depth_eval=0.1, max_depth_eval=10, **kwargs):
    """Compute metrics of predicted depth maps. Applies cropping and masking as necessary or specified via arguments. Refer to compute_errors for more details on metrics.
    """
    if 'config' in kwargs:
        config = kwargs['config']
        garg_crop = config.garg_crop
        eigen_crop = config.eigen_crop
        min_depth_eval = config.min_depth_eval
        max_depth_eval = config.max_depth_eval

    if gt.shape[-2:] != pred.shape[-2:] and interpolate:
        pred = nn.functional.interpolate(
            pred, gt.shape[-2:], mode='bilinear', align_corners=True)

    pred = pred.squeeze().cpu().numpy()
    pred[pred < min_depth_eval] = min_depth_eval
    pred[pred > max_depth_eval] = max_depth_eval
    pred[np.isinf(pred)] = max_depth_eval
    pred[np.isnan(pred)] = min_depth_eval

    gt_depth = gt.squeeze().cpu().numpy()

    # if sparse_mask is not None:
    #     sparse_mask = sparse_mask.squeeze().cpu().numpy()
    #     new_dim = (480, 640)
    #     sparse_mask = rescale_mask(sparse_mask, new_dim)
    

    valid_mask = np.logical_and(
        gt_depth > min_depth_eval, gt_depth < max_depth_eval)

    # print(config.dataset)
    if garg_crop or eigen_crop:
        gt_height, gt_width = gt_depth.shape
        eval_mask = np.zeros(valid_mask.shape)

        if garg_crop:
            eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                      int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
        
        elif eigen_crop:
            # print("-"*10, " EIGEN CROP ", "-"*10)
            if config.dataset == 'kitti':
                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                          int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                
            elif config.dataset == 'lizard_sparse_feature' or config.dataset == 'flsea_sparse_feature':
                # print("yes")
                eval_mask[:,int(0 * gt_width):int(0.9 * gt_width)] = 1
            else:
                # print("no")
                # assert gt_depth.shape == (480, 640), "Error: Eigen crop is currently only valid for (480, 640) images"
                eval_mask[45:471, 41:601] = 1
        else:
            eval_mask = np.ones(valid_mask.shape)

    else:
        eval_mask = np.ones(valid_mask.shape)
    valid_mask = np.logical_and(valid_mask, eval_mask)
    # sparse_mask = np.logical_and(valid_mask, sparse_mask)


    return compute_errors(gt_depth[valid_mask], pred[valid_mask])

def rescale_mask(old_mask, new_shape):
    old_rows, old_cols = old_mask.shape
    new_rows, new_cols = new_shape
    new_mask = np.zeros(new_shape, dtype=old_mask.dtype)
    
    row_scale = new_rows / old_rows
    col_scale = new_cols / old_cols

    # Loop through all valid (i.e. == 1) points in the old mask.
    for i in range(old_rows):
        for j in range(old_cols):
            if old_mask[i, j] == 1:
                # Compute new indices using scaling factors.
                new_i = int(np.floor(i * row_scale))
                new_j = int(np.floor(j * col_scale))
                
                # Ensure the new indices are within bounds.
                new_i = min(new_i, new_rows - 1)
                new_j = min(new_j, new_cols - 1)
                
                new_mask[new_i, new_j] = 1
                
    return new_mask
    
def evaluation_on_ranges(gt, pred, interpolate=True, dataset=None, evaluation_range = [18.0, 18, 18, 18, 18]):
    """Compute metrics of predicted depth maps. Applies cropping and masking as necessary or specified via arguments. Refer to compute_errors for more details on metrics.
    """

    if gt.shape[-2:] != pred.shape[-2:] and interpolate:
        pred = nn.functional.interpolate(
            pred, gt.shape[-2:], mode='bilinear', align_corners=True)

    gt_depth = gt.squeeze().cpu().numpy()
    
    # sparse_mask = sparse_mask.squeeze().cpu().numpy()
    # new_dim = (608, 968)
    # sparse_mask = rescale_mask(sparse_mask, new_dim)
    
    
    
    # print(np.shape(sparse_mask))
    pred = pred.squeeze().cpu().numpy()
    min_depth_eval = 0.1
    result_dicts = []
    for i in evaluation_range:
        pred_copy = pred.copy()
        pred_copy[pred_copy < min_depth_eval] = min_depth_eval
        pred_copy[pred_copy > i] = i
        pred_copy[np.isinf(pred_copy)] = i
        pred_copy[np.isnan(pred_copy)] = min_depth_eval

        

        valid_mask = np.logical_and(
            gt_depth > min_depth_eval, gt_depth < i)

        # print(config.dataset)

        eval_mask = np.ones(valid_mask.shape)
        valid_mask = np.logical_and(valid_mask, eval_mask)

        # non_zero_count = np.count_nonzero(valid_mask)
        # print("Number of non-zero values in valid_mask:", non_zero_count)
        
        # sparse_mask_temp = np.logical_and(valid_mask, sparse_mask)
        # non_zero_count = np.count_nonzero(sparse_mask_temp)
        # print("Number of non-zero values in sparse_mask:", non_zero_count)
        
        # fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # # Plot the first mask.
        # im1 = axes[0].imshow(valid_mask, cmap='gray', interpolation='nearest')
        # axes[0].set_title('Mask 1')
        # axes[0].set_xlabel('Columns')
        # axes[0].set_ylabel('Rows')
        # fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # # Plot the second mask.
        # im2 = axes[1].imshow(sparse_mask_temp, cmap='gray', interpolation='nearest')
        # axes[1].set_title('Mask 2')
        # axes[1].set_xlabel('Columns')
        # axes[1].set_ylabel('Rows')
        # fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # plt.tight_layout()
        # plt.show()

        dict = compute_errors(gt_depth[valid_mask], pred[valid_mask])
        result_dicts.append(dict)
    return result_dicts
    


#################################### Model uilts ################################################


def parallelize(config, model, find_unused_parameters=True):

    if config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        model = model.cuda(config.gpu)

    config.multigpu = False
    if config.distributed:
        # Use DDP
        config.multigpu = True
        config.rank = config.rank * config.ngpus_per_node + config.gpu
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                world_size=config.world_size, rank=config.rank)
        config.batch_size = int(config.batch_size / config.ngpus_per_node)
        # config.batch_size = 8
        config.workers = int(
            (config.num_workers + config.ngpus_per_node - 1) / config.ngpus_per_node)
        print("Device", config.gpu, "Rank",  config.rank, "batch size",
              config.batch_size, "Workers", config.workers)
        torch.cuda.set_device(config.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(config.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu], output_device=config.gpu,
                                                          find_unused_parameters=find_unused_parameters)

    elif config.gpu is None:
        # Use DP
        config.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    return model


#################################################################################################


#####################################################################################################


class colors:
    '''Colors class:
    Reset all colors with colors.reset
    Two subclasses fg for foreground and bg for background.
    Use as colors.subclass.colorname.
    i.e. colors.fg.red or colors.bg.green
    Also, the generic bold, disable, underline, reverse, strikethrough,
    and invisible work with the main class
    i.e. colors.bold
    '''
    reset = '\033[0m'
    bold = '\033[01m'
    disable = '\033[02m'
    underline = '\033[04m'
    reverse = '\033[07m'
    strikethrough = '\033[09m'
    invisible = '\033[08m'

    class fg:
        black = '\033[30m'
        red = '\033[31m'
        green = '\033[32m'
        orange = '\033[33m'
        blue = '\033[34m'
        purple = '\033[35m'
        cyan = '\033[36m'
        lightgrey = '\033[37m'
        darkgrey = '\033[90m'
        lightred = '\033[91m'
        lightgreen = '\033[92m'
        yellow = '\033[93m'
        lightblue = '\033[94m'
        pink = '\033[95m'
        lightcyan = '\033[96m'

    class bg:
        black = '\033[40m'
        red = '\033[41m'
        green = '\033[42m'
        orange = '\033[43m'
        blue = '\033[44m'
        purple = '\033[45m'
        cyan = '\033[46m'
        lightgrey = '\033[47m'


def printc(text, color):
    print(f"{color}{text}{colors.reset}")

############################################

def get_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img

def url_to_torch(url, size=(384, 384)):
    img = get_image_from_url(url)
    img = img.resize(size, Image.ANTIALIAS)
    img = torch.from_numpy(np.asarray(img)).float()
    img = img.permute(2, 0, 1)
    img.div_(255)
    return img

def pil_to_batched_tensor(img):
    return ToTensor()(img).unsqueeze(0)

def save_raw_16bit(depth, fpath="raw.png"):
    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().cpu().numpy()
    
    assert isinstance(depth, np.ndarray), "Depth must be a torch tensor or numpy array"
    assert depth.ndim == 2, "Depth must be 2D"
    depth = depth * 256  # scale for 16-bit png
    depth = depth.astype(np.uint16)
    depth = Image.fromarray(depth)
    depth.save(fpath)
    print("Saved raw depth to", fpath)