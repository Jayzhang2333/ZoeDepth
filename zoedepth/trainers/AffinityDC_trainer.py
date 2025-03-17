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
import torch.cuda.amp as amp
import torch.nn as nn

from zoedepth.trainers.loss import GradL1Loss, SILogLoss, RMSELoss, L1SmoothLoss, NegativeLogLikelihoodLoss, InvNegativeLogLikelihoodLoss, InvRMSELoss, InvSILogLoss
from zoedepth.utils.config import DATASETS_CONFIG
from zoedepth.utils.misc import compute_metrics
from zoedepth.data.preprocess import get_black_border

from .base_trainer import BaseTrainer
from torchvision import transforms
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

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


class Trainer(BaseTrainer):
    def __init__(self, config, model, train_loader, test_loader=None, device=None):
        super().__init__(config, model, train_loader,
                         test_loader=test_loader, device=device)
        self.device = device
        self.silog_loss = InvSILogLoss()
        # self.grad_loss = GradL1Loss()
        self.rmse_loss = InvRMSELoss()
        self.NegLogLikelihood_loss = InvNegativeLogLikelihoodLoss()
        self.scaler = amp.GradScaler(enabled=self.config.use_amp)
        self.dataset = config.dataset
        # if self.dataset == 'tartanair':
        #     self.l1smooth_loss = L1SmoothLoss()

    def train_on_batch(self, batch, train_step):
        """
        Expects a batch of images and depth as input
        batch["image"].shape : batch_size, c, h, w
        batch["depth"].shape : batch_size, 1, h, w
        """

        images, depths_gt,  = batch['image'].to(
            self.device), batch['depth'].to(self.device)
        if self.config.prior_channels > 0:
            sparse_features = batch['sparse_map'].to(self.device).float() 
        dataset = batch['dataset'][0]

        b, c, h, w = images.size()
        mask = batch["mask"].to(self.device).to(torch.bool)
        # show_images_three_sources(images, depths_gt, sparse_features)
        losses = {}
    
        # print(f"image shape is {images.shape}")
        # print(f"gt shape is {depths_gt.shape}")
        # print(f"sparse map shape is {sparse_features.shape}")
        
        with amp.autocast(enabled=self.config.use_amp):

           
            output = self.model(images, prompt_depth = sparse_features)

            pred_depths = output['metric_depth']
            pred = nn.functional.interpolate(pred_depths, depths_gt.shape[-2:], mode='bilinear', align_corners=True)
            pred_conf = output['confidence']
            pred_inverse_depths = output['inverse_depth']
            # show_images_three_sources(images, pred_depths, sparse_features)
            # breakpoint()
            
            # intermedian_pred_depths = output['intermedian_pred']
            # print(pred_depths[0].shape)
            l_si = self.silog_loss(
                pred_inverse_depths, depths_gt, mask=mask, interpolate=True, return_interpolated=False)
            
            # l_si_intermedian = self.silog_loss(
            #     intermedian_pred_depths, depths_gt, mask=mask, interpolate=True,)

            l_nll = self.NegLogLikelihood_loss(pred_inverse_depths, pred_conf, depths_gt, mask=mask, interpolate=True)
            
            l_rmse= self.rmse_loss(
                pred_inverse_depths, depths_gt, mask=mask, interpolate=True)
            
            # if self.dataset == 'tartanair':
            #     l_l1smooth = self.l1smooth_loss(
            #     pred_depths, depths_gt, mask=mask, interpolate=True)
            
            # l_rmse_intermedian= self.rmse_loss(
            #     intermedian_pred_depths, depths_gt, mask=mask, interpolate=True)

            if self.dataset == 'tartanair':
                loss = self.config.w_nll*l_nll + self.config.w_si*l_si + self.config.w_rmse*l_rmse #+ self.config.w_l1smooth*l_l1smooth #+ self.config.w_si_intermedian*l_si_intermedian #+ self.config.w_rmse_intermedian*l_rmse_intermedian
            else:
                loss = self.config.w_nll*l_nll + self.config.w_si*l_si + self.config.w_rmse * l_rmse #+ self.config.w_si_intermedian * l_si_intermedian + self.config.w_rmse_intermedian * l_rmse_intermedian
            losses[self.silog_loss.name] = l_si
            losses[self.rmse_loss.name] = l_rmse
            losses['l_nll'] = l_nll

            # if self.dataset == 'tartanair':
            #     losses['l_l1smooth'] = l_l1smooth
            # losses['l_silog_intermedian'] = l_si_intermedian
            # losses['l_rmse_intermedian'] = l_rmse_intermedian

            # if self.config.w_grad > 0:
            #     l_grad = self.grad_loss(pred, depths_gt, mask=mask)
            #     loss = loss + self.config.w_grad * l_grad
            #     losses[self.grad_loss.name] = l_grad
            # else:
            #     l_grad = torch.Tensor([0])

        self.scaler.scale(loss).backward()

        if self.config.clip_grad > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.clip_grad)

        self.scaler.step(self.optimizer)

        # print(f'log image every: {self.config.log_images_every}')
        # print(f'iters_per_epoch {self.iters_per_epoch}')
        if self.should_log and (self.step % int(self.config.log_images_every * self.iters_per_epoch)) == 0:
            # -99 is treated as invalid depth in the log_images function and is colored grey.
            depths_gt[torch.logical_not(mask)] = -99
            # print(images[0, ...].shape)
            # print(depths_gt[0].shape[-2:])
            # print(pred_depths[0].shape)
            input_resized = nn.functional.interpolate(images[0, ...].unsqueeze(0), size=depths_gt[0].shape[-2:], mode='bilinear', align_corners=False)
            pred_resized = nn.functional.interpolate(pred[0].unsqueeze(0), size=depths_gt[0].shape[-2:], mode='bilinear', align_corners=False)
            
            self.log_images(rgb={"Input": input_resized}, depth={"GT": depths_gt[0], "PredictedMono": pred_resized}, prefix="Train",
                            min_depth=DATASETS_CONFIG[dataset]['min_depth'], max_depth=DATASETS_CONFIG[dataset]['max_depth'])
            # self.log_images(rgb={"Input": images[0, ...]}, depth={"GT": depths_gt[0], "PredictedMono": pred[0]}, prefix="Train",
            #                 min_depth=DATASETS_CONFIG[dataset]['min_depth'], max_depth=DATASETS_CONFIG[dataset]['max_depth'])


            if self.config.get("log_rel", False):
                self.log_images(
                    scalar_field={"RelPred": output["relative_depth"][0]}, prefix="TrainRel")

        self.scaler.update()
        self.optimizer.zero_grad()

        return losses
    
    @torch.no_grad()
    def eval_infer(self, x, sparse_feature = None):
        with amp.autocast(enabled=self.config.use_amp):
            m = self.model.module if self.config.multigpu else self.model
           
            output = m(x, prompt_depth = sparse_feature)
            pred_depths = output['metric_depth']
            pred_depths_inverse = output['inverse_depth']
        return pred_depths, pred_depths_inverse

    @torch.no_grad()
    def crop_aware_infer(self, x, sparse_feature = None):
        # if we are not avoiding the black border, we can just use the normal inference
        if not self.config.get("avoid_boundary", False):
            return self.eval_infer(x, sparse_feature)
        
        # otherwise, we need to crop the image to avoid the black border
        # For now, this may be a bit slow due to converting to numpy and back
        # We assume no normalization is done on the input image

        # THIS IS NOT MODIFIED FOR SPARSE FEATURE YET!!!!!!
        # get the black border
        assert x.shape[0] == 1, "Only batch size 1 is supported for now"
        x_pil = transforms.ToPILImage()(x[0].cpu())
        x_np = np.array(x_pil, dtype=np.uint8)
        black_border_params = get_black_border(x_np)
        top, bottom, left, right = black_border_params.top, black_border_params.bottom, black_border_params.left, black_border_params.right
        x_np_cropped = x_np[top:bottom, left:right, :]
        x_cropped = transforms.ToTensor()(Image.fromarray(x_np_cropped))

        # run inference on the cropped image
        pred_depths_cropped = self.eval_infer(x_cropped.unsqueeze(0).to(self.device))

        # resize the prediction to x_np_cropped's size
        pred_depths_cropped = nn.functional.interpolate(
            pred_depths_cropped, size=(x_np_cropped.shape[0], x_np_cropped.shape[1]), mode="bilinear", align_corners=False)
        

        # pad the prediction back to the original size
        pred_depths = torch.zeros((1, 1, x_np.shape[0], x_np.shape[1]), device=pred_depths_cropped.device, dtype=pred_depths_cropped.dtype)
        pred_depths[:, :, top:bottom, left:right] = pred_depths_cropped

        return pred_depths



    def validate_on_batch(self, batch, val_step):
        images = batch['image'].to(self.device)
        if self.config.prior_channels > 0:
            sparse_feature = batch['sparse_map'].to(self.device).float() 
        depths_gt = batch['depth'].to(self.device)
        dataset = batch['dataset'][0]
        mask = batch["mask"].to(self.device)
        
        if 'has_valid_depth' in batch:
            if not batch['has_valid_depth']:
                return None, None

        depths_gt = depths_gt.squeeze().unsqueeze(0).unsqueeze(0)
        mask = mask.squeeze().unsqueeze(0).unsqueeze(0)
        
        pred_depths, pred_depths_inverse = self.eval_infer(images, sparse_feature = sparse_feature)
            
        pred_depths = pred_depths.squeeze().unsqueeze(0).unsqueeze(0)
        pred_depths_inverse = pred_depths_inverse.squeeze().unsqueeze(0).unsqueeze(0)

        with amp.autocast(enabled=self.config.use_amp):
            l_depth = self.silog_loss(
                pred_depths_inverse, depths_gt, mask=mask.to(torch.bool), interpolate=True)
            
            l_rmse = self.rmse_loss(
                pred_depths_inverse, depths_gt, mask=mask.to(torch.bool), interpolate=True)

        valid_points_mask = (sparse_feature >= self.config.min_depth) & (sparse_feature <= self.config.max_depth)
        metrics = compute_metrics(depths_gt, pred_depths,sparse_mask=valid_points_mask, **self.config)
        losses = {f"{self.silog_loss.name}": l_depth.item()}
        losses[self.rmse_loss.name] = l_rmse

        if val_step == 1 and self.should_log:
            depths_gt[torch.logical_not(mask)] = -99
            # print(images[0].shape)
            # print(depths_gt[0].shape)
            # print(pred_depths[0].shape)

            input_resized = nn.functional.interpolate(images[0].unsqueeze(0), size=depths_gt[0].shape[-2:], mode='bilinear', align_corners=False)
            pred_resized = nn.functional.interpolate(pred_depths[0].unsqueeze(0), size=depths_gt[0].shape[-2:], mode='bilinear', align_corners=False)
            
            self.log_images(rgb={"Input": input_resized}, depth={"GT": depths_gt[0], "PredictedMono": pred_resized}, prefix="Test",
                            min_depth=DATASETS_CONFIG[dataset]['min_depth'], max_depth=DATASETS_CONFIG[dataset]['max_depth'])
            
            # self.log_images(rgb={"Input": images[0]}, depth={"GT": depths_gt[0], "PredictedMono": pred_depths[0]}, prefix="Test",
            #                 min_depth=DATASETS_CONFIG[dataset]['min_depth'], max_depth=DATASETS_CONFIG[dataset]['max_depth'])

        return metrics, losses
