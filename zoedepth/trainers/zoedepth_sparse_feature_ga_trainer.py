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

from zoedepth.trainers.loss import GradL1Loss, SILogLoss, RMSELoss
from zoedepth.utils.config import DATASETS_CONFIG
from zoedepth.utils.misc import compute_metrics
from zoedepth.data.preprocess import get_black_border

from .base_trainer import BaseTrainer
from torchvision import transforms
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

class Trainer(BaseTrainer):
    def __init__(self, config, model, train_loader, test_loader=None, device=None):
        super().__init__(config, model, train_loader,
                         test_loader=test_loader, device=device)
        self.device = device
        self.silog_loss = SILogLoss()
        self.grad_loss = GradL1Loss()
        self.rmse_loss = RMSELoss()
        self.scaler = amp.GradScaler(enabled=self.config.use_amp)

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

        losses = {}


        rgb_image = batch['image'][0].permute(1, 2, 0).cpu().numpy()  # Convert to HWC for visualization
        depth_map = batch['depth'][0].squeeze().cpu().numpy()  # Squeeze to remove (1, H, W) -> (H, W)
        feature_map = batch['sparse_map'][0].cpu().numpy()  # Feature map (2 channels)

        # Print shapes
        # print(batch['image_path'][0])
        # print(f"RGB Image shape: {rgb_image.shape}")          # Should be (H, W, 3)
        # print(f"Depth Map shape: {depth_map.shape}")          # Should be (H, W)
        # print(f"Feature Map shape: {feature_map.shape}")      # Should be (2, H, W)

        # Visualize the data
        # fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # Adjust figsize to control overall size
        
        # # RGB Image
        # axs[0].imshow(rgb_image)
        # axs[0].set_title('RGB Image')
        # axs[0].axis('off')
        # axs[0].set_aspect('equal')  # Or 'equal' to preserve natural aspect ratio

        # # Depth Map
        # im = axs[1].imshow(depth_map, cmap='plasma')
        # axs[1].set_title('Depth Map')
        # axs[1].axis('off')
        # axs[1].set_aspect('equal')  # Or 'equal'
        # fig.colorbar(im, ax=axs[1])

        # # Feature Map Channel 1
        # im = axs[2].imshow(feature_map[0], cmap='plasma')
        # axs[2].set_title('Feature Map - Channel 1')
        # axs[2].axis('off')
        # axs[2].set_aspect('equal')  # Or 'equal'
        # fig.colorbar(im, ax=axs[2])

        # # Feature Map Channel 2
        # im = axs[3].imshow(feature_map[1], cmap='plasma')
        # axs[3].set_title('Feature Map - Channel 2')
        # axs[3].axis('off')
        # axs[3].set_aspect('equal')  # Or 'equal'
        # fig.colorbar(im, ax=axs[3])

        # plt.show()
    

        with amp.autocast(enabled=self.config.use_amp):

            if self.config.prior_channels > 0:
                # print(images[0].shape)
                # print(depths_gt[0].shape)
                
                output = self.model(images, sparse_feature = sparse_features, input_height = self.config.input_height, input_width = self.config.input_width)
            else:
                output = self.model(images)
            pred_depths = output['metric_depth']
            # print(pred_depths[0].shape)
            l_si, pred = self.silog_loss(
                pred_depths, depths_gt, mask=mask, interpolate=True, return_interpolated=True)
            
            l_rmse= self.rmse_loss(
                pred_depths, depths_gt, mask=mask, interpolate=True)

            loss = self.config.w_si * l_si + self.config.w_rmse * l_rmse
            
            losses[self.silog_loss.name] = l_si
            losses[self.rmse_loss.name] = l_rmse

            if self.config.w_grad > 0:
                l_grad = self.grad_loss(pred, depths_gt, mask=mask)
                loss = loss + self.config.w_grad * l_grad
                losses[self.grad_loss.name] = l_grad
            else:
                l_grad = torch.Tensor([0])

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
            pred_depths = m(x, sparse_feature = sparse_feature, input_height = self.config.input_height, input_width = self.config.input_width)['metric_depth']
        return pred_depths

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
        if dataset == 'nyu' or dataset == 'nyu_sparse_feature':
            if self.config.prior_channels > 0:
                pred_depths = self.crop_aware_infer(images, sparse_feature = sparse_feature)
            else:
                pred_depths = self.crop_aware_infer(images)
        else:
            if self.config.prior_channels > 0:
                pred_depths = self.eval_infer(images, sparse_feature = sparse_feature)
            else:
                pred_depths = self.eval_infer(images)
        pred_depths = pred_depths.squeeze().unsqueeze(0).unsqueeze(0)

        with amp.autocast(enabled=self.config.use_amp):
            l_depth = self.silog_loss(
                pred_depths, depths_gt, mask=mask.to(torch.bool), interpolate=True)
            
            l_rmse = self.rmse_loss(
                pred_depths, depths_gt, mask=mask.to(torch.bool), interpolate=True)

        # metrics = compute_metrics(depths_gt, pred_depths, **self.config)
        metrics = compute_metrics(depths_gt, pred_depths, config = self.config)
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
