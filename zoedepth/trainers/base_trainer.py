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

import os
import uuid
import warnings
from datetime import datetime as dt
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm

from zoedepth.utils.config import flatten
from zoedepth.utils.misc import RunningAverageDict, colorize, colors


def is_rank_zero(args):
    return args.rank == 0

# def print_learning_rates(model, optimizer):
#     mvit_lr = None
#     other_lr = None

#     for param_group in optimizer.param_groups:
#         lr = param_group['lr']
#         params = param_group['params']
        
#         # Check if this parameter group belongs to mViT
#         if any(p in model.ScaleMapLearner.mViT.parameters() for p in params):
#             mvit_lr = lr
#         else:
#             other_lr = lr
    
#     print(f"Learning Rate for mViT: {mvit_lr}")
#     print(f"Learning Rate for Other Modules: {other_lr}")


class BaseTrainer:
    def __init__(self, config, model, train_loader, test_loader=None, device=None):
        """ Base Trainer class for training a model."""
        
        self.config = config
        self.metric_criterion = "abs_rel"
        if device is None:
            device = torch.device(
                'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        self.model = model
        self.train_loader = train_loader
        
        self.test_loader = test_loader
        self.optimizer = self.init_optimizer()
        num_param_groups = len(self.optimizer.param_groups)
        print(f"Number of parameter groups in optimizer: {num_param_groups}")
        self.scheduler = self.init_scheduler()
        self.print_learning_rates()

    def print_learning_rates(self):
        """Prints learning rates for mViT and other parameters after initializing optimizer & scheduler."""
        m = self.model.module if self.config.multigpu else self.model
        if 'zoedepth' in self.config.name:
            mvit_params = set(m.ScaleMapLearner.mViT.parameters())

            print("\n=== Learning Rates After Initialization ===")
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = param_group['lr']
                params = param_group['params']

                if any(p in mvit_params for p in params):
                    print(f"Group {i}: Learning Rate for mViT = {lr}")
                else:
                    print(f"Group {i}: Learning Rate for Other Parameters = {lr}")
        else:
            print("\n=== Learning Rates After Initialization ===")
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = param_group['lr']
                group_name = param_group.get('name', f'Group {i}')  # uses 'name' if exists, else defaults to 'Group i'
                print(f"Group {i} (name: {group_name}): Learning Rate = {lr}")


    def resize_to_target(self, prediction, target):
        if prediction.shape[2:] != target.shape[-2:]:
            prediction = nn.functional.interpolate(
                prediction, size=target.shape[-2:], mode="bilinear", align_corners=True
            )
        return prediction

    def load_ckpt(self, checkpoint_dir="./checkpoints", ckpt_type="best"):
        import glob
        import os

        from zoedepth.models.model_io import load_wts

        if hasattr(self.config, "checkpoint"):
            checkpoint = self.config.checkpoint
        elif hasattr(self.config, "ckpt_pattern"):
            pattern = self.config.ckpt_pattern
            matches = glob.glob(os.path.join(
                checkpoint_dir, f"*{pattern}*{ckpt_type}*"))
            if not (len(matches) > 0):
                raise ValueError(f"No matches found for the pattern {pattern}")
            checkpoint = matches[0]
        else:
            return
        model = load_wts(self.model, checkpoint)
        # TODO : Resuming training is not properly supported in this repo. Implement loading / saving of optimizer and scheduler to support it.
        print("Loaded weights from {0}".format(checkpoint))
        warnings.warn(
            "Resuming training is not properly supported in this repo. Implement loading / saving of optimizer and scheduler to support it.")
        self.model = model

    # def init_optimizer(self):
    #     m = self.model.module if self.config.multigpu else self.model

    #     base_lr = self.config.lr  # Base learning rate
    #     mvit_lr = base_lr * 10  # Higher learning rate for mViT

    #     # Get all parameters of ScaleMapLearner
    #     scale_map_learner_params = dict(m.ScaleMapLearner.named_parameters())

    #     # Separate mViT parameters
    #     mvit_params = {k: v for k, v in scale_map_learner_params.items() if "mViT" in k}
    #     other_params = {k: v for k, v in scale_map_learner_params.items() if "mViT" not in k}

    #     param_groups = [
    #         {"params": list(mvit_params.values()), "lr": mvit_lr},  # Higher LR for mViT
    #         {"params": list(other_params.values()), "lr": base_lr},  # Base LR for other parameters
    #     ]

    #     return optim.AdamW(param_groups, lr=base_lr, weight_decay=self.config.wd)
    def init_optimizer(self):
        m = self.model.module if self.config.multigpu else self.model

        if self.config.same_lr:
            print("Using same LR")
            if hasattr(m, 'core'):
                m.core.unfreeze()
            params = self.model.parameters()
        else:
            print("Using diff LR")
            if not hasattr(m, 'get_lr_params'):
                raise NotImplementedError(
                    f"Model {m.__class__.__name__} does not implement get_lr_params. Please implement it or use the same LR for all parameters.")

            params = m.get_lr_params(self.config.lr)
            # lrs = [l['lr'] for l in params]
            # print(lrs)

        # since params is already a list of dict that conatins lr for each paramater group, the lr passing into AdamW here is to make sure 
        # parameters that is not included in params also has a learning rate
        return optim.AdamW(params, lr=self.config.lr, weight_decay=self.config.wd)

    def init_scheduler(self):
        lrs = [l['lr'] for l in self.optimizer.param_groups]
        
        return optim.lr_scheduler.OneCycleLR(self.optimizer, lrs, epochs=self.config.epochs, steps_per_epoch=len(self.train_loader),
                                             cycle_momentum=self.config.cycle_momentum,
                                             base_momentum=0.85, max_momentum=0.95, div_factor=self.config.div_factor, final_div_factor=self.config.final_div_factor, pct_start=self.config.pct_start, three_phase=self.config.three_phase)

    def train_on_batch(self, batch, train_step):
        raise NotImplementedError

    def validate_on_batch(self, batch, val_step):
        raise NotImplementedError

    def raise_if_nan(self, losses):
        for key, value in losses.items():
            if torch.isnan(value):
                raise ValueError(f"{key} is NaN, Stopping training")

    @property
    def iters_per_epoch(self):
        return len(self.train_loader)

    @property
    def total_iters(self):
        return self.config.epochs * self.iters_per_epoch

    def should_early_stop(self):
        if self.config.get('early_stop', False) and self.step > self.config.early_stop:
            return True

    def train(self):
        print(f"Training {self.config.name}")
        if self.config.uid is None:
            self.config.uid = str(uuid.uuid4()).split('-')[-1]
        run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-{self.config.uid}"
        self.config.run_id = run_id
        self.config.experiment_id = f"{self.config.name}{self.config.version_name}_{run_id}"
        self.should_write = ((not self.config.distributed)
                             or self.config.rank == 0)
        self.should_log = self.should_write  # and logging
        if self.should_log:
            tags = self.config.tags.split(
                ',') if self.config.tags != '' else None
            wandb.init(project=self.config.project, name=self.config.experiment_id, config=flatten(self.config), dir=self.config.root,
                       tags=tags, notes=self.config.notes, settings=wandb.Settings(start_method="fork"))

        self.model.train()
        self.step = 0
        best_loss = np.inf
        validate_every = int(self.config.validate_every * self.iters_per_epoch)


        if self.config.prefetch:

            for i, batch in tqdm(enumerate(self.train_loader), desc=f"Prefetching...",
                                 total=self.iters_per_epoch) if is_rank_zero(self.config) else enumerate(self.train_loader):
                pass

        losses = {}
        def stringify_losses(L): return "; ".join(map(
            lambda kv: f"{colors.fg.purple}{kv[0]}{colors.reset}: {round(kv[1].item(),3):.4e}", L.items()))
        for epoch in range(self.config.epochs):
            if self.should_early_stop():
                break
            
            self.epoch = epoch
            ################################# Train loop ##########################################################
            # if epoch > 2:
            #     self.config.w_rmse_intermedian = 0
            #     self.config.w_si_intermedian = 0
            #     print('Stop adding intermedian loss')
            if self.should_log:
                wandb.log({"Epoch": epoch}, step=self.step)
            pbar = tqdm(enumerate(self.train_loader), desc=f"Epoch: {epoch + 1}/{self.config.epochs}. Loop: Train",
                        total=self.iters_per_epoch) if is_rank_zero(self.config) else enumerate(self.train_loader)
            
            
            # for i, param_group in enumerate(self.optimizer.param_groups):
            #     group_name = param_group.get('name', f'Group {i}')  # Default to 'Group {i}' if no name is set
            #     current_lr = param_group['lr']
            #     print(f"Epoch {epoch+1}, {group_name}: Learning Rate = {current_lr:.8f}")

            for i, batch in pbar:
                if self.should_early_stop():
                    print("Early stopping")
                    break
                # print(f"Batch {self.step+1} on rank {self.config.rank}")
                losses = self.train_on_batch(batch, i)
                # print(f"trained batch {self.step+1} on rank {self.config.rank}")

                self.raise_if_nan(losses)
                if is_rank_zero(self.config) and self.config.print_losses:
                    pbar.set_description(
                        f"Epoch: {epoch + 1}/{self.config.epochs}. Loop: Train. Losses: {stringify_losses(losses)}")
                self.scheduler.step()

                if self.should_log and self.step % 1000 == 0:
                    wandb.log({f"Train/{name}": loss.item() for name, loss in losses.items()}, step=self.step)

                self.step += 1

                ########################################################################################################

                if self.test_loader:
                    if (self.step % validate_every) == 0:
                        self.print_learning_rates()
                        self.model.eval()
                        if self.should_write:
                            self.save_checkpoint(
                                f"{self.config.experiment_id}_latest.pt")

                        ################################# Validation loop ##################################################
                        # validate on the entire validation set in every process but save only from rank 0, I know, inefficient, but avoids divergence of processes
                        metrics, test_losses = self.validate()
                        # print("Validated: {}".format(metrics))
                        if self.should_log:
                            wandb.log(
                                {f"Test/{name}": tloss for name, tloss in test_losses.items()}, step=self.step)

                            wandb.log({f"Metrics/{k}": v for k,
                                      v in metrics.items()}, step=self.step)

                            if (metrics[self.metric_criterion] < best_loss) and self.should_write:
                                self.save_checkpoint(
                                    f"{self.config.experiment_id}_best.pt")
                                best_loss = metrics[self.metric_criterion]

                        self.model.train()

                        if self.config.distributed:
                            dist.barrier()
                        # print(f"Validated: {metrics} on device {self.config.rank}")

                # print(f"Finished step {self.step} on device {self.config.rank}")
                #################################################################################################

        # Save / validate at the end
        self.step += 1  # log as final point
        self.model.eval()
        self.save_checkpoint(f"{self.config.experiment_id}_latest.pt")
        if self.test_loader:

            ################################# Validation loop ##################################################
            metrics, test_losses = self.validate()
            # print("Validated: {}".format(metrics))
            if self.should_log:
                wandb.log({f"Test/{name}": tloss for name,
                          tloss in test_losses.items()}, step=self.step)
                wandb.log({f"Metrics/{k}": v for k,
                          v in metrics.items()}, step=self.step)

                if (metrics[self.metric_criterion] < best_loss) and self.should_write:
                    self.save_checkpoint(
                        f"{self.config.experiment_id}_best.pt")
                    best_loss = metrics[self.metric_criterion]

        self.model.train()

    def validate(self):
        with torch.no_grad():
            losses_avg = RunningAverageDict()
            metrics_avg = RunningAverageDict()
            for i, batch in tqdm(enumerate(self.test_loader), desc=f"Epoch: {self.epoch + 1}/{self.config.epochs}. Loop: Validation", total=len(self.test_loader), disable=not is_rank_zero(self.config)):
                metrics, losses = self.validate_on_batch(batch, val_step=i)

                if losses:
                    losses_avg.update(losses)
                if metrics:
                    metrics_avg.update(metrics)
            # breakpoint()
            return metrics_avg.get_value(), losses_avg.get_value()

    def save_checkpoint(self, filename):
        if not self.should_write:
            return
        root = self.config.save_dir
        if not os.path.isdir(root):
            os.makedirs(root)

        fpath = os.path.join(root, filename)
        m = self.model.module if self.config.multigpu else self.model
        torch.save(
            {
                "model": m.state_dict(),
                "optimizer": None,  # TODO : Change to self.optimizer.state_dict() if resume support is needed, currently None to reduce file size
                "epoch": self.epoch
            }, fpath)

    def log_images(self, rgb: Dict[str, list] = {}, depth: Dict[str, list] = {}, scalar_field: Dict[str, list] = {}, prefix="", scalar_cmap="jet", min_depth=None, max_depth=None):
        if not self.should_log:
            return

        if min_depth is None:
            try:
                min_depth = self.config.min_depth
                max_depth = self.config.max_depth
            except AttributeError:
                min_depth = None
                max_depth = None

        depth = {k: colorize(v, vmin=min_depth, vmax=max_depth)
                 for k, v in depth.items()}
        scalar_field = {k: colorize(
            v, vmin=None, vmax=None, cmap=scalar_cmap) for k, v in scalar_field.items()}
        images = {**rgb, **depth, **scalar_field}
        wimages = {
            prefix+"Predictions": [wandb.Image(v, caption=k) for k, v in images.items()]}
        wandb.log(wimages, step=self.step)

    def log_line_plot(self, data):
        if not self.should_log:
            return

        plt.plot(data)
        plt.ylabel("Scale factors")
        wandb.log({"Scale factors": wandb.Image(plt)}, step=self.step)
        plt.close()

    def log_bar_plot(self, title, labels, values):
        if not self.should_log:
            return

        data = [[label, val] for (label, val) in zip(labels, values)]
        table = wandb.Table(data=data, columns=["label", "value"])
        wandb.log({title: wandb.plot.bar(table, "label",
                  "value", title=title)}, step=self.step)
