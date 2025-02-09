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

import argparse
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from zoedepth.utils.easydict import EasyDict as edict
from tqdm import tqdm

from zoedepth.data.data_mono import DepthDataLoader
from zoedepth.models.builder import build_model
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.utils.config import change_dataset, get_config, ALL_EVAL_DATASETS, ALL_INDOOR, ALL_OUTDOOR
from zoedepth.utils.misc import (RunningAverageDict, colors, compute_metrics,
                        count_parameters)

import tifffile


@torch.no_grad()
def infer(model, images, sparse_features, config, **kwargs):
    # this inference has flip augmentation
    """Inference with flip augmentation"""
    # images.shape = N, C, H, W
    def get_depth_from_prediction(pred):
        if isinstance(pred, torch.Tensor):
            pred = pred  # pass
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        elif isinstance(pred, dict): #this is our case, the output is a dictionary
            pred = pred['metric_depth'] if 'metric_depth' in pred else pred['out']
        else:
            raise NotImplementedError(f"Unknown output type {type(pred)}")
        return pred

    pred1 = model(images,sparse_feature=sparse_features, input_height = config.input_height, input_width = config.input_width, **kwargs)
    # rel_1 = pred1['rel'].squeeze().cpu().numpy()
    pred1 = get_depth_from_prediction(pred1)
    # return pred1

    pred2 = model(torch.flip(images, [3]), sparse_feature=torch.flip(sparse_features, [3]), input_height = config.input_height, input_width = config.input_width, **kwargs)
    # rel_2 = pred2['rel'].squeeze().cpu().numpy()
    # flipped_rel_2 = rel_2[:, ::-1]
    # difference = np.abs(rel_1 - flipped_rel_2)
    pred2 = get_depth_from_prediction(pred2)
    pred2 = torch.flip(pred2, [3])

    mean_pred = 0.5 * (pred1 + pred2)

    

    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # axs[0].imshow(rel_1, cmap='viridis')
    # axs[0].set_title('rel 1')
    # axs[0].axis('off')

    # axs[1].imshow(rel_2, cmap='viridis')
    # axs[1].set_title('rel 2')
    # axs[1].axis('off')

    # axs[2].imshow(difference, cmap='viridis')
    # axs[2].set_title('difference')
    # axs[2].axis('off')

    # plt.show()

    return mean_pred

    


@torch.no_grad()
def evaluate(model, test_loader, config, round_vals=True, round_precision=3):
    model.eval()
    metrics = RunningAverageDict()
    count = 0
    for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
        if 'has_valid_depth' in sample:
            
            
            if not sample['has_valid_depth']:
                
                continue
        image, depth = sample['image'], sample['depth']
        image, depth = image.cuda(), depth.cuda()
        sparse_features = sample['sparse_map'].cuda().float() 
        nonzero_elements = sparse_features[sparse_features != 0]

        # Handle the case where there might be no non-zero elements
        if nonzero_elements.numel() > 0:
            nonzero_min = nonzero_elements.min()
            nonzero_max = nonzero_elements.max()
            print("Non-zero min:", nonzero_min.item(), "Non-zero max:", nonzero_max.item())
        else:
            print("No non-zero elements in the tensor.")
        # print(sparse_features.shape)
        
        valid_points_mask = (sparse_features >= config.min_depth) & (sparse_features <= config.max_depth)

        # Count the number of valid points within the range
        valid_points_count = valid_points_mask.sum().item()  # Convert to a Python int
    
        if valid_points_count<=4:
            continue

        valid_points_mask = (depth >= config.min_depth) & (depth <= config.max_depth)

        # Count the number of valid points within the range
        valid_points_count = valid_points_mask.sum().item()  # Convert to a Python int
    
        if valid_points_count<=0:
            continue
        
        depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
        focal = sample.get('focal', torch.Tensor(
            [715.0873]).cuda())  # This magic number (focal length) is only used for evaluating BTS model
        
        pred = infer(model, image, sparse_features, config, dataset=sample['dataset'][0], focal=focal)

        # print(pred.shape)

        # Save image, depth, pred for visualization
        if "save_images" in config and config.save_images:
            import os
            # print("Saving images ...")
            from PIL import Image
            import torchvision.transforms as transforms
            from zoedepth.utils.misc import colorize

            os.makedirs(config.save_images, exist_ok=True)
            # def save_image(img, path):
            d = colorize(depth.squeeze().cpu().numpy(), 0, 10)
            p = colorize(pred.squeeze().cpu().numpy(), 0, 10)
            im = transforms.ToPILImage()(image.squeeze().cpu())
            im.save(os.path.join(config.save_images, f"{i}_img.png"))
            Image.fromarray(d).save(os.path.join(config.save_images, f"{i}_depth.png"))
            Image.fromarray(p).save(os.path.join(config.save_images, f"{i}_pred.png"))

        ##################################################################
        # predict_depth = nn.functional.interpolate(
        #     pred, depth.shape[-2:], mode='bilinear', align_corners=True)

        # predict_depth = predict_depth.squeeze().cpu().numpy()
        # predict_depth[predict_depth < config.min_depth_eval] = config.min_depth_eval
        # predict_depth[predict_depth > config.max_depth_eval] = config.max_depth_eval
        # predict_depth[np.isinf(predict_depth)] = config.max_depth_eval
        # predict_depth[np.isnan(predict_depth)] = config.min_depth_eval

        # # print(sample['image_path'])
        # # depth_path = sample['image_path'][0].replace('/imgs/', '/depth_pred_masking/')
        # # tifffile.imwrite(depth_path, predict_depth, dtype=np.float32)



        # gt_depth = depth.squeeze().cpu().numpy()
        # gt_depth_display = gt_depth.copy()
        # gt_depth_display[gt_depth_display==0] = config.max_depth_eval
        # gt_depth_display[gt_depth_display < config.min_depth_eval] = config.min_depth_eval
        # gt_depth_display[gt_depth_display > config.max_depth_eval] = config.max_depth_eval


        # valid_mask = np.logical_and(
        #     gt_depth > config.min_depth_eval, gt_depth < config.max_depth_eval)

        # masked_difference = np.where(valid_mask, np.abs(predict_depth - gt_depth), 0)
        # # error_path = sample['image_path'][0].replace('/imgs/', '/error_map_masking/')
        # # tifffile.imwrite(error_path, masked_difference, dtype=np.float32)
        # # print(np.shape(masked_difference))
        # # print(np.shape(valid_mask))

        # rgb_image = image.squeeze().cpu().numpy()
        # rgb_image = np.transpose(rgb_image, (1, 2, 0))

        # fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # Adjust the number of subplots to 4 and increase the figure width

        # # RGB Image plot
        # axs[0].imshow(rgb_image)
        # axs[0].set_title('RGB Image', fontsize=14, fontweight='bold')
        # axs[0].axis('off')

        # # Prediction plot
        # im1 = axs[1].imshow(predict_depth, cmap='inferno_r')
        # axs[1].set_title('Prediction', fontsize=14, fontweight='bold')
        # axs[1].axis('off')
        # cbar1 = fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
        # cbar1.set_label('Depth', fontsize=14, fontweight='bold')

        # # Ground Truth plot
        # im2 = axs[2].imshow(gt_depth_display, cmap='inferno_r')  # Use gt_depth here
        # axs[2].set_title('Ground Truth', fontsize=14, fontweight='bold')
        # axs[2].axis('off')
        # cbar2 = fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
        # cbar2.set_label('Depth', fontsize=14, fontweight='bold')

        # # Difference Map plot
        # im3 = axs[3].imshow(masked_difference, cmap='inferno')  # Use masked_difference here
        # axs[3].set_title('Difference Map', fontsize=14, fontweight='bold')
        # axs[3].axis('off')
        # cbar3 = fig.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)
        # cbar3.set_label('Depth', fontsize=14, fontweight='bold')

        # plt.tight_layout()
        # plt.show()

        
        ######################################################################

        # print(depth.shape, pred.shape)
        result = compute_metrics(depth, pred, config=config)
        if result == None:
            continue
        else:
            metrics.update(result)
            count = count +1

    if round_vals:
        def r(m): return round(m, round_precision)
    else:
        def r(m): return m
    metrics = {k: r(v) for k, v in metrics.get_value().items()}
    print(count)
    return metrics

def main(config):
    model = build_model(config)
    test_loader = DepthDataLoader(config, 'online_eval').data
    model = model.cuda()
    metrics = evaluate(model, test_loader, config)
    print(f"{colors.fg.green}")
    print(metrics)
    print(f"{colors.reset}")
    metrics['#params'] = f"{round(count_parameters(model, include_all=True)/1e6, 2)}M"
    return metrics


def eval_model(model_name, pretrained_resource, dataset='nyu', **kwargs):

    # Load default pretrained resource defined in config if not set
    overwrite = {**kwargs, "pretrained_resource": pretrained_resource} if pretrained_resource else kwargs
    config = get_config(model_name, "eval", dataset, **overwrite)
    # config = change_dataset(config, dataset)  # change the dataset
    pprint(config)
    print(f"Evaluating {model_name} on {dataset}...")
    metrics = main(config)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        required=True, help="Name of the model to evaluate")
    parser.add_argument("-p", "--pretrained_resource", type=str,
                        required=False, default=None, help="Pretrained resource to use for fetching weights. If not set, default resource from model config is used,  Refer models.model_io.load_state_from_resource for more details.")
    parser.add_argument("-d", "--dataset", type=str, required=False,
                        default='nyu', help="Dataset to evaluate on")

    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)

    if "ALL_INDOOR" in args.dataset:
        datasets = ALL_INDOOR
    elif "ALL_OUTDOOR" in args.dataset:
        datasets = ALL_OUTDOOR
    elif "ALL" in args.dataset:
        datasets = ALL_EVAL_DATASETS
    elif "," in args.dataset:
        datasets = args.dataset.split(",")
    else:
        datasets = [args.dataset]
    
    for dataset in datasets:
        eval_model(args.model, pretrained_resource=args.pretrained_resource,
                    dataset=dataset, **overwrite_kwargs)
