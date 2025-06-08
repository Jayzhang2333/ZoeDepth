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
import matplotlib as mpl
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0
import torch
import torch.nn as nn
from zoedepth.utils.easydict import EasyDict as edict
from tqdm import tqdm

from zoedepth.data.data_mono import DepthDataLoader
from zoedepth.models.builder import build_model
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.utils.config import change_dataset, get_config, ALL_EVAL_DATASETS, ALL_INDOOR, ALL_OUTDOOR
from zoedepth.utils.misc import (RunningAverageDict, colors, compute_metrics,evaluation_on_ranges,
                        count_parameters)

import tifffile
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from scipy.spatial import QhullError

def show_images_four_sources(source1, source2, source3, source4):
    """
    Display images from four sources side by side in a row,
    with the colorbar for the last image having 0.9 the height of the image and centered vertically.
    
    Args:
        source1, source2, source3, source4:
            Four input tensors or numpy arrays of images.
            Expected shapes: (N, C, H, W) for tensors.
    """
    def preprocess_images(source):
        # Convert to numpy if torch.Tensor
        if not isinstance(source, np.ndarray):
            source = source.detach().cpu().numpy()
        # Move channel dimension from (N, C, H, W) -> (N, H, W, C)
        return np.transpose(source, (0, 2, 3, 1))
    
    images1 = preprocess_images(source1)
    images2 = preprocess_images(source2)
    images3 = preprocess_images(source3)
    images4 = preprocess_images(source4)
    
    # Ensure all have the same batch size
    batch_size = images1.shape[0]
    assert batch_size == images2.shape[0] == images3.shape[0] == images4.shape[0], \
           "All inputs must have the same batch size."
    
    # Create a figure with 5 columns: 4 for images, 1 for the colorbar
    fig = plt.figure(figsize=(20, 5 * batch_size))
    # The first 4 columns are equally sized; the 5th column is allocated a narrow slot.
    gs = gridspec.GridSpec(nrows=batch_size, ncols=5,
                           width_ratios=[1, 1, 1, 1, 0.05],
                           wspace=0.0,  # no horizontal gap
                           hspace=0.0)  # no vertical gap
    
    for i in range(batch_size):
        # First image
        ax1 = fig.add_subplot(gs[i, 0])
        im1 = ax1.imshow(images1[i])
        ax1.axis('off')
        
        # Second image
        ax2 = fig.add_subplot(gs[i, 1])
        im2 = ax2.imshow(images2[i], cmap='viridis_r')
        ax2.axis('off')
        
        # Third image
        ax3 = fig.add_subplot(gs[i, 2])
        im3 = ax3.imshow(images3[i], cmap='viridis_r')
        ax3.axis('off')
        
        # Fourth image
        ax4 = fig.add_subplot(gs[i, 3])
        im4 = ax4.imshow(images4[i], cmap='viridis')
        ax4.axis('off')
        
        # Colorbar in its own column so it doesn't shrink the image
        cax = fig.add_subplot(gs[i, 4])
        # Get the position of the fourth image axes
        pos4 = ax4.get_position()
        # Get the original width of the colorbar axes
        pos_cax = cax.get_position()
        # Set new height for the colorbar (90% of the image height)
        new_height = pos4.height * 0.9
        # Center the colorbar vertically with the image
        new_y = pos4.y0 + (pos4.height - new_height) / 2
        # Adjust the colorbar axes position: same x position as before, new y and height
        cax.set_position([pos4.x1, new_y, pos_cax.width, new_height])
        plt.colorbar(im4, cax=cax)
    
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

    for idx in range(batch_size):
        # Display source1 image
        axes[idx][0].imshow(images1[idx])
        axes[idx][0].set_title("Source 1")
        axes[idx][0].axis('off')

        # Display source2 image
        im2 = axes[idx][1].imshow(images2[idx])
        axes[idx][1].set_title("Source 2")
        axes[idx][1].axis('off')
        plt.colorbar(im2, ax=axes[idx][1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

def show_images_three_sources(source1, source2, source3,
                              name1='Source1', name2='Source2', name3='Source3',
                              figure_name='My Figure'):
    """
    Display images from three sources side by side in a row, with a custom figure window name.

    Args:
        source1, source2, source3: Input tensors or numpy arrays of images.
                                   Expected shapes: (N, C, H, W) for tensors.
        name1, name2, name3: Titles for each image source.
        figure_name: Name for the figure window.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    def preprocess_images(source):
        if isinstance(source, np.ndarray):
            images = source
        else:  # Assume tensor
            images = source.detach().cpu().numpy()
        images = np.transpose(images, (0, 2, 3, 1))  # Convert from CHW to HWC
        return images

    # Preprocess all three sources
    images1 = preprocess_images(source1)
    images2 = preprocess_images(source2)
    images3 = preprocess_images(source3)

    # Ensure the batch sizes match
    assert images1.shape[0] == images2.shape[0] == images3.shape[0], "Batch sizes must match."

    batch_size = images1.shape[0]
    fig, axes = plt.subplots(batch_size, 3, figsize=(20, 5 * batch_size), num=figure_name)

    if batch_size == 1:  # Handle case where batch size is 1
        axes = [axes]

    for idx in range(batch_size):
        # Display source1 image
        axes[idx][0].imshow(images1[idx])
        axes[idx][0].set_title(name1)
        axes[idx][0].axis('off')

        # Display source2 image
        im2 = axes[idx][1].imshow(images2[idx])
        axes[idx][2].axis('off')
        axes[idx][1].set_title(name2)
        plt.colorbar(im2, ax=axes[idx][1], fraction=0.046, pad=0.04)

        # Display source3 image
        im3 = axes[idx][2].imshow(images3[idx])
        axes[idx][2].set_title(name3)
        axes[idx][2].axis('off')
        plt.colorbar(im3, ax=axes[idx][2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def infer(model, images, sparse_features, config, image_no_norm, **kwargs):
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

    pred1 = model(images,prompt_depth=sparse_features, image_no_norm = image_no_norm)
    # rel_1 = pred1['rel'].squeeze().cpu().numpy()
    pred1 = get_depth_from_prediction(pred1)
    return pred1

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

    


def evaluate(model, test_loader, config, round_vals=True, round_precision=3, multi_range_evaluation = None):
    model.eval()
    if multi_range_evaluation is not None:
        metrics_list = []
        for i in multi_range_evaluation:
             metric_temp = RunningAverageDict()
             metrics_list.append(metric_temp)
    else:
        metrics = RunningAverageDict()
    # count = 0
    for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
       
        # if i == 100:
        #     break
        if 'has_valid_depth' in sample:
            
            
            if not sample['has_valid_depth']:
                print('no valid depth')
                
                continue
        image, depth, image_no_norm = sample['image'], sample['depth'], sample['image_no_norm']
        image, depth, image_no_norm = image.cuda(), depth.cuda(), image_no_norm.cuda()
        sparse_features = sample['sparse_map'].cuda().float() 

        # print(sparse_features.shape)
        
        valid_points_mask = (sparse_features >= config.min_depth) & (sparse_features <= config.max_depth)

        # Count the number of valid points within the range
        valid_points_count = valid_points_mask.sum().item()  # Convert to a Python int
    
        if valid_points_count<4:
            print("valid_points_count < 4")
            continue

        gt_valid_points_mask = (depth >= config.min_depth) & (depth <= config.max_depth)

        # Count the number of valid points within the range
        gt_valid_points_count = gt_valid_points_mask.sum().item()  # Convert to a Python int
        
        if gt_valid_points_count<=0:
            print('no valid gt')
            continue
        
        depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
        # print(depth.shape)
        # show_images_two_sources(image,depth)
        focal = sample.get('focal', torch.Tensor(
            [715.0873]).cuda())  # This magic number (focal length) is only used for evaluating BTS model
        # print('infer')
        # pred = infer(model, image, sparse_features, config, image_no_norm,dataset=sample['dataset'][0], focal=focal)

        try:
            pred = infer(
                model,
                image,
                sparse_features,
                config,
                image_no_norm,
                dataset=sample['dataset'][0],
                focal=focal
            )
        except QhullError as e:
            # Skip this sample entirely if the griddata interpolation fails
            print(f"Skipped {sample['dataset'][0]} due to QhullError: {e}")
            continue


        # depth_filtered = torch.where(depth >10.0, 10.0, depth)
        # depth_filtered = torch.where(depth_filtered <0.1, depth_filtered.max()+2, depth_filtered)
        
        # upsampled_pred = nn.functional.interpolate(
        #     pred, depth_filtered.shape[2:], mode="bilinear", align_corners=True
        # )
        # upsampled_pred = torch.where(upsampled_pred >10.0, 10.0, upsampled_pred)
        # upsampled_image = nn.functional.interpolate(
        #     image_no_norm, depth_filtered.shape[2:], mode="bilinear", align_corners=True
        # )
        
        # error_map = torch.abs(upsampled_pred - depth_filtered)
        # error_map = torch.where(depth_filtered == depth_filtered.max(), 0, error_map)
        # show_images_four_sources(upsampled_image,upsampled_pred, depth_filtered, error_map)
        # show_images_three_sources(upsampled_image,upsampled_pred, depth_filtered, name1 = 'Image', name2 = 'Prediction', name3 = 'Stereo Depth', figure_name = i)
        
        if multi_range_evaluation is not None:
            result = evaluation_on_ranges(depth, pred, evaluation_range=multi_range_evaluation)
        else:
            result = compute_metrics(depth, pred, config=config)

        if result == None:
            continue
        else:
            if multi_range_evaluation is not None:
                for i in range(len(metrics_list)):
                    metrics_list[i].update(result[i])

            else:

                metrics.update(result)
            # count = count +1
        

    if round_vals:
        def r(m): return round(m, round_precision)
    else:
        def r(m): return m

    
    if multi_range_evaluation is not None:
        
        for i in range(len(metrics_list)):
            if metrics_list[i] is not None and metrics_list[i].get_value() is not None:
                metrics_list[i] = {k: r(v) for k, v in metrics_list[i].get_value().items()}
            else:
                metrics_list[i] = {}
    else:
        metrics = {k: r(v) for k, v in metrics.get_value().items()}
    # print(count)
    if multi_range_evaluation is not None:
        return metrics_list
    else:
        return [metrics]

def main(config):
    model = build_model(config)
    
    # # 1) Load and (if necessary) fix up prefixes so ckpt_keys == model_keys
    # depth_anything_ckpt = torch.load('/home/jay/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth', map_location="cpu")
    # # 3) Grab the model’s state dict
    # mdl_sd = model.state_dict()
    # # 4) For each checkpoint key you care about, compute the max‐abs difference
    # for k, v in depth_anything_ckpt.items():
    #     if k in mdl_sd:
    #         diff = (mdl_sd[k] - v).abs().max().item()
    #         print(f"{k:40s}  max abs diff = {diff:.3e}")
    # breakpoint()

    test_loader = DepthDataLoader(config, 'online_eval').data
    model = model.cuda()
    multi_range_evaluation = [2,5,10]
    # metrics_list = evaluate(model, test_loader, config, multi_range_evaluation = multi_range_evaluation)
    metrics_list = evaluate(model, test_loader, config)
    print(f"{colors.fg.green}")
    for i in range(len(metrics_list)):
        print(f'Range 0.1 - {multi_range_evaluation[i]}')
        print(metrics_list[i])
        print()
    print(f"{colors.reset}")

    for i in metrics_list:
        i['#params'] = f"{round(count_parameters(model, include_all=True)/1e6, 2)}M"
    return metrics_list


def eval_model(model_name, pretrained_resource, dataset='nyu', **kwargs):

    # Load default pretrained resource defined in config if not set
    overwrite = {**kwargs, "pretrained_resource": pretrained_resource} if pretrained_resource else kwargs
    config = get_config(model_name, "eval", dataset, **overwrite)
    # config = change_dataset(config, dataset)  # change the dataset
    pprint(config)
    print(f"Evaluating {model_name} on {dataset}...")
    metrics_list = main(config)
    return metrics_list

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

   
    datasets = [args.dataset]
    
    for dataset in datasets:
        eval_model(args.model, pretrained_resource=args.pretrained_resource,
                    dataset=dataset, **overwrite_kwargs)
