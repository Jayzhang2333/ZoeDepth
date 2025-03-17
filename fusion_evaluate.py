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
from zoedepth.utils.misc import (RunningAverageDict, colors, compute_metrics,evaluation_on_ranges,
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

    pred1 = model(images, sparse_feature=sparse_features, input_height = config.input_height, input_width = config.input_width, **kwargs)
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
                
                continue
        image, depth = sample['image'], sample['depth']
        image, depth = image.cuda(), depth.cuda()
        sparse_features = sample['sparse_map'].cuda().float() 

        # print(sparse_features.shape)
        
        valid_points_mask = (sparse_features >= config.min_depth) & (sparse_features <= config.max_depth)

        # Count the number of valid points within the range
        valid_points_count = valid_points_mask.sum().item()  # Convert to a Python int
    
        if valid_points_count<=4:
            continue

        gt_valid_points_mask = (depth >= config.min_depth) & (depth <= config.max_depth)

        # Count the number of valid points within the range
        gt_valid_points_count = gt_valid_points_mask.sum().item()  # Convert to a Python int
    
        if gt_valid_points_count<=0:
            continue
        
        depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
        focal = sample.get('focal', torch.Tensor(
            [715.0873]).cuda())  # This magic number (focal length) is only used for evaluating BTS model
        pred = infer(model, image, sparse_features, config, dataset=sample['dataset'][0], focal=focal)

       
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
