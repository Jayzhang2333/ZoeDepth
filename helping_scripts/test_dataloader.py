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

from zoedepth.utils.misc import count_parameters, parallelize
from zoedepth.utils.config import get_config
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.trainers.builder import get_trainer
from zoedepth.models.builder import build_model
from zoedepth.data.data_mono import DepthDataLoader
import torch.utils.data.distributed
import torch.multiprocessing as mp
import torch
import numpy as np
from pprint import pprint
import argparse
import os
import matplotlib.pyplot as plt




def fix_random_seed(seed: int):
    import random

    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True




def main_worker(gpu, config):

    seed = config.seed if 'seed' in config and config.seed else 43
    fix_random_seed(seed)

    config.gpu = gpu

    
    # .data returns the actual datalodar
    train_loader = DepthDataLoader(config, "train").data

    for i, batch in enumerate(train_loader):
        if i == 0:
            # Get the first sample from the batch
            rgb_image = batch['image'][0].permute(1, 2, 0).cpu().numpy()  # Convert to HWC for visualization
            depth_map = batch['depth'][0].squeeze().cpu().numpy()  # Squeeze to remove (1, H, W) -> (H, W)
            feature_map = batch['sparse_map'][0].cpu().numpy()  # Feature map (2 channels)

            # Print shapes
            print(f"RGB Image shape: {rgb_image.shape}")          # Should be (H, W, 3)
            print(f"Depth Map shape: {depth_map.shape}")          # Should be (H, W)
            print(f"Feature Map shape: {feature_map.shape}")      # Should be (2, H, W)

            # Visualize the data
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # Adjust figsize to control overall size
            
            # RGB Image
            axs[0].imshow(rgb_image)
            axs[0].set_title('RGB Image')
            axs[0].axis('off')
            axs[0].set_aspect('equal')  # Or 'equal' to preserve natural aspect ratio

            # Depth Map
            im = axs[1].imshow(depth_map, cmap='plasma')
            axs[1].set_title('Depth Map')
            axs[1].axis('off')
            axs[1].set_aspect('equal')  # Or 'equal'
            fig.colorbar(im, ax=axs[1])

            # Feature Map Channel 1
            im = axs[2].imshow(feature_map[0], cmap='plasma')
            axs[2].set_title('Feature Map - Channel 1')
            axs[2].axis('off')
            axs[2].set_aspect('equal')  # Or 'equal'
            fig.colorbar(im, ax=axs[2])

            # Feature Map Channel 2
            im = axs[3].imshow(feature_map[1], cmap='plasma')
            axs[3].set_title('Feature Map - Channel 2')
            axs[3].axis('off')
            axs[3].set_aspect('equal')  # Or 'equal'
            fig.colorbar(im, ax=axs[3])

            plt.show()
            break
        


if __name__ == '__main__':
    mp.set_start_method('forkserver')

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="zoedepth_sparse_feature")
    parser.add_argument("-d", "--dataset", type=str, default='nyu_sparse_feature')
    parser.add_argument("--trainer", type=str, default=None)

    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)

    overwrite_kwargs["model"] = args.model
    if args.trainer is not None:
        overwrite_kwargs["trainer"] = args.trainer

    config = get_config(args.model, "train", args.dataset, **overwrite_kwargs)
    # git_commit()
    if config.use_shared_dict:
        shared_dict = mp.Manager().dict()
    else:
        shared_dict = None
    config.shared_dict = shared_dict

    config.batch_size = config.bs
    config.mode = 'train'
    if config.root != "." and not os.path.isdir(config.root):
        os.makedirs(config.root)

    try:
        node_str = os.environ['SLURM_JOB_NODELIST'].replace(
            '[', '').replace(']', '')
        nodes = node_str.split(',')

        config.world_size = len(nodes)
        config.rank = int(os.environ['SLURM_PROCID'])
        # config.save_dir = "/ibex/scratch/bhatsf/videodepth/checkpoints"

    except KeyError as e:
        # We are NOT using SLURM
        config.world_size = 1
        config.rank = 0
        nodes = ["127.0.0.1"]

    if config.distributed:

        print(config.rank)
        port = np.random.randint(15000, 15025)
        config.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        print(config.dist_url)
        config.dist_backend = 'nccl'
        config.gpu = None

    ngpus_per_node = torch.cuda.device_count()
    config.num_workers = config.workers
    config.ngpus_per_node = ngpus_per_node
    print("Config:")
    pprint(config)
    if config.distributed:
        config.world_size = ngpus_per_node * config.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, config))
    else:
        if ngpus_per_node == 1:
            config.gpu = 0
        main_worker(config.gpu, config)
