

import itertools
import os
import random

import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.utils.data.distributed
from zoedepth.utils.easydict import EasyDict as edict
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from zoedepth.utils.config import change_dataset

from .ddad import get_ddad_loader
from .diml_indoor_test import get_diml_indoor_loader
from .diml_outdoor_test import get_diml_outdoor_loader
from .diode import get_diode_loader
from .hypersim import get_hypersim_loader
from .ibims import get_ibims_loader
from .sun_rgbd_loader import get_sunrgbd_loader
from .vkitti import get_vkitti_loader
from .vkitti2 import get_vkitti2_loader

from .preprocess import CropParams, get_white_border, get_black_border
import matplotlib.pyplot as plt
from zoedepth.models.layers.global_alignment import AnchorInterpolator2D


def sparsify_feature_map(sparse_feature_map):
    # Get indices of valid points (nonzero values)
    valid_indices = np.argwhere(sparse_feature_map > 0)

    # If there are more than 200 valid points, randomly select 200
    if len(valid_indices) > 200:
        selected_indices = valid_indices[np.random.choice(len(valid_indices), 200, replace=False)]
        
        # Create a new sparse map with zeros
        new_sparse_map = np.zeros_like(sparse_feature_map)
        
        # Assign original values to the selected indices
        for idx in selected_indices:
            new_sparse_map[tuple(idx)] = sparse_feature_map[tuple(idx)]
        
        return new_sparse_map
    else:
        return sparse_feature_map  # Return as is if <= 200 valid points
    

def filter_lower_33_percent(sparse_feature_map):
    """
    sparse_feature_map: 2D NumPy array, where non-zero entries represent valid depth/feature values.
    We only keep values in the lower 33% of valid entries; everything else becomes zero.
    """
    # 1. Extract valid (non-zero) values
    valid_values = sparse_feature_map[sparse_feature_map > 0]
    
    # Edge case: if there are no non-zero entries, just return an array of zeros
    if valid_values.size == 0:
        return np.zeros_like(sparse_feature_map)
    
    # 2. Find the threshold corresponding to the 33rd percentile of valid values
    threshold_33 = np.quantile(valid_values, 0.33)  # 0.33 => 33rd percentile

    # 3. Create a mask that keeps only values <= that threshold (and originally non-zero)
    filtered = np.where((sparse_feature_map > 0) & 
                        (sparse_feature_map <= threshold_33),
                        sparse_feature_map,
                        0)
    
    return filtered


def resample_sparse_depth(depth_map, new_height, new_width):
    """
    Resample a sparse depth map to a new resolution.
    
    Parameters
    ----------
    depth_map : np.ndarray (H, W)
        2D array of sparse depth values (0 indicates no depth).
    new_height : int
        Desired height for the output.
    new_width : int
        Desired width for the output.
    
    Returns
    -------
    new_depth_map : np.ndarray (new_height, new_width)
        Resampled sparse depth map. Pixels without any mapped depth will be 0.
    """
    old_height, old_width = depth_map.shape
    
    # Create arrays to store summed depth and counts in the new resolution
    new_depth_map = np.zeros((new_height, new_width), dtype=np.float32)
    new_counts = np.zeros((new_height, new_width), dtype=np.int32)
    
    # Find all valid (non-zero) pixels in the original depth map
    valid_mask = (depth_map != 0)
    # Get their indices (row = y, col = x)
    ys, xs = np.where(valid_mask)
    
    # Gather depths
    depths = depth_map[valid_mask]
    
    # For each valid pixel, compute its normalized coordinates
    #   y_frac in [0,1) = y / old_height
    #   x_frac in [0,1) = x / old_width
    # Then map to the new image by multiplying:
    #   new_y = floor(y_frac * new_height)
    #   new_x = floor(x_frac * new_width)
    # (We use floor by default with int-casting in Python.)
    
    y_fracs = ys / float(old_height)
    x_fracs = xs / float(old_width)
    
    # Compute new indices
    new_ys = (y_fracs * new_height).astype(np.int32)
    new_xs = (x_fracs * new_width).astype(np.int32)
    
    # Accumulate depth values in the new resolution
    for y_new, x_new, depth_val in zip(new_ys, new_xs, depths):
        new_depth_map[y_new, x_new] += depth_val
        new_counts[y_new, x_new] += 1
    
    # Average out pixels where more than one old pixel landed
    nonzero_mask = (new_counts > 0)
    new_depth_map[nonzero_mask] /= new_counts[nonzero_mask]
    
    return new_depth_map

def generate_feature_map_for_ga(
    feature_fp,
    original_height=480,
    original_width=640,
    new_height=336,
    new_width=448,
    inverse_depth=False,
    random=False
):
    # Check the file extension to determine the delimiter
    file_extension = os.path.splitext(feature_fp)[-1].lower()

    if file_extension == '.csv':
        df = pd.read_csv(feature_fp)
    elif file_extension == '.txt':
        df = pd.read_csv(
            feature_fp,
            delimiter=' ',
            header=0,
            names=['row', 'column', 'depth']
        )
    else:
        raise ValueError("Unsupported file format. Only CSV and TXT files are supported.")

    # If requested, randomly keep 90% of the points
    if random:
        df = df.sample(frac=0.9).reset_index(drop=True)

    # Initialize a blank depth map for the new image size
    sparse_depth_map = np.zeros((new_height, new_width), dtype=np.float32)

    # Calculate scaling factors
    scale_y = new_height / original_height
    scale_x = new_width / original_width

    # Populate the depth map
    for _, row in df.iterrows():
        # Scale pixel coordinates
        pixel_row = int(row['row'] * scale_y)
        pixel_col = int(row.get('column', row.get('col', 0)) * scale_x)

        depth_value = float(row['depth'])
        if inverse_depth:
            depth_value = 1.0 / depth_value

        # Write into sparse map if inside bounds
        if 0 <= pixel_row < new_height and 0 <= pixel_col < new_width:
            sparse_depth_map[pixel_row, pixel_col] = depth_value

    # Add channel dim
    return sparse_depth_map[..., np.newaxis]

def generate_feature_map_for_ga_left(
    feature_fp,
    original_height=480,
    original_width=640,
    new_height=336,
    new_width=448,
    inverse_depth=False,
    random=False
):
    """
    Like generate_feature_map_for_ga, but only uses sparse points
    whose rescaled column lies in the left half of the new image.
    """
    # --- Load data ---
    ext = os.path.splitext(feature_fp)[-1].lower()
    if ext == '.csv':
        df = pd.read_csv(feature_fp)
    elif ext == '.txt':
        df = pd.read_csv(feature_fp, delimiter=' ', header=0,
                         names=['row', 'column', 'depth'])
    else:
        raise ValueError("Unsupported file format. Only CSV and TXT supported.")

    # --- Optional random subsample ---
    if random:
        df = df.sample(frac=0.9).reset_index(drop=True)

    # --- Prepare empty map ---
    sparse_depth_map = np.zeros((new_height, new_width), dtype=np.float32)

    # --- Compute scale factors ---
    scale_y = new_height / original_height
    scale_x = new_width  / original_width

    # --- Populate only left-half points ---
    half_col = new_width // 2
    for _, row in df.iterrows():
        # rescale coords
        r = int(row['row'] * scale_y)
        c = int(row['column'] * scale_x)
        if not (0 <= r < new_height and 0 <= c < new_width):
            continue
        # only left half
        if c >= half_col:
            continue

        val = float(row['depth'])
        if inverse_depth:
            val = 1.0 / val

        sparse_depth_map[r, c] = val

    # add channel dim
    return sparse_depth_map[..., np.newaxis]

def generate_feature_map_for_ga_topleft(
    feature_fp,
    original_height=480,
    original_width=640,
    new_height=336,
    new_width=448,
    inverse_depth=False,
    random=False
):
    """
    Like generate_feature_map_for_ga, but only uses sparse points
    whose rescaled coordinates fall in the top-left quarter of the new image.
    """
    # --- Load data ---
    ext = os.path.splitext(feature_fp)[-1].lower()
    if ext == '.csv':
        df = pd.read_csv(feature_fp)
    elif ext == '.txt':
        df = pd.read_csv(feature_fp, delimiter=' ', header=0,
                         names=['row', 'column', 'depth'])
    else:
        raise ValueError("Unsupported file format. Only CSV and TXT supported.")

    # --- Optional random subsample ---
    if random:
        df = df.sample(frac=0.9).reset_index(drop=True)

    # --- Prepare empty map ---
    sparse_depth_map = np.zeros((new_height, new_width), dtype=np.float32)

    # --- Compute scale factors ---
    scale_y = new_height / original_height
    scale_x = new_width  / original_width

    # --- Define top-left patch bounds ---
    half_h = new_height // 2
    half_w = new_width  // 2

    # --- Populate only top-left patch points ---
    for _, row in df.iterrows():
        # rescale coords
        r = int(row['row'] * scale_y)
        c = int(row['column'] * scale_x)
        # skip out-of-bounds
        if not (0 <= r < new_height and 0 <= c < new_width):
            continue
        # only top-left quarter
        if r >= half_h or c >= half_w:
            continue

        val = float(row['depth'])
        if inverse_depth:
            val = 1.0 / val

        sparse_depth_map[r, c] = val

    # add channel dim and return
    return sparse_depth_map[..., np.newaxis]


def get_distance_maps(height, width, idcs_height, idcs_width):
    """Generates distance maps from feature points to every pixel."""
    num_features = len(idcs_height)
    dist_maps = np.empty((num_features, height, width))
    
    for idx in range(num_features):
        y, x = idcs_height[idx], idcs_width[idx]
        y_grid, x_grid = np.ogrid[:height, :width]
        dist_maps[idx] = np.sqrt((y_grid - y) ** 2 + (x_grid - x) ** 2)
    
    return dist_maps

# def get_probability_maps(distance_maps):
#     """Convert pixel distance to probability."""
#     max_dist = np.sqrt(distance_maps.shape[-2] ** 2 + distance_maps.shape[-1] ** 2)
#     probabilities = np.exp(-distance_maps / max_dist)
#     return probabilities
from scipy.stats import norm
def get_probability_maps(dist_map):
    """Takes a Nx1xHxW distance map as input and outputs a probability map.
    Pixels with small distance to closest keypoint have high probability and vice versa."""

    # Normal distribution parameters
    loc = 0.0
    scale_param = 10.0

    # Normal distribution scaling factor to ensure prob=1 at dist=0
    distribution = norm(loc=loc, scale=scale_param)
    scale = np.exp(distribution.logpdf(0))  # Calculate scale factor for normalization

    # Prior probability for every pixel
    prob_map = np.exp(distribution.logpdf(dist_map)) / scale

    # Uncomment below for exponential distribution alternative
    # r = 0.05  # rate
    # prob_map = np.exp(-r * dist_map)  # prior=1 at dist=0 without multiplying by r

    return prob_map

def get_depth_prior_from_features(features, prior_channels, height=240, width=320):
    """Takes lists of pixel indices and their respective depth probes and
    returns a dense depth prior parametrization using NumPy, displaying results
    using Matplotlib."""
    
    # batch_size = features.shape[0]

    # depth prior maps
    prior_maps = np.empty((height, width))

    # euclidean distance maps
    distance_maps = np.empty((height, width))

    # for i in range(batch_size):
    # use only entries with valid depth
    mask = features[ :, 2] > 0.0

    if not np.any(mask):
        max_dist = np.sqrt(height ** 2 + width ** 2)
        prior_maps[ ...] = 0.0
        distance_maps[ ...] = max_dist
        print(f"WARNING: Img has no valid features, using placeholders.")
        # continue

    # get list of indices and depth values
    idcs_height = np.round(features[ mask, 0]).astype(int)
    idcs_width = np.round(features[ mask, 1]).astype(int)
    depth_values = features[ mask, 2]

    # get distance maps for each feature
    sample_dist_maps = get_distance_maps(height, width, idcs_height, idcs_width)

    # find min and argmin
    dist_map_min = np.min(sample_dist_maps, axis=0)
    dist_argmin = np.argmin(sample_dist_maps, axis=0)

    # nearest neighbor prior map
    prior_map = depth_values[dist_argmin]

    # concat
    prior_maps[...] = prior_map
    distance_maps[...] = dist_map_min

    # # Display results for each image
    # plt.figure(figsize=(10, 5))

    # # Display depth prior map
    # plt.subplot(1, 2, 1)
    # plt.title(f"Depth Prior Map {i+1}")
    # plt.imshow(prior_map, cmap='jet')
    # plt.colorbar()

    # # Display probability map
    # plt.subplot(1, 2, 2)
    # probability_map = get_probability_maps(dist_map_min)
    # plt.title(f"Probability Map {i+1}")
    # plt.imshow(probability_map, cmap='jet')
    # plt.colorbar()

    # plt.show()

    # parametrization (concatenating depth map and probability map for each image)
    if prior_channels<=1:
        return np.expand_dims(prior_map, axis=2)
    else:
        parametrization = np.stack([prior_maps, get_probability_maps(distance_maps)], axis=2)
        # print(parametrization.shape)

        return parametrization

# def load_features_from_csv(csv_file):
#     """Loads features from a CSV file into a NumPy array."""
#     df = pd.read_csv(csv_file)
#     col_name = 'col' if 'col' in df.columns else 'column'

#     features = df[['row', col_name, 'depth']].to_numpy()
#     return features
import csv
def load_features_from_csv(csv_file):
    """Loads features from a CSV file into a NumPy array."""
    features = []

    with open(csv_file, mode='r') as file:
        csv_reader = csv.DictReader(file)
        col_name = 'col' if 'col' in csv_reader.fieldnames else 'column'

        # Extract the relevant columns ('row', col_name, 'depth')
        for row in csv_reader:
            features.append([float(row['row']), float(row[col_name]), float(row['depth'])])

    return np.array(features)


def load_features_from_npy(npy_file):
    data = np.load(npy_file)
    # data[:,:,0] = data[:,:,0]/1.44
    return data


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode, use_ga, **kwargs):
    return transforms.Compose([
        ToTensor(mode=mode, use_ga = use_ga,  **kwargs)
    ])


class DepthDataLoader(object):
    def __init__(self, config, mode, device='cpu', transform=None, **kwargs):
        """
        Data loader for depth datasets

        Args:
            config (dict): Config dictionary. Refer to utils/config.py
            mode (str): "train" or "online_eval"
            device (str, optional): Device to load the data on. Defaults to 'cpu'.
            transform (torchvision.transforms, optional): Transform to apply to the data. Defaults to None.
        """

        self.config = config

        if config.dataset == 'ibims':
            self.data = get_ibims_loader(config, batch_size=1, num_workers=1)
            return

        if config.dataset == 'sunrgbd':
            self.data = get_sunrgbd_loader(
                data_dir_root=config.sunrgbd_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'diml_indoor':
            self.data = get_diml_indoor_loader(
                data_dir_root=config.diml_indoor_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'diml_outdoor':
            self.data = get_diml_outdoor_loader(
                data_dir_root=config.diml_outdoor_root, batch_size=1, num_workers=1)
            return

        # original training with diode
        # if "diode" in config.dataset:
        #     self.data = get_diode_loader(
        #         config[config.dataset+"_root"], batch_size=1, num_workers=1)
        #     return

        if config.dataset == 'hypersim_test':
            self.data = get_hypersim_loader(
                config.hypersim_test_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'vkitti':
            self.data = get_vkitti_loader(
                config.vkitti_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'vkitti2':
            self.data = get_vkitti2_loader(
                config.vkitti2_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'ddad':
            self.data = get_ddad_loader(config.ddad_root, resize_shape=(
                352, 1216), batch_size=1, num_workers=1)
            return

        img_size = self.config.get("img_size", None)
        img_size = img_size if self.config.get(
            "do_input_resize", False) else None

        if transform is None:
            # print("transform is none")
            # print(img_size)
            if config.name == 'ZoeDepth_sparse_feature_ga':
                use_ga = True
            else:
                use_ga = False
            transform = preprocessing_transforms(mode, use_ga, size=img_size)

        if mode == 'train':

            Dataset = DataLoadPreprocess
            self.training_samples = Dataset(
                config, mode, transform=transform, device=device)

            if config.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.training_samples)
            else:
                self.train_sampler = None

            print(f'number of workers actually used: {config.workers}')
            
            self.data = DataLoader(self.training_samples,
                                   batch_size=config.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=config.workers,
                                   pin_memory=True,
                                   persistent_workers=True,
                                #    prefetch_factor=2,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(
                config, mode, transform=transform)
            if config.distributed:  # redundant. here only for readability and to be more explicit
                # Give whole test set to all processes (and report evaluation only on one) regardless
                self.eval_sampler = None
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=kwargs.get("shuffle_test", True),
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(
                config, mode, transform=transform)
            self.data = DataLoader(self.testing_samples,
                                   1, shuffle=False, num_workers=1)

        else:
            print(
                'mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


def repetitive_roundrobin(*iterables):
    """
    cycles through iterables but sample wise
    first yield first sample from first iterable then first sample from second iterable and so on
    then second sample from first iterable then second sample from second iterable and so on

    If one iterable is shorter than the others, it is repeated until all iterables are exhausted
    repetitive_roundrobin('ABC', 'D', 'EF') --> A D E B D F C D E
    """
    # Repetitive roundrobin
    iterables_ = [iter(it) for it in iterables]
    exhausted = [False] * len(iterables)
    while not all(exhausted):
        for i, it in enumerate(iterables_):
            try:
                yield next(it)
            except StopIteration:
                exhausted[i] = True
                iterables_[i] = itertools.cycle(iterables[i])
                # First elements may get repeated if one iterable is shorter than the others
                yield next(iterables_[i])


class RepetitiveRoundRobinDataLoader(object):
    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders

    def __iter__(self):
        return repetitive_roundrobin(*self.dataloaders)

    def __len__(self):
        # First samples get repeated, thats why the plus one
        return len(self.dataloaders) * (max(len(dl) for dl in self.dataloaders) + 1)


class MixedNYUKITTI(object):
    def __init__(self, config, mode, device='cpu', **kwargs):
        config = edict(config)
        config.workers = config.workers // 2
        self.config = config
        nyu_conf = change_dataset(edict(config), 'nyu')
        kitti_conf = change_dataset(edict(config), 'kitti')

        # make nyu default for testing
        self.config = config = nyu_conf
        img_size = self.config.get("img_size", None)
        img_size = img_size if self.config.get(
            "do_input_resize", False) else None
        if mode == 'train':
            nyu_loader = DepthDataLoader(
                nyu_conf, mode, device=device, transform=preprocessing_transforms(mode, size=img_size)).data
            kitti_loader = DepthDataLoader(
                kitti_conf, mode, device=device, transform=preprocessing_transforms(mode, size=img_size)).data
            # It has been changed to repetitive roundrobin
            self.data = RepetitiveRoundRobinDataLoader(
                nyu_loader, kitti_loader)
        else:
            self.data = DepthDataLoader(nyu_conf, mode, device=device).data


def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s


class CachedReader:
    def __init__(self, shared_dict=None):
        if shared_dict:
            self._cache = shared_dict
        else:
            self._cache = {}

    def open(self, fpath):
        im = self._cache.get(fpath, None)
        if im is None:
            im = self._cache[fpath] = Image.open(fpath)
        return im


class ImReader:
    def __init__(self):
        pass

    # @cache
    def open(self, fpath):
        return Image.open(fpath)
    
def generate_sparse_feature_map(featrue_path, prior_channels, height, width):
    # print(featrue_path)
    features = load_features_from_csv(featrue_path)

    # Reshape features into the required batch format (batch_size, num_features, 3)
    # Since we're working with one image, we expand the dimensions to simulate a batch of size 1
    # i think the np.newaxis should be repalced with batch size
    # features = features[np.newaxis, :, :]
    # print(features.shape)

    # Process and visualize the depth prior
    # parametrization = get_depth_prior_from_features(features,prior_channels, height=height, width=width)
    parametrization = get_depth_prior_from_features(features,prior_channels, height=240, width=320)

    return parametrization


def generate_sparse_feature_map_npy(featrue_path, prior_channels):
    # print(featrue_path)
    parametrization = load_features_from_npy(featrue_path)

    if prior_channels <=1:
        parametrization = np.expand_dims(parametrization[:,:,0], axis=2)

    return parametrization




class DataLoadPreprocess(Dataset):
    def __init__(self, config, mode, transform=None, is_for_online_eval=False, **kwargs):
        self.config = config
        if mode == 'online_eval':
            with open(config.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(config.filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        # print(config.img_size)
        # self.to_tensor = ToTensor(mode)
        self.is_for_online_eval = is_for_online_eval
        if config.use_shared_dict:
            self.reader = CachedReader(config.shared_dict)
        else:
            self.reader = ImReader()

    def postprocess(self, sample):
        return sample

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        # print(sample_path)
        if len(sample_path.split()) > 3:
            # focal = float(sample_path.split()[2])
            focal = 500
        else:
            focal = 500 #give a dummy value
        sample = {}

        if self.mode == 'train':
           
            image_path = os.path.join(
                self.config.data_path, remove_leading_slash(sample_path.split()[0]))
            feature_path = os.path.join(
                self.config.data_path, remove_leading_slash(sample_path.split()[-1]))
            depth_path = os.path.join(
                self.config.gt_path, remove_leading_slash(sample_path.split()[1]))

            image = self.reader.open(image_path)
           
            #at this stage, the input sparse map is still numpy
            if (self.config.name == 'videpth_spn'  or self.config.name == 'PromptDA' or self.config.name == 'ZoeDepth_sparse_feature_fusion' 
                  or self.config.name == 'ZoeDepth_videpth'or self.config.name == 'ZoeDepth_conv_trans'
                  or self.config.name == 'AffinityDC' or self.config.name == 'DA_SML') and  (feature_path.lower().endswith('.csv') or feature_path.lower().endswith('.txt')):
                
                sparse_feature_map = generate_feature_map_for_ga(feature_path, original_height=self.config.sparse_feature_height,\
                                                                  original_width=self.config.sparse_feature_width, new_height=self.config.img_size[0], \
                                                                    new_width=self.config.img_size[1], inverse_depth=False, random=True) 
                   
            else:
                sparse_feature_map = generate_sparse_feature_map_npy(feature_path, self.config.prior_channels)
            
            if self.config.dataset == 'tartanair':
                depth_gt = np.load(depth_path)
            else:
                depth_gt = self.reader.open(depth_path)
            
            
            w, h = image.size

            image = np.asarray(image, dtype=np.float32) / 255.0
            # sparse_feature_map = np.asarray(sparse_feature_map, dtype=np.float32)
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
           
            depth_gt = np.expand_dims(depth_gt, axis=2)

           
            
            if  self.config.dataset == 'tartanair'  or self.config.dataset == 'flsea_sparse_feature':
                depth_gt = depth_gt / 1.0
            else:
               
                depth_gt = depth_gt / 256.0

            image, sparse_feature_map, depth_gt = self.train_preprocess(image, sparse_feature_map, depth_gt)
            
            #until now, data is still numpy
            sample = {'image': image, 'sparse_map': sparse_feature_map , 'depth': depth_gt, 'focal': focal, **sample}

        else:
            if self.mode == 'online_eval':
                data_path = self.config.data_path_eval
            else:
                data_path = self.config.data_path

            image_path = os.path.join(
                data_path, remove_leading_slash(sample_path.split()[0]))
            
            feature_path = os.path.join(
                data_path, remove_leading_slash(sample_path.split()[-1]))
            
            # print(sample_path.split()[0])
            image = np.asarray(self.reader.open(image_path),
                               dtype=np.float32) / 255.0

           
            if (self.config.name == 'videpth_spn'  or self.config.name == 'PromptDA' or self.config.name == 'ZoeDepth_sparse_feature_fusion' 
                  or self.config.name == 'ZoeDepth_videpth'or self.config.name == 'ZoeDepth_conv_trans'
                  or self.config.name == 'AffinityDC' or self.config.name == 'DA_SML') and  (feature_path.lower().endswith('.csv') or feature_path.lower().endswith('.txt')):
               
                sparse_feature_map = generate_feature_map_for_ga(feature_path, original_height=self.config.sparse_feature_height, \
                                                                 original_width=self.config.sparse_feature_width, \
                                                                    new_height=self.config.img_size[0], new_width=self.config.img_size[1],\
                                                                          inverse_depth=False, random=False)
                if self.config.dataset == 'lizard_sparse_feature':
                    sparse_feature_map = sparse_feature_map/1.44
                    
            else:
                sparse_feature_map = generate_sparse_feature_map_npy(feature_path, self.config.prior_channels)
                
            
            if self.mode == 'online_eval':
                gt_path = self.config.gt_path_eval
                depth_path = os.path.join(
                    gt_path, remove_leading_slash(sample_path.split()[1]))
                has_valid_depth = False
                try:
                    if self.config.dataset == 'tartanair':
                        depth_gt = np.load(depth_path)   
                    else:
                        depth_gt = self.reader.open(depth_path)

                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    # print('Missing gt for {}'.format(image_path))

                if has_valid_depth:
                    depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    
                    if self.config.dataset == "flsea_sparse_feature" or self.config.dataset == "lizard_sparse_feature" or self.config.dataset == 'diode_sparse_feature' or self.config.dataset == 'tartanair' \
                    or self.config.dataset == 'reach_tank_sparse_feature' or self.config.dataset == 'sea_thru' or self.config.dataset == 'turbid_test' or self.config.dataset == 'cape_don':
                        depth_gt = depth_gt / 1.0

                    else:
                        depth_gt = depth_gt / 256.0

            if self.mode == 'online_eval':
                sample = {'image': image,'sparse_map': sparse_feature_map, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth,
                          'image_path': sample_path.split()[0], 'feature_path': sample_path.split()[-1],'depth_path': sample_path.split()[1]}
            else:
                sample = {'image': image, 'sparse_map': sparse_feature_map,'focal': focal}

        # if  (self.mode == 'train') or ('has_valid_depth' in sample and sample['has_valid_depth']):
            
        mask = np.logical_and(depth_gt > self.config.min_depth,
                            depth_gt < self.config.max_depth).squeeze()[None, ...]
        
        infinit_mask = (depth_gt >= self.config.max_depth).squeeze()[None, ...]
        
        sample['mask'] = mask
        sample['infinit_mask'] = infinit_mask

        
        if self.transform:
            # print("Go through transform")
            sample = self.transform(sample)

        
        

        sample = self.postprocess(sample)
        sample['dataset'] = self.config.dataset
        sample = {**sample, 'image_path': sample_path.split()[0], 'feature_path': sample_path.split()[-1],'depth_path': sample_path.split()[1]}

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result
    
    def rotate_feature_map(self, feature_map, angle, flag=Image.BILINEAR):
        from scipy.ndimage import rotate
        rotated_feature_map = rotate(feature_map, angle, reshape=False, order=1)
        
        return rotated_feature_map

    def random_crop(self, img, feature_map, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        assert img.shape[1] == feature_map.shape[1]
        assert img.shape[0] == feature_map.shape[0]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        feature_map = feature_map[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]

        return img,feature_map, depth
    
    def random_translate(self, img, feature_map, depth, max_t=20):
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        assert img.shape[1] == feature_map.shape[1]
        assert img.shape[0] == feature_map.shape[0]
        p = self.config.translate_prob
        do_translate = random.random()
        if do_translate > p:
            return img, depth
        x = random.randint(-max_t, max_t)
        y = random.randint(-max_t, max_t)
        M = np.float32([[1, 0, x], [0, 1, y]])
        # print(img.shape, depth.shape)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        feature_map = cv2.warpAffine(feature_map, M, (img.shape[1], img.shape[0]))
        depth = cv2.warpAffine(depth, M, (depth.shape[1], depth.shape[0]))
        depth = depth.squeeze()[..., None]  # add channel dim back. Affine warp removes it
        # print("after", img.shape, depth.shape)
        return img, feature_map, depth

    def train_preprocess(self, image, feature_map, depth_gt):
        if self.config.aug:
            # Random flipping
            do_flip = random.random()
            if do_flip > 0.5:
                image = (image[:, ::-1, :]).copy()
                feature_map = (feature_map[:, ::-1, :]).copy()
                depth_gt = (depth_gt[:, ::-1, :]).copy()

            # Random gamma, brightness, color augmentation
            do_augment = random.random()
            if do_augment > 0.5:
                image = self.augment_image(image)

        return image, feature_map, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.config.dataset == 'nyu' or self.config.dataset == 'nyu_sparse_feature':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode, do_normalize=True, size=None, use_ga = False, multiple_of = 1):
        self.mode = mode
        self.use_ga = use_ga
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if do_normalize else nn.Identity()
        # self.normalize = transforms.Normalize(
        #     mean=[0.171, 0.306, 0.167], std=[0.088, 0.156, 0.087]) if do_normalize else nn.Identity()
        # self.normalize = transforms.Normalize(
        #     mean=[0.138, 0.381, 0.449], std=[0.097, 0.239, 0.256]) if do_normalize else nn.Identity()
        self.sparse_depth_mean = 1.6908
        self.sparse_depth_std = 0.7952
        self.size = size
        print(size)
        if size is not None:
            # print("size is not none")
            self.resize = transforms.Resize(size=size)
        else:
            # print("size is none")
            self.resize = nn.Identity()

    def __call__(self, sample):
        image, focal, sparse_feature_map = sample['image'], sample['focal'],sample['sparse_map']
        # print(f"image shape before to tensor {np.shape(image)}")
        image = self.to_tensor(image)
        image_no_norm = self.resize(image)
        image = self.normalize(image)
        image = self.resize(image)
        # print(image.shape)
        # show_images_two_sources(image_normal.unsqueeze(0), image.unsqueeze(0))
        # print("Image normlaised")
        # print(f"image shape after to tensor {image.shape}")

        # print(f"image shape after to tensor: {image.shape}")

        # print(f"sparse_feature_map type: {type(sparse_feature_map)}")
        # print(f"sparse feature map before resize and totensor {sparse_feature_map.shape}")
        # to tesnor handles both PIL image and numpy array
        
        # sparse_feature_map[sparse_feature_map>0] = (sparse_feature_map[sparse_feature_map>0] - self.sparse_depth_mean) / self.sparse_depth_std
        # print(f"sparse_feature_map shape before to tensor {np.shape(sparse_feature_map)}")
        sparse_feature_map = self.to_tensor(sparse_feature_map)
        # print(f"sparse_feature_map shape after to tensor {sparse_feature_map.shape}")

        # print('before resize')
        # print(torch.max(sparse_feature_map))
        # print(torch.min(sparse_feature_map[sparse_feature_map>0]))
        # print(sparse_feature_map.shape)

        # if not self.use_ga:
        #     sparse_feature_map = self.resize(sparse_feature_map)

        # sparse_feature_map = sparse_feature_map.unsqueeze(1)
        # sparse_feature_map = nn.functional.interpolate(
        #             sparse_feature_map,
        #             size=self.size,
        #             mode="bilinear",
        #             align_corners=True,
        #         )
        # sparse_feature_map = sparse_feature_map.squeeze(1)

        # print('after resize')
        # print(torch.max(sparse_feature_map))
        # print(torch.min(sparse_feature_map[sparse_feature_map > 0]))
        # print(f"sparse_feature_map type after to tensor: {type(sparse_feature_map)}")
        # print(f"sparse_feature_map shape after to tensor: {sparse_feature_map.shape}")

        if self.mode == 'test':
            return {'image': image, 'sparse_map':sparse_feature_map,'focal': focal}

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            # print(f"GT shape after to tensor: {depth.shape}")
            return {**sample, 'image': image, 'depth': depth, 'sparse_map':sparse_feature_map, 'focal': focal}
        else:
            has_valid_depth = sample['has_valid_depth']
            # image = self.resize(image)
            return {**sample, 'image': image, 'image_no_norm':image_no_norm, 'depth': depth, 'focal': focal, 'sparse_map':sparse_feature_map, 'has_valid_depth': has_valid_depth,
                    'image_path': sample['image_path'], 'depth_path': sample['depth_path'], 'depth_path': sample['depth_path'], 'feature_path':sample['feature_path']}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
        

import numpy as np  
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
        im = axes[idx].imshow(tensor_images[idx], cmap='inferno')
        axes[idx].axis('off')
        # Add a colorbar to the current subplot
        fig.colorbar(im, ax=axes[idx])
        
    
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
        axes[idx][0].set_title("Image")
        axes[idx][0].axis('off')
        # plt.colorbar(im1, ax=axes[idx][0], fraction=0.046, pad=0.04)

        # Display source2 image
        im2 = axes[idx][1].imshow(images2[idx],cmap='inferno')
        axes[idx][1].set_title("Metric Depth Prediction")
        axes[idx][1].axis('off')
        # plt.colorbar(im2, ax=axes[idx][1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

