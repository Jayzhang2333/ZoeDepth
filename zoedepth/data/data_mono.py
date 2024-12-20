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

# This file is partly inspired from BTS (https://github.com/cleinc/bts/blob/master/pytorch/bts_dataloader.py); author: Jin Han Lee

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

def generate_feature_map_for_ga(feature_fp, original_height=240, original_width=320, new_height=480, new_width=640):
    # Read the CSV file
    df = pd.read_csv(feature_fp)

    # Initialize a blank depth map for the new image size with zeros
    sparse_depth_map = np.full((new_height, new_width), 0.0, dtype=np.float32)

    # Calculate scaling factors
    scale_y = new_height / original_height
    scale_x = new_width / original_width

    # Iterate through the dataframe and populate the depth map with scaled coordinates
    for index, row in df.iterrows():
        # Scale pixel coordinates to new image size
        pixel_row = int(row['row'] * scale_y)
        pixel_col = int(row.get('column', row.get('col', 0)) * scale_x)
        depth_value = float(row['depth'])

        # Ensure the scaled coordinates are within the bounds of the new image size
        if 0 <= pixel_row < new_height and 0 <= pixel_col < new_width:
            sparse_depth_map[pixel_row, pixel_col] = depth_value

    # print(np.shape(sparse_depth_map))
    # print('data mono')
    # print(np.max(sparse_depth_map))
    # print(np.min(sparse_depth_map[sparse_depth_map>0]))
    sparse_depth_map = sparse_depth_map[..., np.newaxis]
    return sparse_depth_map



def get_distance_maps(height, width, idcs_height, idcs_width):
    """Generates distance maps from feature points to every pixel."""
    num_features = len(idcs_height)
    dist_maps = np.empty((num_features, height, width))
    
    for idx in range(num_features):
        y, x = idcs_height[idx], idcs_width[idx]
        y_grid, x_grid = np.ogrid[:height, :width]
        dist_maps[idx] = np.sqrt((y_grid - y) ** 2 + (x_grid - x) ** 2)
    
    return dist_maps

def get_probability_maps(distance_maps):
    """Convert pixel distance to probability."""
    max_dist = np.sqrt(distance_maps.shape[-2] ** 2 + distance_maps.shape[-1] ** 2)
    probabilities = np.exp(-distance_maps / max_dist)
    return probabilities

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
    data[:,:,0] = data[:,:,0]/1.44
    return data


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode, **kwargs):
    return transforms.Compose([
        ToTensor(mode=mode, **kwargs)
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

        if "diode" in config.dataset:
            self.data = get_diode_loader(
                config[config.dataset+"_root"], batch_size=1, num_workers=1)
            return

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
            transform = preprocessing_transforms(mode, size=img_size)

        if mode == 'train':

            Dataset = DataLoadPreprocess
            self.training_samples = Dataset(
                config, mode, transform=transform, device=device)

            if config.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.training_samples)
            else:
                self.train_sampler = None

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
                                   shuffle=kwargs.get("shuffle_test", False),
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
            focal = float(sample_path.split()[2])
        else:
            focal = 500 #give a dummy value
        sample = {}

        if self.mode == 'train':
            if self.config.dataset == 'kitti' and self.config.use_right and random.random() > 0.5:
                image_path = os.path.join(
                    self.config.data_path, remove_leading_slash(sample_path.split()[3]))
                depth_path = os.path.join(
                    self.config.gt_path, remove_leading_slash(sample_path.split()[4]))
            else:
                image_path = os.path.join(
                    self.config.data_path, remove_leading_slash(sample_path.split()[0]))
                feature_path = os.path.join(
                    self.config.data_path, remove_leading_slash(sample_path.split()[-1]))
                depth_path = os.path.join(
                    self.config.gt_path, remove_leading_slash(sample_path.split()[1]))

            image = self.reader.open(image_path)
            # sparse feature map is numpy array
            # sparse_feature_map = generate_sparse_feature_map(featrue_path, self.config.prior_channels, self.config.sparse_feature_height, self.config.sparse_feature_width)
            sparse_feature_map = generate_sparse_feature_map_npy(feature_path, self.config.prior_channels)
            # sparse_feature_map = generate_feature_map_for_ga(feature_path, original_height=480, original_width=640, new_height=480, new_width=640)
            # print(np.shape(sparse_feature_map))
            depth_gt = self.reader.open(depth_path)
            w, h = image.size

            if self.config.do_kb_crop:
                height = image.height
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth_gt = depth_gt.crop(
                    (left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop(
                    (left_margin, top_margin, left_margin + 1216, top_margin + 352))
                sparse_feature_map = sparse_feature_map[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]

            # Avoid blank boundaries due to pixel registration?
            # Train images have white border. Test images have black border.
            if (self.config.dataset == 'nyu' or self.config.dataset == 'nyu_sparse_feature') and self.config.avoid_boundary:
                # print("Avoiding Blank Boundaries!")
                # We just crop and pad again with reflect padding to original size
                # original_size = image.size
                crop_params = get_white_border(np.array(image, dtype=np.uint8))
                image = image.crop((crop_params.left, crop_params.top, crop_params.right, crop_params.bottom))
                sparse_feature_map = sparse_feature_map[crop_params.top:crop_params.bottom, crop_params.left:crop_params.right, :]
                depth_gt = depth_gt.crop((crop_params.left, crop_params.top, crop_params.right, crop_params.bottom))

                # Use reflect padding to fill the blank
                image = np.array(image)
                image = np.pad(image, ((crop_params.top, h - crop_params.bottom), (crop_params.left, w - crop_params.right), (0, 0)), mode='reflect')
                image = Image.fromarray(image)

                # sparse_feature_map should already be a numpy array
                sparse_feature_map = np.pad(sparse_feature_map, ((crop_params.top, h - crop_params.bottom), (crop_params.left, w - crop_params.right)), 'constant', constant_values=0)
                # I am not sure if this needs to be converted into an image
                # sparse_feature_map = Image.fromarray(sparse_feature_map)

                depth_gt = np.array(depth_gt)
                depth_gt = np.pad(depth_gt, ((crop_params.top, h - crop_params.bottom), (crop_params.left, w - crop_params.right)), 'constant', constant_values=0)
                depth_gt = Image.fromarray(depth_gt)


            if self.config.do_random_rotate and (self.config.aug):
                random_angle = (random.random() - 0.5) * 2 * self.config.degree
                image = self.rotate_image(image, random_angle)
                # flag = 0 is the same as 'nearest'
                sparse_feature_map = self.rotate_feature_map(sparse_feature_map, random_angle, flag=0)
                depth_gt = self.rotate_image(
                    depth_gt, random_angle, flag=Image.NEAREST)

            image = np.asarray(image, dtype=np.float32) / 255.0
            # sparse_feature_map = np.asarray(sparse_feature_map, dtype=np.float32)
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)

            if self.config.dataset == 'nyu' or self.config.dataset == 'nyu_sparse_feature':
                depth_gt = depth_gt / 1000.0
            else:
                depth_gt = depth_gt / 256.0

            if self.config.aug and (self.config.random_crop):
                image, sparse_feature_map, depth_gt = self.random_crop(
                    image, sparse_feature_map, depth_gt, self.config.input_height, self.config.input_width)
            
            if self.config.aug and self.config.random_translate:
                # print("Random Translation!")
                image, sparse_feature_map, depth_gt = self.random_translate(image, sparse_feature_map, depth_gt, self.config.max_translation)

            image, sparse_feature_map, depth_gt = self.train_preprocess(image, sparse_feature_map, depth_gt)
            mask = np.logical_and(depth_gt > self.config.min_depth,
                                  depth_gt < self.config.max_depth).squeeze()[None, ...]
            sample = {'image': image, 'sparse_map': sparse_feature_map , 'depth': depth_gt, 'focal': focal,
                      'mask': mask, **sample}

        else:
            if self.mode == 'online_eval':
                data_path = self.config.data_path_eval
            else:
                data_path = self.config.data_path

            image_path = os.path.join(
                data_path, remove_leading_slash(sample_path.split()[0]))
            
            feature_path = os.path.join(
                data_path, remove_leading_slash(sample_path.split()[-1]))
            
            # feature_path = feature_path.replace("matched_features", "matched_features_centreline")
            
            # print(sample_path.split()[0])
            image = np.asarray(self.reader.open(image_path),
                               dtype=np.float32) / 255.0

            # sparse_feature_map = generate_sparse_feature_map(feature_path, self.config.prior_channels, self.config.sparse_feature_height, self.config.sparse_feature_width)
            sparse_feature_map = generate_sparse_feature_map_npy(feature_path, self.config.prior_channels)
            # sparse_feature_map = generate_feature_map_for_ga(feature_path, original_height=480, original_width=640, new_height=480, new_width=640)
            if self.mode == 'online_eval':
                gt_path = self.config.gt_path_eval
                depth_path = os.path.join(
                    gt_path, remove_leading_slash(sample_path.split()[1]))
                has_valid_depth = False
                try:
                    depth_gt = self.reader.open(depth_path)
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    # print('Missing gt for {}'.format(image_path))

                if has_valid_depth:
                    depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    
                    if self.config.dataset == 'nyu' or self.config.dataset == 'nyu_sparse_feature':
                        depth_gt = depth_gt / 1000.0
                    elif self.config.dataset == "flsea_sparse_feature" or self.config.dataset == "lizard_sparse_feature":
                        depth_gt = depth_gt / 1.0

                    else:
                        depth_gt = depth_gt / 256.0

                    mask = np.logical_and(
                        depth_gt >= self.config.min_depth, depth_gt <= self.config.max_depth).squeeze()[None, ...]
                    
                    # if np.all(mask == False):
                    #     print("The mask is an array of all zeros.")
                    # else:
                    #     print("The mask contains non-zero values.")
                    
                else:
                    mask = False

            if self.config.do_kb_crop:
                height = image.shape[0]
                width = image.shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                image = image[top_margin:top_margin + 352,
                              left_margin:left_margin + 1216, :]
                
                sparse_feature_map = sparse_feature_map[top_margin:top_margin + 352,
                              left_margin:left_margin + 1216, :]
                
                if self.mode == 'online_eval' and has_valid_depth:
                    depth_gt = depth_gt[top_margin:top_margin +
                                        352, left_margin:left_margin + 1216, :]

            if self.mode == 'online_eval':
                sample = {'image': image,'sparse_map': sparse_feature_map, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth,
                          'image_path': sample_path.split()[0], 'feature_path': sample_path.split()[-1],'depth_path': sample_path.split()[1],
                          'mask': mask}
            else:
                sample = {'image': image, 'sparse_map': sparse_feature_map,'focal': focal}

        if (self.mode == 'train') or ('has_valid_depth' in sample and sample['has_valid_depth']):
            mask = np.logical_and(depth_gt > self.config.min_depth,
                                  depth_gt < self.config.max_depth).squeeze()[None, ...]
            sample['mask'] = mask

        
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
    def __init__(self, mode, do_normalize=False, size=None):
        self.mode = mode
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if do_normalize else nn.Identity()
        self.size = size
        # print(size)
        if size is not None:
            # print("size is not none")
            self.resize = transforms.Resize(size=size)
        else:
            # print("size is none")
            self.resize = nn.Identity()

    def __call__(self, sample):
        image, focal, sparse_feature_map = sample['image'], sample['focal'],sample['sparse_map']
        image = self.to_tensor(image)
        image = self.normalize(image)
        image = self.resize(image)

        # print(f"image shape after to tensor: {image.shape}")

        # print(f"sparse_feature_map type: {type(sparse_feature_map)}")
        # print(f"sparse feature map before resize and totensor {sparse_feature_map.shape}")
        # to tesnor handles both PIL image and numpy array
        

        sparse_feature_map = self.to_tensor(sparse_feature_map)

        # print('before resize')
        # print(torch.max(sparse_feature_map))
        # print(torch.min(sparse_feature_map[sparse_feature_map>0]))
        # print(sparse_feature_map.shape)

        
        sparse_feature_map = self.resize(sparse_feature_map)

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
            image = self.resize(image)
            return {**sample, 'image': image, 'depth': depth, 'focal': focal, 'sparse_map':sparse_feature_map, 'has_valid_depth': has_valid_depth,
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
