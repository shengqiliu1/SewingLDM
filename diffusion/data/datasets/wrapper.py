from argparse import Namespace
import json
import numpy as np
import random
import time
from datetime import datetime
import os

import torch
from torch.utils.data import DataLoader, Subset

from diffusion.data.datasets.pattern_converter import InvalidPatternDefError
import diffusion.data.datasets.transforms as transforms
from diffusion.data.builder import DATAWRAPPERS

def collate_fn(batches):
    if isinstance(batches[0]["ground_truth"], dict):
        bdict = {key: [] for key in batches[0].keys()}
        bdict["ground_truth"] = {key:[] for key in batches[0]["ground_truth"]}
        cum_sum = 56
        for i, batch in enumerate(batches):
            for key, val in batch.items():
                if key in ["image", "name", "data_folder", "img_fn"]:
                    bdict[key].append(val)
                else:
                    for k, v in batch["ground_truth"].items():
                        if k != "label_indices":
                            bdict["ground_truth"][k].append(v)
                        else:
                            new_label_indices = v.clone()
                            new_label_indices[:, :, 0] += cum_sum * i
                            bdict["ground_truth"][k].append(new_label_indices)
        
        for key in bdict.keys():
            if key == "image":
                bdict[key] = torch.stack(bdict[key])
            elif key == "ground_truth":
                for k in bdict[key]:
                    if k in ["label_indices", "masked_stitches", "stitch_edge_mask", "reindex_stitches"]:
                        bdict[key][k] = torch.vstack(bdict[key][k])
                    else:
                        bdict[key][k] = torch.stack(bdict[key][k])
        return bdict
    else:
        bdict = {key: [] for key in batches[0].keys()}
        for i, batch in enumerate(batches):
            for key, val in batch.items():
                bdict[key].append(val)
        bdict["features"] = torch.stack(bdict["features"])
        bdict["ground_truth"] = torch.stack(bdict["ground_truth"])
        return bdict

@DATAWRAPPERS.register_module()
class SewingLDMDatasetWrapper(object):
    """Resposible for keeping dataset, its splits, loaders & processing routines.
        Allows to reproduce earlier splits
    """
    def __init__(self, in_dataset, known_split=None, batch_size=None, validation_size=2, num_workers=16, shuffle_train=True):
        """Initialize wrapping around provided dataset. If splits/batch_size is known """

        self.dataset = in_dataset
        self.data_section_list = ['full', 'train', 'validation', 'test']

        self.training = in_dataset
        self.validation = None
        self.test = None
        self.full_per_datafolder = None

        self.batch_size = batch_size
        self.validation_size = validation_size

        self.loaders = Namespace(
            full=None,
            full_per_data_folder=None,
            train=None,
            test=None,
            test_per_data_folder=None,
            validation=None,
            valid_per_data_folder=None
        )

        self.split_info = {
            'random_seed': None, 
            'valid_per_type': None, 
            'test_per_type': None
        }

        if known_split is not None:
            self.load_split(known_split)
        if batch_size is not None:
            self.batch_size = batch_size
            self.new_loaders(batch_size, num_workers, shuffle_train)
        self.standardize_data()
    
    def get_loader(self, data_section='full'):
        """Return loader that corresponds to given data section. None if requested loader does not exist"""
        try:
            return getattr(self.loaders, data_section)
        except AttributeError:
            raise ValueError('DataWrapper::requested loader on unknown data section {}'.format(data_section))
        
    def load_split(self, split_info=None):
        """Get the split by provided parameters. Can be used to reproduce splits on the same dataset.
            NOTE this function re-initializes torch random number generator!
        """
        if split_info:
            self.split_info = split_info

        if 'random_seed' not in self.split_info or self.split_info['random_seed'] is None:
            self.split_info['random_seed'] = int(time.time())
        # init for all libs =)
        torch.manual_seed(self.split_info['random_seed'])
        random.seed(self.split_info['random_seed'])
        np.random.seed(self.split_info['random_seed'])

        # if file is provided
        if 'filename' in self.split_info and self.split_info['filename'] is not None:
            print('DataWrapper::Loading data split from {}'.format(self.split_info['filename']))
            with open(self.split_info['filename'], 'r') as f_json:
                split_dict = json.load(f_json)

            self.training, self.validation, self.test, self.training_per_datafolder, self.validation_per_datafolder, self.test_per_datafolder = self.dataset.split_from_dict(
                split_dict, 
                with_breakdown=True)

        print('DatasetWrapper::Dataset split: {} / {} / {}'.format(
            len(self.training) if self.training else None, 
            len(self.validation) if self.validation else None, 
            len(self.test) if self.test else None))
        self.split_info['size_train'] = len(self.training) if self.training else 0
        self.split_info['size_valid'] = len(self.validation) if self.validation else 0
        self.split_info['size_test'] = len(self.test) if self.test else 0
        
        self.print_subset_stats(self.training_per_datafolder, len(self.training), 'Training')
        self.print_subset_stats(self.validation_per_datafolder, len(self.validation), 'Validation')
        self.print_subset_stats(self.test_per_datafolder, len(self.test), 'Test')

        return self.training, self.validation, self.test

    def print_subset_stats(self, subset_breakdown_dict, total_len, subset_name='', log_to_config=True):
        """Print stats on the elements of each datafolder contained in given subset"""
        # gouped by data_folders
        if not total_len:
            print('{}::Warning::Subset {} is empty, no stats printed'.format(self.__class__.__name__, subset_name))
            return
        self.split_info[subset_name] = {}
        message = ''
        for data_folder, subset in subset_breakdown_dict.items():
            if log_to_config:
                self.split_info[subset_name][data_folder] = len(subset)
            message += '{} : {:.1f}%;\n'.format(data_folder, 100 * len(subset) / total_len)
        
        print('DatasetWrapper::{} subset breakdown::\n{}'.format(subset_name, message))

    def _loaders_dict(self, subsets_dict, batch_size, shuffle=False):
        """Create loaders for all subsets in dict"""
        loaders_dict = {}
        for name, subset in subsets_dict.items():
            loaders_dict[name] = DataLoader(subset, batch_size, shuffle=shuffle)
        return loaders_dict

    def new_loaders(self, batch_size=None, num_workers=16, shuffle_train=True, pin_memory=False):
        """Create loaders for current data split. Note that result depends on the random number generator!
        
            if the data split was not specified, only the 'full' loaders are created
        """
        if batch_size is not None:
            self.batch_size = batch_size
        if self.batch_size is None:
            raise RuntimeError('DataWrapper:Error:cannot create loaders: batch_size is not set')
        
        # if pin_memory:
        #     self.dataset.body_caching = False
        #     self.dataset.gt_caching = False
        #     self.dataset.feature_caching = False

        self.loaders.full = DataLoader(self.dataset,
                                       self.batch_size,
                                       shuffle=shuffle_train,
                                       num_workers=num_workers,
                                       pin_memory=pin_memory)
        if self.full_per_datafolder is None:
            self.full_per_datafolder = self.dataset.subsets_per_datafolder()
        self.loaders.full_per_data_folder = self._loaders_dict(self.full_per_datafolder, self.batch_size)

        if self.validation is not None and self.test is not None:
            self.loaders.train = DataLoader(self.training, self.batch_size, shuffle=shuffle_train, num_workers=num_workers, pin_memory=pin_memory)
            # no need for breakdown per datafolder for training -- for now

            self.loaders.validation = DataLoader(self.validation, self.validation_size, num_workers=num_workers, pin_memory=pin_memory)
            self.loaders.valid_per_data_folder = self._loaders_dict(self.validation_per_datafolder, self.validation_size)

            self.loaders.test = DataLoader(self.test, self.batch_size, num_workers=num_workers, pin_memory=pin_memory)
            self.loaders.test_per_data_folder = self._loaders_dict(self.test_per_datafolder, self.batch_size)

        return self.loaders.train, self.loaders.validation, self.loaders.test

    # ---------- Standardinzation ----------------
    def standardize_data(self):
        """Apply data normalization based on stats from training set"""
        self.dataset.standardize(range(len(self.dataset)))

    def save_single_prediction(self, preds, save_folder, name):
        save_to = os.path.join(save_folder, name)
        pattern = self.dataset._pred_to_pattern(preds, save_to=save_to, return_stitches=True)
        try: 
            final_dir = pattern.serialize(save_to, to_subfolder=True, tag='_predicted_')
        except (RuntimeError, InvalidPatternDefError, TypeError) as e:
            print('Garment3DPatternDataset::Error::{} serializing skipped: {}'.format(name, e))
        return final_dir