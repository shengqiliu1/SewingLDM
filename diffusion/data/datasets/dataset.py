import json
import numpy as np
import os
from pathlib import Path, PureWindowsPath
import shutil
import glob
from PIL import Image, ImageDraw, ImageOps
import random
import time
import yaml

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.io import read_image
import torchvision.transforms as T

from diffusion.data.builder import get_data_path, DATASETS
from diffusion.data.datasets.pattern_converter import NNSewingPattern, InvalidPatternDefError
import diffusion.data.datasets.transforms as transforms
from diffusion.data.datasets.panel_classes import PanelClasses
from diffusion.data.datasets.utils import euler_angle_to_rot_6d

@DATASETS.register_module()
class GarmentDetrDataset(Dataset):
    def __init__(self, root_dir, start_config, data_json=None, gt_caching=False, feature_caching=False, in_transforms=[], **kwargs):

        self.root_path = root_dir
        self.load_clip_feat = kwargs.get("load_clip_feat", False)
        self.load_tokenizer_feat = kwargs.get("load_tokenizer_feat", False)
        self.load_pattern_vector = kwargs.get("load_pattern_vector", False)
        self.load_text_feat = kwargs.get("load_text_feat", False)
        self.load_body_params = kwargs.get("load_body_params", False)
        self.load_sketch = kwargs.get("load_sketch", False)
        self.condition = kwargs.get("condition", 'text')
        self.load_probability = kwargs.get("load_probability", 0)
        self.max_length = kwargs.get("max_length", 100)
        self.sketch_threshold_range = kwargs.get("sketch_threshold_range", [150, 200])
        
        self.config = {}
        pattern_size_initialized = self.update_config(start_config)
        self.config['class'] = self.__class__.__name__
        self.datapoints_names = []
        self.dataset_start_ids = []  # (folder, start_id) tuples -- ordered by start id
        folders = [folder for folder in os.listdir(self.root_path) if os.path.isdir(os.path.join(self.root_path, folder))]
        self.data_folders = [os.path.join(self.root_path, folder, "random_body") for folder in folders]
        self.data_folders_nicknames = dict(zip(self.data_folders, self.data_folders))

        if data_json is None:
            for folder in folders:
                self.dataset_start_ids.append((folder, len(self.datapoints_names)))
                body_dir = os.path.join(self.root_path, folder, "random_body")
                for cloth_dir in os.listdir(body_dir):
                    if os.path.isdir(os.path.join(body_dir, cloth_dir)):
                        gt_folder = os.path.join(self.root_path, folder, "random_body", cloth_dir)
                        img_name = [os.path.join(gt_folder, fn) for fn in os.listdir(gt_folder) if "render_front.png" in fn][0]
                        body_yaml = [os.path.join(gt_folder, fn) for fn in os.listdir(gt_folder) if "body_measurements.yaml" in fn][0]

                        merge_names = [(img_name, body_yaml, gt_folder)]
                        self.datapoints_names += merge_names
        else:
            data_info = json.load(open(data_json, "r"))
            for folder, info in data_info.items():
                folder = os.path.join(self.root_path, folder)
                self.dataset_start_ids.append((folder, len(self.datapoints_names)))
                for cloth_dir in info.keys():
                    gt_folder = os.path.join(folder, cloth_dir)
                    img_name = [os.path.join(gt_folder, fn) for fn in os.listdir(gt_folder) if "render_front.png" in fn][0]
                    body_name = info[cloth_dir]["body_name"]
                    merge_names = [np.array([img_name, body_name, gt_folder]).astype(str)]
                    self.datapoints_names += merge_names
        
        self.datapoints_names = np.array(self.datapoints_names)
        self.dataset_start_ids.append((None, len(self.datapoints_names)))
        self.config['size'] = len(self)
        print("GarmentDetrDataset::Info::Total valid datanames is {}".format(self.config['size']))

        self.body_cached = {}
        self.body_caching  = kwargs.get("body_caching", False)
        self.gt_cached, self.gt_caching = {}, gt_caching
        self.feature_cached, self.feature_caching = {}, feature_caching
        if self.gt_caching:
            print('GarmentDetrDataset::Info::Storing datapoints ground_truth info in memory')
        if self.feature_caching:
            print('GarmentDetrDataset::Info::Storing datapoints feature info in memory')
        
        if not self.load_clip_feat:
            self.color_transform = transforms.tv_make_color_img_transforms()
            self.geo_tranform = transforms.tv_make_geo_img_transforms(color=255)
            self.img_transform = transforms.tv_make_img_transforms()

        # Use default tensor transform + the ones from input
        self.transforms = [transforms.SampleToTensor()] + in_transforms
        
        self.is_train = False
        self.gt_jsons = {"spec_dict":{}, "specs":{}}

        # Load panel classifier
        if self.config['panel_classification'] is not None:
            self.panel_classifier = PanelClasses(self.config['panel_classification'])
            assert self.config['max_pattern_len']==len(self.panel_classifier)
        else:
            raise RuntimeError('GarmentDetrDataset::Error::panel_classification not found')

    def standardize(self, training=None):
        """Use shifting and scaling for fitting data to interval comfortable for NN training.
            Accepts either of two inputs: 
            * training subset to calculate the data statistics -- the stats are only based on training subsection of the data
            * if stats info is already defined in config, it's used instead of calculating new statistics (usually when calling to restore dataset from existing experiment)
            configuration has a priority: if it's given, the statistics are NOT recalculated even if training set is provided
                => speed-up by providing stats or speeding up multiple calls to this function
        """
        print('GarmentDetrDataset::Using data normalization for features & ground truth')
        if 'standardize' in self.config:
            print('{}::Using stats from config'.format(self.__class__.__name__))
            stats = self.config['standardize']
        else:
            raise ValueError('GarmentDetrDataset::Error::Standardization cannot be applied: supply either stats in config or training set to use standardization')
        print(stats)
        for key in stats.keys():
            for k in stats[key].keys():
                stats[key][k] = np.array(stats[key][k]).astype(np.float32)
        
        # clean-up tranform list to avoid duplicates
        self.transforms = [t for t in self.transforms if not isinstance(t, transforms.GTtandartization) and not isinstance(t, transforms.FeatureStandartization)]

        if not self.load_tokenizer_feat:
            self.transforms.append(transforms.GTtandartization(stats['gt_shift'], stats['gt_scale']))

    def save_to_wandb(self, experiment):
        """Save data cofiguration to current expetiment run"""
        # config
        experiment.add_config('dataset', self.config)
        # panel classes
        if self.panel_classifier is not None:
            shutil.copy(
                self.panel_classifier.filename, 
                experiment.local_wandb_path() / ('panel_classes.json'))
    
    def set_training(self, is_train=True):
        self.is_train = is_train
    
    def update_config(self, in_config):
        """Define dataset configuration:
            * to be part of experimental setup on wandb
            * Control obtainign values for datapoints"""

        # initialize keys for correct dataset initialization
        if ('max_pattern_len' not in in_config
                or 'max_panel_len' not in in_config
                or 'max_num_stitches' not in in_config):
            in_config.update(max_pattern_len=None, max_panel_len=None, max_num_stitches=None)
            pattern_size_initialized = False
        else:
            pattern_size_initialized = True
        
        if 'obj_filetag' not in in_config:
            in_config['obj_filetag'] = ''  # look for objects with this tag in filename when loading 3D models

        if 'panel_classification' not in in_config:
            in_config['panel_classification'] = None
        
        self.config.update(in_config)
        # check the correctness of provided list of datasets
        if ('data_folders' not in self.config 
                or not isinstance(self.config['data_folders'], list)
                or len(self.config['data_folders']) == 0):
            print(f'{self.__class__.__name__}::Info::Collecting all datasets (no sub-folders) to use')
        return pattern_size_initialized

    def __len__(self, ):
        """Number of entries in the dataset"""
        return len(self.datapoints_names) 

    def __getitem__(self, idx):
        """Called when indexing: read the corresponding data. 
        Does not support list indexing"""
        if torch.is_tensor(idx):  # allow indexing by tensors
            idx = idx.tolist()

        datapoint_name = self.datapoints_names[idx][0]
        body_name = self.datapoints_names[idx][1]
        gt_folder = self.datapoints_names[idx][2]
        name = os.path.basename(gt_folder)
        folder = os.path.dirname(gt_folder)

        condition, ground_truth, body_params, sketch = self._get_sample_info(datapoint_name, gt_folder, body_name)
        
        # stitches
        if "stitch_adj" in ground_truth.keys():
            masked_stitches, stitch_edge_mask, reindex_stitches = self.match_edges(ground_truth["free_edges_mask"], \
                                                                                stitches=ground_truth["stitches"], \
                                                                                max_num_stitch_edges=self.config["max_stitch_edges"])
            label_indices = self.split_pos_neg_pairs(reindex_stitches, num_max_edges=1000)

            ground_truth.update({"masked_stitches": masked_stitches,
                                 "stitch_edge_mask": stitch_edge_mask,
                                 "reindex_stitches": reindex_stitches,
                                 "label_indices": label_indices})
        
        sample = {
                'ground_truth': ground_truth, 
                'body_params': body_params,
                'sketch': sketch,
                'condition': condition,
                'name': name, 
                'body_name': body_name,
                'data_folder': folder, 
                'img_fn': datapoint_name}
        for transform in self.transforms:
            sample = transform(sample)
        return sample
    
    def indices_by_data_folder(self, index_list):
        """
            Separate provided indices according to dataset folders used in current dataset
        """
        ids_dict = dict.fromkeys(self.data_folders)  # consists of elemens of index_list
        ids_mapping_dict = dict.fromkeys(self.data_folders)  # reference to the elements in index_list
        index_list = np.array(index_list)
        
        # assign by comparing with data_folders start & end ids
        # enforce sort Just in case
        self.dataset_start_ids = sorted(self.dataset_start_ids, key=lambda idx: idx[1])

        for i in range(0, len(self.dataset_start_ids) - 1):
            ids_filter = (index_list >= self.dataset_start_ids[i][1]) & (index_list < self.dataset_start_ids[i + 1][1])
            ids_dict[self.dataset_start_ids[i][0]] = index_list[ids_filter]
            ids_mapping_dict[self.dataset_start_ids[i][0]] = np.flatnonzero(ids_filter)
        
        return ids_dict, ids_mapping_dict
    
    def subsets_per_datafolder(self, index_list=None):
        """
            Group given indices by datafolder and Return dictionary with Subset objects for each group.
            if None, a breakdown for the full dataset is given
        """
        if index_list is None:
            index_list = range(len(self))
        per_data, _ = self.indices_by_data_folder(index_list)
        breakdown = {}
        for folder, ids_list in per_data.items():
            breakdown[self.data_folders_nicknames[folder]] = Subset(self, ids_list)
        return breakdown

    def split_from_dict(self, split_dict, with_breakdown=False):
        """
            Reproduce the data split in the provided dictionary: 
            the elements of the currect dataset should play the same role as in provided dict
        """
        train_ids, valid_ids, test_ids = [], [], []
        train_breakdown, valid_breakdown, test_breakdown = {}, {}, {}

        training_datanames = set(split_dict['training'])
        valid_datanames = set(split_dict['validation'])
        test_datanames = set(split_dict['test'])
        
        for idx in range(len(self.datapoints_names)):
            data_name = '/'.join(self.datapoints_names[idx][2].split('/')[-3:])
            default_data_name = data_name.replace("random_body", "default_body")
            if data_name in training_datanames or default_data_name in training_datanames:  # usually the largest, so check first
                train_ids.append(idx)
            elif len(test_datanames) > 0 and (data_name in test_datanames or default_data_name in test_datanames):
                test_ids.append(idx)
            elif len(valid_datanames) > 0 and (data_name in valid_datanames or default_data_name in valid_datanames):
                valid_ids.append(idx)
            # othervise, just skip

        if with_breakdown:
            train_breakdown = self.subsets_per_datafolder(train_ids)
            valid_breakdown = self.subsets_per_datafolder(valid_ids)
            test_breakdown = self.subsets_per_datafolder(test_ids)

            return (
                Subset(self, train_ids), 
                Subset(self, valid_ids),
                Subset(self, test_ids) if len(test_ids) > 0 else None,
                train_breakdown, valid_breakdown, test_breakdown
            )

        return (
            Subset(self, train_ids), 
            Subset(self, valid_ids),
            Subset(self, test_ids) if len(test_ids) > 0 else None
        )
    
    def _load_gt_folders_from_indices(self, indices):
        gt_folders = [self.datapoints_names[idx][-1] for idx in indices]
        return list(set(gt_folders))
    
    def _drop_cache(self):
        """Clean caches of datapoints info"""
        self.gt_cached = {}
        self.feature_cached = {}
    
    def _renew_cache(self):
        """Flush the cache and re-fill it with updated information if any kind of caching is enabled"""
        self.gt_cached = {}
        self.feature_cached = {}
        if self.feature_caching or self.gt_caching:
            for i in range(len(self)):
                self[i]
            print('Data cached!')

    # ----- Sample -----
    def _get_sample_info(self, datapoint_name, gt_folder, body_name=None):
        """
            Get features and Ground truth prediction for requested data example
        """
        folder_elements = [os.path.basename(file) for file in glob.glob(os.path.join(gt_folder, "*"))]  # all files in this directory
        if datapoint_name in self.feature_cached:
            condition = self.feature_cached[datapoint_name]
        else:
            if self.condition == 'text':
                if self.load_text_feat:
                    text_path = datapoint_name.replace("render_front.png", "caption.npz")
                    txt_info = dict(np.load(text_path))
                    txt_fea = torch.from_numpy(txt_info['caption_feature']).to(torch.float16)     # 1xTx4096
                    attention_mask = torch.ones(1, 1, txt_fea.shape[1]).to(torch.int16)     # 1x1xT
                    if 'attention_mask' in txt_info.keys():
                        attention_mask = torch.from_numpy(txt_info['attention_mask'])[None].to(torch.int16) 
                    if txt_fea.shape[1] != self.max_length:
                        txt_fea = torch.cat([txt_fea, txt_fea[:, -1:].repeat(1, self.max_length-txt_fea.shape[1], 1)], dim=1)
                        attention_mask = torch.cat([attention_mask, torch.zeros(1, 1, self.max_length-attention_mask.shape[-1])], dim=-1)
                    condition = {'caption_feature': txt_fea, 'attention_mask': attention_mask}
                else:
                    text_path = datapoint_name.replace("render_front.png", "caption.txt")
                    with open(text_path, 'r') as f:
                        condition = {'caption': [item.strip() for item in f.readlines()][0]}
            else:
                condition = None
            if self.feature_caching:
                self.feature_cached[datapoint_name] = condition

        # GT -- pattern
        if gt_folder in self.gt_cached: # might not be compatible with list indexing
            ground_truth = self.gt_cached[gt_folder]
        else:
            if self.load_tokenizer_feat:
                latent_path = datapoint_name.replace("render_front.png", "latent.npy")
                ground_truth = {'latent': np.load(latent_path)}
            elif self.load_pattern_vector:
                pattern_vector_path = datapoint_name.replace("render_front.png", "pattern_vector.npz")
                ground_truth = dict(np.load(pattern_vector_path))
            else:
                ground_truth = self._get_pattern_ground_truth(gt_folder, folder_elements)
            if self.gt_caching:
                self.gt_cached[gt_folder] = ground_truth
        
        load_probability = self.load_probability
        load_text = True
        # Body parameters
        if self.load_body_params:
            if body_name in self.body_cached:
                body_params = self.body_cached[body_name]
            else:
                body_path = os.path.join(self.config['body_root'], body_name[:5]+"_meta.json")
                with open(body_path, "r") as f_json:
                    body_params = json.load(f_json)['pca_weights']
                body_params = np.array(body_params).astype(np.float32)
            if self.body_caching:
                self.body_cached[body_name] = body_params
        else:
            body_params = None
        
        if self.load_sketch:
            sketch_threshold = random.randint(self.sketch_threshold_range[0], self.sketch_threshold_range[1])
            sketch_path = datapoint_name.replace("render_front.png", "front_sketch.png")
            if self.is_train and random.random() > load_probability:
                sketch_front = torch.zeros(1, 800, 800)
            elif os.path.exists(sketch_path):
                sketch_front = Image.open(sketch_path)
                sketch_front = ImageOps.invert(sketch_front)
                # threshold grayscale pil image
                sketch_front = sketch_front.point(lambda p: 255 if p > sketch_threshold else 0)
                sketch_front = T.functional.to_tensor(sketch_front)
                if random.random() < load_probability:
                    load_text = False
            else:
                sketch_front = torch.zeros(1, 800, 800)
            sketch_path = datapoint_name.replace("render_front.png", "back_sketch.png")
            if self.is_train and random.random() > load_probability:
                sketch_back = torch.zeros(1, 800, 800)
            elif os.path.exists(sketch_path):
                sketch_back = Image.open(sketch_path)
                sketch_back = ImageOps.invert(sketch_back)
                # threshold grayscale pil image
                sketch_back = sketch_back.point(lambda p: 255 if p > sketch_threshold else 0)
                sketch_back = T.functional.to_tensor(sketch_back)
                if random.random() < load_probability:
                    load_text = False
            else:
                sketch_back = torch.zeros(1, 800, 800)
            sketch = torch.cat([sketch_front, sketch_back], dim=0)
            sketch = torchvision.transforms.functional.resize(
                sketch, size=(512, 512),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                antialias=True)
        else:
            sketch = None
        if not load_text and self.is_train:
            if self.load_text_feat:
                text_fea = torch.load(f'output/pretrained_models/null_embed_diffusers_{self.max_length}token.pth')['uncond_prompt_embeds'].cpu().detach()
                attention_mask = torch.ones(1, 1, text_fea.shape[1]).to(torch.int16)
                attention_mask = torch.ones_like(attention_mask).to(torch.int16) 
                condition = {'caption_feature': text_fea, 'attention_mask': attention_mask}
            else:
                condition = {'caption': ''}

        return condition, ground_truth, body_params, sketch
        
    def _get_pattern_ground_truth(self, gt_folder, folder_elements):
        """Get the pattern representation with 3D placement"""
        patterns = self._read_pattern(
            gt_folder, folder_elements,
            pad_panels_to_len=self.config['max_panel_len'],
            pad_panel_num=self.config['max_pattern_len'],
            pad_stitches_num=self.config['max_num_stitches'],
            with_placement=True, with_stitches=True, with_stitch_tags=True)
        pattern, num_edges, num_panels, rots, tranls, stitches, num_stitches, stitch_tags, stitch_per_edge, reverse_stitch = patterns 
        free_edges_mask = self.free_edges_mask(pattern, stitches, num_stitches)
        assert ~np.all(stitch_per_edge[free_edges_mask])
        empty_edges_mask, empty_panels_mask = self._empty_edges_panels_mask(num_edges)  # useful for evaluation
        edge_type = pattern[:, :, 9:]
        
        ground_truth ={
            'outlines': pattern[:, :, :9], # max_panel_num, max_edge_num, 9-element_size
            'num_edges': num_edges,
            'rotations': rots, # max_panel_num, 4
            'translations': tranls, # max_panel_num, 3
            'num_panels': num_panels, # real panel num
            'empty_panels_mask': empty_panels_mask, # max_panel_num
            'num_stitches': num_stitches, # real stitches num
            'stitches': stitches, # 2, max_stitch_edges
            'edge_type': edge_type, # max_panel_num, max_edge_num, 4 + 3
            'free_edges_mask': free_edges_mask, # max_panel_num, max_edge_num
            'reverse_stitch': reverse_stitch, # max_panel_num, max_edge_num
            'stitch_tags': stitch_tags,  # max_panel_num, max_edge_num, 3
            'empty_edges_mask': empty_edges_mask, # max_panel_num, max_edge_num
        }

        return ground_truth
    
    def _load_ground_truth(self, gt_folder):
        folder_elements = [os.path.basename(file) for file in glob.glob(os.path.join(gt_folder, "*"))] 
        spec_dict = self._load_spec_dict(gt_folder)
        ground_truth = self._get_pattern_ground_truth(gt_folder, folder_elements, spec_dict)
        return ground_truth
    
    def _empty_edges_panels_mask(self, num_edges):
        """Empty panels as boolean mask"""

        empty_edges_mask = np.ones([len(num_edges), self.config['max_panel_len']], dtype=bool)
        empty_panels_mask = np.zeros([len(num_edges)], dtype=bool)
        for i, num in enumerate(num_edges):
            empty_edges_mask[i][:num] = False
        
        empty_panels_mask[num_edges == 0] = True

        return empty_edges_mask, empty_panels_mask

     # ----- Stitches tools -----
    
    @staticmethod
    def tags_to_stitches(stitch_tags, free_edges_score):
        """
        Convert per-edge per panel stitch tags into the list of connected edge pairs
        NOTE: expects inputs to be torch tensors, numpy is not supported
        """
        flat_tags = stitch_tags.contiguous().view(-1, stitch_tags.shape[-1])  # with pattern-level edge ids
        
        # to edge classes from logits
        flat_edges_score = free_edges_score.view(-1) 
        flat_edges_mask = torch.round(torch.sigmoid(flat_edges_score)).type(torch.BoolTensor)

        # filter free edges
        non_free_mask = ~flat_edges_mask
        non_free_edges = torch.nonzero(non_free_mask, as_tuple=False).squeeze(-1)  # mapping of non-free-edges ids to full edges list id
        if not any(non_free_mask) or non_free_edges.shape[0] < 2:  # -> no stitches
            print('Garment3DPatternFullDataset::Warning::no non-zero stitch tags detected')
            return torch.tensor([])

        # Check for even number of tags
        if len(non_free_edges) % 2:  # odd => at least one of tags is erroneously non-free
            # -> remove the edge that is closest to free edges class from comparison
            to_remove = flat_edges_score[non_free_mask].argmax()  # the higer the score, the closer the edge is to free edges
            non_free_mask[non_free_edges[to_remove]] = False
            non_free_edges = torch.nonzero(non_free_mask, as_tuple=False).squeeze(-1)

        # Now we have even number of tags to match
        num_non_free = len(non_free_edges) 
        dist_matrix = torch.cdist(flat_tags[non_free_mask].type(torch.float32), flat_tags[non_free_mask].type(torch.float32))

        # remove self-distance on diagonal & lower triangle elements (duplicates)
        tril_ids = torch.tril_indices(num_non_free, num_non_free)
        dist_matrix[tril_ids[0], tril_ids[1]] = float('inf')

        # pair egdes by min distance to each other starting by the closest pair
        stitches = []
        for _ in range(num_non_free // 2):  # this many pair to arrange
            to_match_idx = dist_matrix.argmin()  # current global min is also a best match for the pair it's calculated for!
            row = to_match_idx // dist_matrix.shape[0]
            col = to_match_idx % dist_matrix.shape[0]
            stitches.append([non_free_edges[row], non_free_edges[col]])

            # exlude distances with matched edges from further consideration
            dist_matrix[row, :] = float('inf')
            dist_matrix[:, row] = float('inf')
            dist_matrix[:, col] = float('inf')
            dist_matrix[col, :] = float('inf')
        
        if torch.isfinite(dist_matrix).any():
            raise ValueError('Garment3DPatternFullDataset::Error::Tags-to-stitches::Number of stitches {} & dist_matrix shape {} mismatch'.format(
                num_non_free / 2, dist_matrix.shape))

        return torch.tensor(stitches).transpose(0, 1).to(stitch_tags.device) if len(stitches) > 0 else torch.tensor([])

    @staticmethod
    def match_edges(free_edge_mask, stitches=None, max_num_stitch_edges=56):
        stitch_edges = np.ones((1, max_num_stitch_edges)) * (-1)
        valid_edges = (~free_edge_mask.reshape(-1)).nonzero()
        stitch_edge_mask = np.zeros((1, max_num_stitch_edges))
        if stitches is not None:
            stitches = np.transpose(stitches)
            reindex_stitches = np.zeros((1, max_num_stitch_edges, max_num_stitch_edges))
        else:
            reindex_stitches = None
        
        batch_edges = valid_edges[0]
        num_edges = batch_edges.shape[0]
        stitch_edges[:, :num_edges] = batch_edges
        stitch_edge_mask[:, :num_edges] = 1
        if stitches is not None:
            for stitch in stitches:
                side_i, side_j = stitch
                if side_i != -1 and side_j != -1:
                    reindex_i, reindex_j = np.where(stitch_edges[0] == side_i)[0], np.where(stitch_edges[0] == side_j)[0]
                    reindex_stitches[0, reindex_i, reindex_j] = 1
                    reindex_stitches[0, reindex_j, reindex_i] = 1
        
        return stitch_edges * stitch_edge_mask, stitch_edge_mask, reindex_stitches
    
    @staticmethod
    def split_pos_neg_pairs(stitches, num_max_edges=3000):
        stitch_ind = np.triu_indices_from(stitches[0], 1)
        pos_ind = [[stitch_ind[0][i], stitch_ind[1][i]] for i in range(stitch_ind[0].shape[0]) if stitches[0, stitch_ind[0][i], stitch_ind[1][i]]]
        neg_ind = [[stitch_ind[0][i], stitch_ind[1][i]] for i in range(stitch_ind[0].shape[0]) if not stitches[0, stitch_ind[0][i], stitch_ind[1][i]]]

        assert len(neg_ind) >= num_max_edges
        neg_ind = neg_ind[:num_max_edges]
        pos_inds = np.expand_dims(np.array(pos_ind), axis=1)
        neg_inds = np.repeat(np.expand_dims(np.array(neg_ind), axis=0), repeats=pos_inds.shape[0], axis=0)
        indices = np.concatenate((pos_inds, neg_inds), axis=1)
        return indices
    
    def _read_pattern(self, gt_folder, folder_elements,
                      pad_panels_to_len=None, pad_panel_num=None, pad_stitches_num=None,
                      with_placement=False, with_stitches=False, with_stitch_tags=False):
        """Read given pattern in tensor representation from file"""

        spec_file = [file for file in folder_elements if "specification.json" in file]
        if len(spec_file) > 0:
            spec_file = spec_file[0]
        else:
            raise ValueError("Specification Cannot be found in folder_elements for {}".format(gt_folder))
        
        if not spec_file:
            raise RuntimeError('GarmentDetrDataset::Error::*specification.json not found for {}'.format(gt_folder))

        if gt_folder + "/" + spec_file in self.gt_jsons["specs"]:
            pattern = self.gt_jsons["specs"][gt_folder + "/" + spec_file]
        else:
            pattern = NNSewingPattern(
                gt_folder + "/" + spec_file, 
                panel_classifier=self.panel_classifier)
            self.gt_jsons["specs"][gt_folder + "/" + spec_file] = pattern

        pat_tensor = pattern.pattern_as_tensors(
            pad_panels_to_len, pad_panels_num=pad_panel_num, pad_stitches_num=pad_stitches_num,
            with_placement=with_placement, with_stitches=with_stitches, 
            with_stitch_tags=with_stitch_tags)
        return pat_tensor
    
    
    def get_item_infos(self, idx):
        if torch.is_tensor(idx):  # allow indexing by tensors
            idx = idx.tolist()
        datapoint_name, smpl_name, gt_folder = self.datapoints_names[idx]
        data_prop_fn = os.path.join(os.path.dirname(gt_folder), "data_props.json")
        with open(data_prop_fn, 'r') as f:
            data_props = json.load(f)
        pose_fbx = data_props["body"]["name"].replace(data_props["body_path"] + "\\", "")
        spec_fns = list(data_props["garments"]["config"]["pattern_specs"].values())
        return pose_fbx, spec_fns, (datapoint_name, gt_folder)
    
    def _unpad(self, element, tolerance=1.e-5):
        """Return copy of input element without padding from given element. Used to unpad edge sequences in pattern-oriented datasets"""
        # NOTE: might be some false removal of zero edges in the middle of the list.
        if torch.is_tensor(element):        
            bool_matrix = torch.isclose(element, torch.zeros_like(element), atol=tolerance)  # per-element comparison with zero
            selection = ~torch.all(bool_matrix, axis=1)  # only non-zero rows
        else:  # numpy
            selection = ~np.all(np.isclose(element, 0, atol=tolerance), axis=1)  # only non-zero rows
        return element[selection]
    
    def _get_distribution_stats(self, input_batch, padded=False):
        """Calculates mean & std values for the input tenzor along the last dimention"""

        input_batch = input_batch.view(-1, input_batch.shape[-1])
        if padded:
            input_batch = self._unpad(input_batch)  # remove rows with zeros

        # per dimention means
        mean = input_batch.mean(axis=0)
        # per dimention stds
        stds = ((input_batch - mean) ** 2).sum(0)
        stds = torch.sqrt(stds / input_batch.shape[0])

        return mean, stds
    
    def _get_norm_stats(self, input_batch, padded=False):
        """Calculate shift & scaling values needed to normalize input tenzor 
            along the last dimention to [0, 1] range"""
        input_batch = input_batch.view(-1, input_batch.shape[-1])
        if padded:
            input_batch = self._unpad(input_batch)  # remove rows with zeros

        # per dimention info
        min_vector, _ = torch.min(input_batch, dim=0)
        max_vector, _ = torch.max(input_batch, dim=0)
        scale = torch.empty_like(min_vector)

        # avoid division by zero
        for idx, (tmin, tmax) in enumerate(zip(min_vector, max_vector)): 
            if torch.isclose(tmin, tmax):
                scale[idx] = tmin if not torch.isclose(tmin, torch.zeros(1)) else 1.
            else:
                scale[idx] = tmax - tmin
        
        return min_vector, scale
    
    # ----- Saving predictions -----
    @staticmethod
    def free_edges_mask(pattern, stitches, num_stitches):
        """
        Construct the mask to identify edges that are not connected to any other
        """
        mask = np.ones((pattern.shape[0], pattern.shape[1]), dtype=bool)
        max_edge = pattern.shape[1]

        for side in stitches[:, :num_stitches]:  # ignore the padded part
            for edge_id in side:
                mask[edge_id // max_edge][edge_id % max_edge] = False
        
        return mask
    
    @staticmethod
    def prediction_to_stitches(free_mask_logits, similarity_matrix, return_stitches=False):
        free_mask = (torch.sigmoid(free_mask_logits.squeeze(-1)) > 0.5).flatten()
        if not return_stitches:
            simi_matrix = similarity_matrix + similarity_matrix.transpose(0, 1)
            simi_matrix = torch.masked_fill(simi_matrix, (~free_mask).unsqueeze(0), -float("inf"))
            simi_matrix = torch.masked_fill(simi_matrix, (~free_mask).unsqueeze(-1), 0)
            num_stitches = free_mask.nonzero().shape[0] // 2
        else:
            simi_matrix = similarity_matrix
            num_stitches = simi_matrix.shape[0] // 2
        simi_matrix = torch.triu(simi_matrix, diagonal=1)
        stitches = []
        
        for i in range(num_stitches):
            index = (simi_matrix == torch.max(simi_matrix)).nonzero()
            stitches.append((index[0, 0].cpu().item(), index[0, 1].cpu().item()))
            simi_matrix[index[0, 0], :] = -float("inf")
            simi_matrix[index[0, 1], :] = -float("inf")
            simi_matrix[:, index[0, 0]] = -float("inf")
            simi_matrix[:, index[0, 1]] = -float("inf")
        
        if len(stitches) == 0:
            stitches = None
        else:
            stitches = np.array(stitches)
            if stitches.shape[0] != 2:
                stitches = np.transpose(stitches, (1, 0))
        return stitches


    def save_gt_batch_imgs(self, gt_batch, datanames, data_folders, save_to):
        gt_imgs = []
        for idx, (name, folder) in enumerate(zip(datanames, data_folders)):
            gt = {}
            for key in gt_batch:
                gt[key] = gt_batch[key][idx]
                if (('order_matching' in self.config and self.config['order_matching'])
                    or 'origin_matching' in self.config and self.config['origin_matching']
                    or not self.gt_caching):
                    print(f'{self.__class__.__name__}::Warning::Propagating '
                        'information from GT on prediction is not implemented in given context')
                else:
                    if self.gt_caching and folder + '/static' in self.gt_cached:
                        gtc = self.gt_cached[folder + '/static']
                    else:
                        gtc = self._load_ground_truth(folder + "/static")
                    for key in gtc:
                        if key not in gt:
                            gt[key] = gtc[key]
            
            # Transform to pattern object
            pname = os.path.basename(folder) + "__" + os.path.basename(name.replace(".png", ""))
            pattern = self._pred_to_pattern(gt, pname)

            try: 
                # log gt number of panels
                final_dir = pattern.serialize(save_to, to_subfolder=True, tag=f'_gt_')
            except (RuntimeError, InvalidPatternDefError, TypeError) as e:
                print('GarmentDetrDataset::Error::{} serializing skipped: {}'.format(name, e))
                continue

            final_file = pattern.name + '_gt__pattern.png'
            gt_imgs.append(Path(final_dir) / final_file)
        return gt_imgs
    
    def save_prediction_single(self, prediction, dataname="outside_dataset", save_to=None, return_stitches=False):
        pattern_mask = self._pred_to_pattern(prediction, dataname, return_stitches=return_stitches)
        try: 
            final_mask_dir = pattern_mask.serialize(save_to, to_subfolder=True, tag=dataname[:20].replace(", ", "_") + '_')
        except (RuntimeError, InvalidPatternDefError, TypeError) as e:
            print('GarmentDetrDataset::Error::{} serializing skipped: {}'.format(dataname, e))

        final_file = pattern_mask.name + '_predicted__pattern.png'
        # prediction_img = Path(final_dir) / final_file
        prediction_mask_img = Path(final_mask_dir) / final_file
        
        return pattern_mask.pattern['new_panel_ids'], prediction_mask_img


    def save_prediction_batch(self, predictions, datanames, data_folders, save_to, **kwargs):
        """ 
            Saving predictions on batched from the current dataset
            Saves predicted params of the datapoint to the requested data folder.
            Returns list of paths to files with prediction visualizations
            Assumes that the number of predictions matches the number of provided data names"""

        prediction_imgs = []
        free_text = kwargs.get("free_text", False)
        for idx, (name, folder) in enumerate(zip(datanames, data_folders)):
            # "unbatch" dictionary
            prediction = {}
            pname = os.path.basename(folder) + "__" + os.path.basename(name.replace(".png", ""))
            tmp_path = os.path.join(save_to, pname, '_predicted_specification.json')
            if os.path.exists(tmp_path):
                continue
            
            print("Progress {}".format(tmp_path))

            for key in predictions:
                prediction[key] = predictions[key][idx]
            if "images" in kwargs:
                prediction["input"] = kwargs["images"][idx]
            if "panel_shape" in kwargs:
                prediction["panel_l2"] = kwargs["panel_shape"][idx]
            # Transform to pattern object
            
            pattern = self._pred_to_pattern(prediction, pname)
            # log gt number of panels
            if self.gt_caching and folder + f"/{name}" in self.gt_cached:
                gt = self.gt_cached[folder + f"/{name}"]
                pattern.spec['properties']['correct_num_panels'] = int(gt['num_panels']) if 'num_panels' in gt.keys() else None
            elif "use_gt_stitches" in kwargs and kwargs["use_gt_stitches"]:
                pattern.spec['properties']['correct_num_panels'] = int(gt['num_panels']) if 'num_panels' in gt.keys() else None

            try: 
                tag = f'_predicted_{prediction["panel_l2"]}_' if "panel_l2" in prediction else f"_predicted_"
                final_dir = pattern.serialize(save_to, to_subfolder=True, tag=tag)
            except (RuntimeError, InvalidPatternDefError, TypeError) as e:
                print('GarmentDetrDataset::Error::{} serializing skipped: {}'.format(folder, e))
                continue
            final_file = pattern.name + '_predicted__pattern.png'
            prediction_imgs.append(Path(final_dir) / final_file)
            # copy originals for comparison
            for file in Path(folder + f"/{name}").glob('*'):
                if ('.png' in str(file)) or ('.json' in str(file)):
                    shutil.copy2(str(file), str(final_dir))
            if free_text:
                with open(Path(final_dir) / 'free_text.txt', 'w') as f:
                    f.write(f"Free text")
            else:
                with open(Path(final_dir) / 'free_text.txt', 'w') as f:
                    f.write(f"Use text")

        return prediction_imgs
    
    def _pred_to_pattern(self, prediction, dataname, return_stitches=False):
        """Convert given predicted value to pattern object
        """
        # undo standardization  (outside of generinc conversion function due to custom std structure)
        gt_shifts = self.config['standardize']['gt_shift']
        gt_scales = self.config['standardize']['gt_scale']

        for key in gt_shifts:
            if key == 'stitch_tags':  
                # ignore stitch tags update if explicit tags were not used
                continue
            
            pred_numpy = prediction[key].detach().cpu().numpy()
            if key == 'outlines' and len(pred_numpy.shape) == 2: 
                pred_numpy = pred_numpy.reshape(self.config["max_pattern_len"], self.config["max_panel_len"], 9)

            el_len = pred_numpy.shape[-1]
            prediction[key] = pred_numpy * np.array(gt_scales[key][:el_len]).astype(np.float32) + np.array(gt_shifts[key][:el_len]).astype(np.float32)

        if 'stitches' in prediction:  # if somehow prediction already has an answer
            stitches = prediction['stitches']
        elif 'stitch_tags' in prediction: # stitch tags to stitch list 
            if 'free_edges_mask' in prediction:
                # get the stitches according to the distance between stitch tags
                stitches = self.tags_to_stitches(
                    torch.from_numpy(prediction['stitch_tags']) if isinstance(prediction['stitch_tags'], np.ndarray) else prediction['stitch_tags'],
                    prediction['free_edges_mask']
                )
            else:
                stitches = None
        elif 'edge_cls' in prediction and "edge_similarity" in prediction:
            stitches = self.prediction_to_stitches(prediction['edge_cls'], prediction['edge_similarity'], return_stitches=return_stitches)
        else:
            stitches = None
        
        # Construct the pattern from the data
        pattern_mask = NNSewingPattern(panel_classifier=self.panel_classifier)
        pattern_mask.name = dataname

        try: 
            pattern_mask.pattern_from_tensors(
                prediction['outlines'], 
                edge_type=prediction['edge_type'],
                edge_mask=prediction['edge_mask'],
                panel_rotations=prediction['rotations'],
                panel_translations=prediction['translations'], 
                stitches=stitches,
                reverse_stitch=prediction['reverse_stitch'] if 'reverse_stitch' in prediction.keys() else None,
                padded=True)   
        except (RuntimeError, InvalidPatternDefError) as e:
            print('GarmentDetrDataset::Warning::{} with discrete mask: {}'.format(dataname, e))
            pass
        
        return pattern_mask
    
