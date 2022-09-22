from __future__ import print_function, division
from cProfile import label
from multiprocessing.sharedctypes import Value
import os
from pickle import NONE
import pandas as pd

import torch
from torch.utils.data import Dataset
import numpy as np

# TODO: Import generator classes from 2-dagan
from models.generator import GeneratorMLP, GeneratorTransformer

class WSIDatasetFactory:

    def __init__(self,
        data_dir, 
        csv_path,
        split_dir, 
        seed = 7, 
        augmentation = None,
        dagan = False,
        dagan_settings = {},
        print_info = True
        ):
        r"""
        WSIDatasetFactory 

        Args:
            data_dir (string): Path to patch dataset
            csv_path (string): Path to csv file with slide ids and labels 
            split_dir (string): Path to split info csv 
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
        """

        #---> self
        self.data_dir = data_dir
        self.labels = pd.read_csv(csv_path)
        self.num_classes = 6
        self.split_dir = split_dir
        self.seed = seed
        self.augmentation = augmentation
        self.dagan = dagan
        self.dagan_settings = dagan_settings
        self.print_info = print_info

        #---> summarize
        self._summarize()

    def _train_val_test_split(self):
        # @TODO: read csv with split info, ie which samples belong to train/test/val. 
        return None, None, None

    def _summarize(self):
        if self.print_info:
            print('Init patch dataset factory...')

    def return_splits(self, fold_id):

        all_splits = pd.read_csv(os.path.join(self.split_dir, 'splits_{}.csv'.format(fold_id)))

        # augmentation for training
        train_split = self._get_split_from_df(all_splits=all_splits, split_key='train', augmentation=self.augmentation, dagan=self.dagan, dagan_settings=self.dagan_settings)

        # no augmentation for validation, testing
        val_split = self._get_split_from_df(all_splits=all_splits, split_key='val')
        test_split = self._get_split_from_df(all_splits=all_splits, split_key='test')

        return train_split, val_split, test_split

    def _get_split_from_df(self, all_splits: dict={}, split_key: str='train', augmentation=None, dagan=False, dagan_settings=None, scaler=None):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)
        split = list(split.values)

        # get all the labels whose case ID is in split
        labels = self.labels[self.labels['image_id'].isin(split)]

        if len(split) > 0:
            #---> create patch dataset
            split_dataset = WSIDataset(
                data_dir=self.data_dir,
                labels=labels,
                num_classes=self.num_classes,
                augmentation=augmentation,
                dagan_settings=dagan_settings
            )
        else:
            split_dataset = None
        
        return split_dataset
    

class WSIDataset(Dataset):

    def __init__(self,
        data_dir, 
        labels,
        num_classes=8,
        augmentation=None,
        dagan=False,
        dagan_settings={},
        ): 

        super(WSIDataset, self).__init__()

        #---> self
        self.data_dir = data_dir
        self.labels = labels
        self.num_classes = num_classes
        self.augmentation = augmentation

        if dagan == True:
            self._load_generator_model(dagan_settings)
            

    def _load_generator_model(self, dagan_settings):
        """Initiate generator model and load saved state from 2-dagan experiment"""

        dagan_state_dict = torch.load(os.path.join("../../2-dagan/", dagan_settings.state_path))

        if dagan_settings.model_type == 'mlp':
            self.generator = GeneratorMLP(n_tokens=dagan_settings.n_tokens, dropout=dagan_settings.drop_out)
        elif dagan_settings.model_type == 'transformer':
            self.generator = GeneratorTransformer(n_tokens=dagan_settings.n_tokens, dropout=dagan_settings.drop_out, n_heads=dagan_settings.n_heads, emb_dim=dagan_settings.emb_dim)
        else:
            raise ValueError("Invalid model type for generator")

        self.generator.load(dagan_state_dict["G_state_dict"])


    def __getitem__(self, idx):
        patch_embs = self._load_wsi_from_path(self.labels.iloc[idx]['image_id'])
        label = int(self.labels.iloc[idx]['isup_grade'])
        return patch_embs, label
    
    def _generate_aug_patch_embs(self, patch_embs, num_augs):
        """generate patch embs from dagan generator"""
        aug_embs = patch_embs
        for n in range(num_augs):
            noise = torch.randn(self.n_features)
            gen_aug_emb = self.generator.forward(patch_embs, noise)
            aug_embs = torch.stack((aug_embs, gen_aug_emb), dim=1)
        return aug_embs

    def _get_aug_patch_embs(self, patch_embs, aug_choices):
        """get random mixed augmentation for each patch"""
        aug_indices = np.random.choice([0] + aug_choices, size=patch_embs.shape[0])
        patch_indices = np.arange(patch_embs.shape[0])
        aug_patch_embs = patch_embs[patch_indices, aug_indices, :]
        return aug_patch_embs

    def _load_wsi_from_path(self, slide_id):
        """
        Load a patch embedding. 
        """
        path = os.path.join(self.data_dir, '{}.pt'.format(slide_id))
        patch_embs = torch.load(path)

        if self.dagan == True:
            num_augs = 4
            patch_embs = self._generate_aug_patch_embs(patch_embs, num_augs)
            aug_patch_embs = self._get_aug_patch_embs(patch_embs, [n for n in range(num_augs+1)])
            return aug_patch_embs

        aug_choices_dict = {
            'combined': [6,7,8,9],
            'rotation': [1],
            'hue': [2],
            'saturation': [3],
            'value': [4],
            'zoom': [5],
        }

        if self.augmentation is None:
            return patch_embs[:, 0, :]  # get the original patch embedding
        elif self.augmentation in aug_choices_dict:
            aug_patch_embs = self._get_aug_patch_embs(patch_embs, aug_choices_dict[self.augmentation])
            return aug_patch_embs
        else:
            raise ValueError("Augmentation not recognized.") 
    
    def __len__(self):
        return len(list(self.labels['image_id'].values))
