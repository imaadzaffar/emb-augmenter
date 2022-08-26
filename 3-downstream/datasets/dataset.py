from __future__ import print_function, division
from cProfile import label
from multiprocessing.sharedctypes import Value
import os
from pickle import NONE
import pandas as pd

import torch
from torch.utils.data import Dataset
import numpy as np


class WSIDatasetFactory:

    def __init__(self,
        data_dir, 
        csv_path,
        split_dir, 
        seed = 7, 
        augmentation_type = None,
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
        self.augmentation_type = augmentation_type
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
        train_split = self._get_split_from_df(all_splits=all_splits, split_key='train', augmentation_type=self.augmentation_type)
        val_split = self._get_split_from_df(all_splits=all_splits, split_key='val')
        test_split = self._get_split_from_df(all_splits=all_splits, split_key='test')

        return train_split, val_split, test_split

    def _get_split_from_df(self, all_splits: dict={}, split_key: str='train', augmentation_type=None, scaler=None):
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
                split=split_key,
                augmentation_type=augmentation_type
            )
        else:
            split_dataset = None
        
        return split_dataset
    

class WSIDataset(Dataset):

    def __init__(self,
        data_dir, 
        labels,
        num_classes=8,
        split='train',
        augmentation_type=None
        ): 

        super(WSIDataset, self).__init__()

        #---> self
        self.data_dir = data_dir
        self.labels = labels
        self.num_classes = num_classes
        self.split = split
        self.augmentation_type = augmentation_type

    def __getitem__(self, idx):

        patch_embs = self._load_wsi_from_path(self.labels.iloc[idx]['image_id'])
        label = int(self.labels.iloc[idx]['isup_grade'])
        return patch_embs, label

    def _load_wsi_from_path(self, slide_id):
        """
        Load a patch embedding. 
        """
        path = os.path.join(self.data_dir, '{}.pt'.format(slide_id))
        patch_embs = torch.load(path)

        if self.augmentation_type is None:
            return patch_embs[:, 0, :]  # get the original patch embedding
        elif self.augmentation_type == 'combined':
            aug_indices = np.random.choice([0, 6, 7, 8, 9], size=patch_embs.shape[0])  # get random mixed augmentation for each patch
        elif self.augmentation_type == 'rotation':
            aug_indices = np.random.choice([0, 1], size=patch_embs.shape[0])  # get random augmentation for each patch
        elif self.augmentation_type == 'hue':
            aug_indices = np.random.choice([0, 2], size=patch_embs.shape[0])  # get random augmentation for each patch
        elif self.augmentation_type == 'saturation':
            aug_indices = np.random.choice([0, 3], size=patch_embs.shape[0])  # get random augmentation for each patch
        elif self.augmentation_type == 'value':
            aug_indices = np.random.choice([0, 4], size=patch_embs.shape[0])  # get random augmentation for each patch
        elif self.augmentation_type == 'zoom':
            aug_indices = np.random.choice([0, 5], size=patch_embs.shape[0])  # get random augmentation for each patch
        else:
            raise ValueError("Augmentation not recognized.")

        patch_indices = np.arange(patch_embs.shape[0])
        patch_embs = patch_embs[patch_indices, aug_indices, :]
        return patch_embs
    
    def __len__(self):
        return len(list(self.labels['image_id'].values))
