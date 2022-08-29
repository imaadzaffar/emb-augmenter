from __future__ import print_function, division
import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, ConcatDataset


class PatchDatasetFactory:

    def __init__(self,
        data_dir, 
        csv_path,
        split_dir, 
        seed = 7, 
        n_features = 1024,
        print_info = True
        ):
        r"""
        PatchDatasetFactory 

        Args:
            data_dir (string): Path to patch dataset
            split_dir (string): Path to split info csv 
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
        """

        #---> self
        self.data_dir = data_dir
        self.labels = pd.read_csv(csv_path)
        self.split_dir = split_dir
        self.seed = seed
        self.n_features = n_features
        self.print_info = print_info
        self.train_ids, self.val_ids, self.test_ids  = self._train_val_test_split()

        #---> summarize
        self._summarize()

    def _train_val_test_split(self):
        # @TODO: read csv with split info, ie which samples belong to train/test/val. 
        return None, None, None

    def _summarize(self):
        if self.print_info:
            print('Init patch dataset factory...')
            print("number of cases {}".format(0))

    def return_splits(self, fold_id):
        all_splits = pd.read_csv(os.path.join(self.split_dir, 'splits_{}.csv'.format(fold_id)))
        
        train_split = self._get_split_from_df(all_splits=all_splits, split_key='train')
        val_split = self._get_split_from_df(all_splits=all_splits, split_key='val')
        test_split = self._get_split_from_df(all_splits=all_splits, split_key='test')

        return train_split, val_split, test_split

    def _get_split_from_df(self, all_splits: dict={}, split_key: str='train', scaler=None):
        patch_datasets = []

        # get all the labels whose case ID is in split
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)
        split = list(split.values)
        # print("SLIDE_IDS:", split)

        # make datasets for each WSI in split
        if len(split) > 0:
            for slide_id in split:
                #---> create patch dataset
                split_dataset = PatchDataset(
                    data_dir=self.data_dir,
                    slide_id=slide_id,
                    n_features=self.n_features,
                )
                # print("SPLIT DATASET LEN:", len(split_dataset))
                patch_datasets.append(split_dataset)
        else:
            return None
        
        # concatenate datasets into a single dataset
        patch_dataset = ConcatDataset(patch_datasets)

        # print("PATCH DATASET:", patch_dataset)
        print("PATCH DATASET LEN:", len(patch_dataset))

        return patch_dataset
    

class PatchDataset(Dataset):

    def __init__(self,
        data_dir, 
        slide_id,
        n_features=1024,
        ): 

        super(PatchDataset, self).__init__()

        #---> self
        self.data_dir = data_dir
        self.n_features = n_features
        
        path = os.path.join(self.data_dir, '{}.pt'.format(slide_id))
        self.data = torch.load(path)

    def __getitem__(self, idx):
        original, augmentation  = self._load_patch_pair(idx)
        noise = torch.randn(self.n_features)
        return original, augmentation, noise 

    def _load_patch_pair(self, patch_index):
        """
        Load a pair of patch embeddings. 
        """
        patch = self.data[patch_index, :, :]

        original = patch[0, :]  # get the original patch embedding
        augmentation = patch[np.random.randint(low=6, high=10), :]  # get a random mixed augmentation

        return original, augmentation
    
    def __len__(self):
        return self.data.shape[0]
