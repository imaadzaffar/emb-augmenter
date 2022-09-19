from __future__ import print_function, division
import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, ConcatDataset

import logging

log = logging.getLogger(__name__)


class PatchDatasetFactory:
    def __init__(self, seed=7, dataset_size=1, n_features=1024, print_info=True):
        r"""
        PatchDatasetFactory

        Args:
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
        """

        # ---> self
        self.seed = seed
        self.dataset_size = dataset_size
        self.n_features = n_features
        self.print_info = print_info

        # ---> summarize
        self._summarize()

    def _summarize(self):
        if self.print_info:
            log.debug("Init patch dataset factory...")
            log.debug(f"number of cases {0}")

    def return_splits(self):
        train_split = self._get_split_from_df(split_key="train")
        val_split = self._get_split_from_df(split_key="val")
        test_split = self._get_split_from_df(split_key="test")

        return train_split, val_split, test_split

    def _get_split_from_df(self, split_key="train", scaler=None):
        # ---> create patch dataset
        patch_dataset = PatchSimDataset(
            size=self.dataset_size,
            n_features=self.n_features,
        )

        log.debug(f"Patch dataset len: {len(patch_dataset)}")

        return patch_dataset


class PatchSimDataset(Dataset):
    def __init__(
        self,
        size=1000000,
        n_features=1024,
    ):

        super(PatchSimDataset, self).__init__()

        # ---> self
        self.size = size
        self.n_features = n_features

    def __getitem__(self, idx):
        original = torch.randn(self.n_features)
        noise = torch.randn(self.n_features)
        augmentation = original + noise / 16

        return original, augmentation, noise

    def __len__(self):
        return self.size
