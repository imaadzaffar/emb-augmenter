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
# import sys
# sys.path.append('../../2-dagan/models')
from models.generator import (
    GeneratorMLP,
    GeneratorTransformer,
    GeneratorIndependent,
    GeneratorIndependentFast,
)

import logging

log = logging.getLogger(__name__)


class WSIDatasetFactory:
    def __init__(
        self,
        data_dir,
        csv_path,
        split_dir,
        seed=7,
        augmentation=None,
        dagan_settings=None,
        print_info=True,
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

        # ---> self
        self.data_dir = data_dir
        self.labels = pd.read_csv(csv_path)
        self.num_classes = 6
        self.split_dir = split_dir
        self.seed = seed
        self.augmentation = augmentation
        self.dagan_settings = dagan_settings
        self.print_info = print_info

        # ---> summarize
        self._summarize()

    def _train_val_test_split(self):
        # @TODO: read csv with split info, ie which samples belong to train/test/val.
        return None, None, None

    def _summarize(self):
        if self.print_info:
            log.debug("Init patch dataset factory...")

    def return_splits(self, fold_id):

        all_splits = pd.read_csv(
            os.path.join(self.split_dir, "splits_{}.csv".format(fold_id))
        )

        # augmentation for training
        train_split = self._get_split_from_df(
            all_splits=all_splits,
            split_key="train",
            augmentation=self.augmentation,
            dagan_settings=self.dagan_settings,
        )

        # no augmentation for validation, testing
        val_split = self._get_split_from_df(all_splits=all_splits, split_key="val")
        test_split = self._get_split_from_df(all_splits=all_splits, split_key="test")

        return train_split, val_split, test_split

    def _get_split_from_df(
        self,
        all_splits: dict = {},
        split_key: str = "train",
        augmentation=None,
        dagan_settings=None,
        scaler=None,
    ):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)
        split = list(split.values)

        # get all the labels whose case ID is in split
        labels = self.labels[self.labels["image_id"].isin(split)]

        if len(split) > 0:
            # ---> create patch dataset
            split_dataset = WSIDataset(
                data_dir=self.data_dir,
                labels=labels,
                num_classes=self.num_classes,
                augmentation=augmentation,
                dagan_settings=dagan_settings,
            )
        else:
            split_dataset = None

        return split_dataset


class WSIDataset(Dataset):
    def __init__(
        self,
        data_dir,
        labels,
        num_classes=8,
        augmentation=None,
        dagan_settings=None,
    ):

        super(WSIDataset, self).__init__()

        # ---> self
        self.data_dir = data_dir
        self.labels = labels
        self.num_classes = num_classes
        self.augmentation = augmentation
        self.dagan_settings = dagan_settings

        self.generator = None
        if dagan_settings is not None:
            self.generator = self._load_generator_model(dagan_settings)

    def _get_dagan_state_path(self, dagan_run_code):
        """Get path to file with saved state for specific DA-GAN generator"""

        state_path = f"/home/guillaume/Documents/uda/project-augmented-embeddings/2-dagan/results/sicapv2/{dagan_run_code}/s_4_checkpoint.pt"

        log.debug(f"DA-GAN state path: {state_path}")
        return state_path

    def _load_generator_model(self, dagan_settings):
        """Initiate generator model and load saved state from 2-dagan experiment"""

        log.debug(dagan_settings)

        dagan_state_path = self._get_dagan_state_path(dagan_settings["run_code"])
        dagan_state_dict = torch.load(dagan_state_path)

        # log.debug(dagan_state_dict)

        if dagan_settings["model"] == "mlp":
            generator = GeneratorMLP(
                n_tokens=dagan_settings["n_tokens"], dropout=dagan_settings["drop_out"]
            )
        elif dagan_settings["model"] == "transformer":
            generator = GeneratorTransformer(
                n_tokens=dagan_settings["n_tokens"],
                dropout=dagan_settings["drop_out"],
                n_heads=dagan_settings["n_heads"],
                emb_dim=dagan_settings["emb_dim"],
            )
        elif dagan_settings["model"] == "independent":
            generator = GeneratorIndependent()
        elif dagan_settings["model"] == "independent_fast":
            generator = GeneratorIndependentFast()
        else:
            raise ValueError("Invalid model type for generator")

        generator.load_state_dict(dagan_state_dict["G_state_dict"])

        log.debug(generator)
        return generator

    def __getitem__(self, idx):
        patch_embs = self._load_wsi_from_path(self.labels.iloc[idx]["image_id"])
        label = int(self.labels.iloc[idx]["isup_grade"])
        return patch_embs, label

    def _generate_aug_patch_embs(self, patch_embs, num_augs):
        """generate patch embs from dagan generator"""
        original_emb = patch_embs[:, 0, :]

        aug_embs = [original_emb]

        for n in range(num_augs):
            with torch.no_grad():
                noise = torch.randn(
                    original_emb.size(0), original_emb.size(1), requires_grad=False
                )

                # log.debug(f"embs: {original_emb.size()}")
                # log.debug(f"noise: {noise.size()}")

                aug_emb = self.generator.forward(original_emb, noise)
                aug_embs.append(aug_emb)

        aug_embs = torch.stack(aug_embs, dim=1)
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
        path = os.path.join(self.data_dir, "{}.pt".format(slide_id))
        patch_embs = torch.load(path)

        aug_choices_dict = {
            "combined": [6, 7, 8, 9],
            "rotation": [1],
            "hue": [2],
            "saturation": [3],
            "value": [4],
            "zoom": [5],
        }

        if self.augmentation is None:
            return patch_embs[:, 0, :]  # get the original patch embedding
        elif self.augmentation in aug_choices_dict:
            if self.generator is not None:
                num_augs = 4
                patch_embs = self._generate_aug_patch_embs(patch_embs, num_augs)
                aug_patch_embs = self._get_aug_patch_embs(
                    patch_embs, [n for n in range(num_augs + 1)]
                )
            else:
                aug_patch_embs = self._get_aug_patch_embs(
                    patch_embs, aug_choices_dict[self.augmentation]
                )
            # log.debug(f"aug_embs: {aug_patch_embs.size()}")
            return aug_patch_embs
        else:
            raise ValueError("Augmentation not recognized.")

    def __len__(self):
        return len(list(self.labels["image_id"].values))
