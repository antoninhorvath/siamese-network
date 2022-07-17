"""
:author: anton

This module implements the ability for the model to obtain individual stereo pairs.
"""

import torch
from torch.utils.data import Dataset


class StereoDataset(Dataset):
    """
    Class used for getting each RGB stereo pair from the dataset
    """
    def __init__(self, dataset_path):
        super(StereoDataset, self).__init__()
        self._dataset_path = dataset_path

        self._dataset = torch.load(dataset_path)

        assert self._dataset['left'].size(0) == self._dataset['right'].size(0)

    def __len__(self):
        return self._dataset['left'].size(0)

    def __getitem__(self, idx):
        left_images = self._dataset['left'][idx]
        right_images = self._dataset['right'][idx]

        return left_images / 255.0, right_images / 255.0
