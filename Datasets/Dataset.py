"""
Author: Xiaohan Chen
Mail: cxh_bb@outlook.com
"""

import torch
from torch.utils.data import Dataset
from Datasets.Augmentation import *
import numpy as np

class BaseDataset(Dataset):
    def __init__(self, dataset):
        """
        dataset (dict): a dictionary format dataset
        """
        super(BaseDataset, self).__init__()
        self.dataset = dataset
        self.classes = len(dataset)
        self.x, self.y = self._read(dataset)
    
    def _read(self, dataset):
        x = np.concatenate([dataset[key] for key in dataset.keys()])
        y = []
        for i, key in enumerate(dataset.keys()):
            number = len(dataset[key])
            y.append(np.tile(i, number))
        y = np.concatenate(y)
        return x, y
    
    def __len__(self):
        count = 0
        for key in self.dataset.keys():
            count += len(self.dataset[key])
        return count

    def __getitem__(self, index):
        data = self.x[index]
        label = self.y[index]

        # transform array to tensor
        data = torch.from_numpy(data).float()
        label = torch.tensor(label, dtype=torch.float)
        return data, label

class AugmentDasetsetTFPair(BaseDataset):
    """
    Generate augmentated data pairs.
    """
    def __getitem__(self, index):
        data = self.x[index]
        label = self.y[index]

        # transform array to tensor
        data = torch.from_numpy(data).float()
        label = torch.tensor(label, dtype=torch.float)

        data_t = random_waveform_transforms(data)
        data_f = frequency_transforms(data)

        return data_t, data_f, label

def relabel_dataset(args, dataset):
    """
    Partialy relabel the dataset with -1.
    """
    unlabeled_idx = []

    num_data_per_class = args.num_train
    num_unlabeled_per_class = args.num_train - args.num_labels
    classes = list(int(i) for i in args.labels.split(","))
    num_class = len(classes)

    # shuffle the label index
    idx = np.arange(num_data_per_class * num_class).reshape(num_class, num_data_per_class)
    for i in range(len(idx)):
        idx[i] = np.random.permutation(idx[i])
        unlabeled_idx.append(idx[i][:num_unlabeled_per_class])
    
    # relabel
    unlabeled_idx = np.array(unlabeled_idx).reshape(-1)
    dataset.y[unlabeled_idx] = -1

    unlabeled_indices = set(unlabeled_idx)
    labeled_indices = set(np.arange(num_data_per_class * num_class)) - unlabeled_indices

    return list(labeled_indices), list(unlabeled_indices)
