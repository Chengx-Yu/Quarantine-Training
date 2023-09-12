from torch.utils.data import Dataset
import numpy as np
import torch


class TransformedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.transform = transform
        self.dataset = dataset

    def __getitem__(self, item):
        img = self.dataset[item][0]
        if self.transform is not None:
            img = self.transform(img)
        label_idx = self.dataset[item][1]
        gt_label = self.dataset[item][2]
        is_bad = self.dataset[item][3]
        return img, label_idx, gt_label, is_bad

    def __len__(self):
        return len(self.dataset)
