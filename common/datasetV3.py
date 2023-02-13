## Author: Abbas Haider
## Date: 06 Feb 2023

import torch.utils.data as tordata
import os.path as osp
import os
import numpy as np
import pandas, pdb
from torchvision.datasets.folder import pil_loader

class LFWDataset(tordata.Dataset):
    def __init__(self, file, transform=None):
        pdb.set_trace()
        self.transform = transform
        self.root = osp.join(osp.dirname(osp.dirname(__file__)), 'dataset', 'lfw')
        df = pandas.read_csv(osp.join(osp.dirname(osp.dirname(__file__)), 'dataset', file), delimiter=',')
        self.names = df['name'].to_list()
        self.images = df['image'].to_list()
        self.labels = df['encoded_name'].astype(int).to_list()
        self.classes = np.unique(self.labels)
        
    def __getitem__(self, index):
        image_path = osp.join(os.path.dirname(os.path.dirname(__file__)), self.images[index])
        image = pil_loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[index]
        return image, label
    def __len__(self):
        return len(self.images)

class TrainingData(LFWDataset):
    def __init__(self, file, transform):
        super().__init__(file, transform)

    def __getitem__(self, index):
        return super().__getitem__(index)

class EvaluationData(LFWDataset):
    def __init__(self, file, transform):
        super().__init__(file, transform)
    def __getitem__(self, index):
        return super().__getitem__(index)
    