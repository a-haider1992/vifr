## Author: Abbas Haider
## Date: 06 Feb 2023

import torch.utils.data as tordata
import os.path as osp
import os
import pandas, pdb
from torchvision.datasets.folder import pil_loader

class LFWDataset(tordata.Dataset):
    def __init__(self, file, transform=None):
        # pdb.set_trace()
        self.transform = transform
        self.root = osp.join(osp.dirname(osp.dirname(__file__)), 'dataset', 'lfw')
        df = pandas.read_csv(osp.join(osp.dirname(osp.dirname(__file__)), 'dataset', file), delimiter=',')
        image_folder_path = df['name'].to_list()
        self.labels = []
        self.images = []
        for idx, name in enumerate(image_folder_path):
            image_folder_path[idx] = osp.join(self.root, name)
            images_ = os.listdir(image_folder_path[idx])
            for image in images_:
                self.images.append(osp.join(image_folder_path[idx], image))
                self.labels.append(name)
        # pdb.set_trace()
        self.image_count = df['images'].to_list()
        
    def __getitem__(self, index):
        image = pil_loader(self.images[index])
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
    