## Author: Abbas Haider
## Date: 13 Jan 2023

import numpy as np
import torch.utils.data as tordata
import os.path as osp
from torchvision.datasets.folder import pil_loader

import random


from common.ops import age2group


## VIFR CustomDatasets --------------------------------------------------------------------------
class VIFRDataset(tordata.Dataset):
    def __init__(self, dataset_name, transform=None):
        self.transform = transform
        self.root = osp.join(osp.dirname(osp.dirname(__file__)), 'dataset')
        self.image_label = None
        with open(osp.join(self.root, dataset_name, '/annotated.txt'), 'w') as f:
            self.labels = f.readlines()
            self.image_label = {index: label for index, label in enumerate(self.labels)}
    def __getitem__(self, index):
        return self.image_label[index]
    def __len__(self):
        return len(self.image_label)

class TrainDataset(VIFRDataset):
    def __init__(self, dataset_name, transform=None):
        super().__init__(dataset_name+'/training', transform)

    def __getitem__(self, index):
        img = pil_loader(self.image_list[index])
        if self.transform is not None:
            img = self.transform(img)
        label = self.image_label[index].split('_')
        age = label[0]
        gender = label[1]
        race = label[2]
        return img, age, gender, race
    def __len__(self):
        return len(self.image_label)

class EvaluationDataset(VIFRDataset):
    def __init__(self, dataset_name, transform=None):
        super().__init__(dataset_name+'/evaluation', transform)
    def __getitem__(self, index):
        img = pil_loader(self.image_list[index])
        if self.transforms is not None:
            img = self.transforms(img)
        return img


## Aging Dataset V2#
class AgingDataset(VIFRDataset):
    def __init__(self, dataset_name, age_group, total_pairs, transform=None):
        super().__init__(dataset_name, transform)
        self.ages = []
        self.genders = []
        self.ids = []
        for _, label in self.image_label.items():
            label_value = label.split('_')
            self.ids.append(label_value[3])
            self.ages.append(label_value[0])
            self.genders.append(label_value[2])
        self.groups = age2group(self.ages, age_group=age_group).astype(int)
        self.label_group_images = []
        for i in range(age_group):
            self.label_group_images.append(
                self.image_list[self.groups == i].tolist())
        np.random.seed(0)
        self.target_labels = np.random.randint(0, age_group, (total_pairs,))
        self.total_pairs = total_pairs

    def __getitem__(self, index):
        target_label = self.target_labels[index]
        target_img = pil_loader(random.choice(self.label_group_images[target_label]))
        if self.transforms is not None:
            target_img = self.transforms(target_img)
        return target_img, target_label

    def __len__(self):
        return self.total_pairs
