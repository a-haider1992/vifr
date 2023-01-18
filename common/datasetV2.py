## Author: Abbas Haider
## Date: 13 Jan 2023

import numpy as np
import torch.utils.data as tordata
import os.path as osp
from torchvision.datasets.folder import pil_loader

import random


from common.ops import age2group
import pdb


## VIFR CustomDatasets --------------------------------------------------------------------------
class VIFRDataset(tordata.Dataset):
    def __init__(self, dataset_name, transform=None):
        self.transform = transform
        self.root = osp.join(osp.dirname(osp.dirname(__file__)), 'dataset')
        self.image_labels = None
        self.image_list = None
        pdb.set_trace()
        with open(osp.join(self.root, dataset_name, 'annotated.txt'), 'r') as f:
            contents = f.read()
            self.image_labels = {index: label for index, label in enumerate(contents.splitlines())}
            self.image_list = [osp.join(self.root, dataset_name, label) for _, label in enumerate(contents.splitlines())]
    def __getitem__(self, index):
        return self.image_list[index], self.image_labels[index]
    def __len__(self):
        return len(self.image_labels)

class TrainDataset(VIFRDataset):
    def __init__(self, dataset_name, transform=None):
        super().__init__(osp.join(dataset_name, 'training'), transform)

    def __getitem__(self, index):
        img = pil_loader(self.image_list[index])
        if self.transform is not None:
            img = self.transform(img)
        label = self.image_labels[index].split('_')
        age = label[0]
        gender = label[1]
        race = label[2]
        return img, age, gender, race
    def __len__(self):
        return len(self.image_labels)

class EvaluationDataset(VIFRDataset):
    def __init__(self, dataset_name, transform=None):
        super().__init__(osp.join(dataset_name, 'evaluation'), transform)
    def __getitem__(self, index):
        img = pil_loader(self.image_list[index])
        if self.transform is not None:
            img = self.transform(img)
        label = self.image_labels[index].split('_')
        age = label[0]
        gender = label[1]
        race = label[2]
        return img, age, gender, race


## Aging Dataset V2#
class AgingDatasetV2(VIFRDataset):
    def __init__(self, dataset_name, age_group, total_pairs, transform=None):
        super().__init__(dataset_name, transform)
        self.ages = []
        self.genders = []
        self.image_names = []
        for _, label in self.image_labels.items():
            label_value = label.split('_')
            self.image_names.append(label_value[3])
            self.ages.append(label_value[0])
            self.genders.append(label_value[1])
        ## age2group function takes list of age, and generate age groups for each element
        pdb.set_trace()
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
