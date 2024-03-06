import torch.utils.data as tordata
import os.path as osp
import numpy as np
from torchvision.datasets.folder import pil_loader
import pandas as pd
import random
from collections import Counter
import pdb

from common.ops import age2group

class BaseImageDataset(tordata.Dataset):
    def __init__(self, dataset_name, transforms=None):
        self.transforms = transforms
        self.root = osp.join(osp.dirname(osp.dirname(__file__)), 'dataset')
        df = pd.read_csv(osp.join(self.root, '{}.txt'.format(dataset_name)), header=None, index_col=False, sep=' ')
        self.data = df.values
        self.image_list = np.array([osp.join(self.root, x) for x in self.data[:, 1]])

    def __len__(self):
        return len(self.image_list)


class EvaluationImageDataset(BaseImageDataset):
    def __init__(self, dataset_name, transforms=None):
        super(EvaluationImageDataset, self).__init__(dataset_name, transforms=transforms)
        self.ids = self.data[:, 0].astype(int)
        self.classes = np.unique(self.ids)

    def __getitem__(self, index):
        img = pil_loader(self.image_list[index])
        if self.transforms is not None:
            img = self.transforms(img)
        label = self.ids[index]
        return img, label


class TrainImageDataset(BaseImageDataset):
    def __init__(self, dataset_name, transforms=None):
        super(TrainImageDataset, self).__init__(dataset_name, transforms=transforms)
        self.ids = self.data[:, 0].astype(int)
        self.classes = np.unique(self.ids)
        self.ages = self.data[:, 2].astype(np.float32)
        self.genders = self.data[:, 3].astype(int)
        self.races = self.data[:, 4].astype(int)

         # Count the number of samples in each class
        self.gender_counts = Counter(self.genders)
        self.race_counts = Counter(self.races)


    def __getitem__(self, index):
        # print("index: ", index)
        img = pil_loader(self.image_list[index])
        if self.transforms is not None:
            img = self.transforms(img)
        age = self.ages[index]
        gender = self.genders[index]
        race = self.races[index]
       
        label = self.ids[index]
        return img, label, age, gender, race
    
    def get_gender_counts(self):
        return self.gender_counts
    
    def get_race_counts(self):
        return self.race_counts
    
    def get_gender_batch_counts(self, gender):
        print("gender test-----------------")
    
    def get_race_batch_counts(self, batch):
        print("race test-----------------")


class AgingDataset(BaseImageDataset):
    def __init__(self, dataset_name, age_group, total_pairs, transforms=None):
        super(AgingDataset, self).__init__(dataset_name, transforms=transforms)
        self.ids = self.data[:, 0].astype(int)
        self.classes = np.unique(self.ids)
        self.ages = self.data[:, 2].astype(np.float32)
        self.genders = self.data[:, 3].astype(int)
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
