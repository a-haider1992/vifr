## Author: Abbas Haider
## Date: 06 Feb 2023

import torch.utils.data as tordata
import os.path as osp
import os
import pandas
from torchvision.datasets.folder import pil_loader
import numpy as np

def find_image_paths(images, image_num1, image_num2):
    for image in images:
        parts = image.split('_')
        num = parts[len(parts)-1]
        image_num1 = image_num2 = ''
        if image_num1 in num:
            for idx in range(0, len(parts)-1):
                image_num1 += parts[idx]+'_'
            image_num1 += num
        if image_num2 in num:
            for idx in range(0, len(parts)-1):
                image_num2 += parts[idx]+'_'
            image_num2 += num
    return image_num1, image_num2

class LFWDataset(tordata.Dataset):
    def __init__(self, file, transform=None):
        self.transform = transform
        self.root = osp.join(osp.dirname(osp.dirname(__file__)), 'dataset', 'lfw')
        df = pandas.read_csv(osp.join(osp.dirname(osp.dirname(__file__)), 'dataset', file), delimiter=',')
        self.names = df['name'].to_list()
        image1 = df['imagenum1'].to_list()
        image2 = df['imagenum2'].to_list()
        self.images = {}
        self.classes = self.names
        for idx in range(0, len(image1)):
            images = os.listdir(os.path.join(self.root, self.names[idx]))
            im1, im2 = find_image_paths(images, str(image1[idx]), str(image2[idx]))
            self.images[idx] = [os.path.join(self.root, self.names[idx], im1), os.path.join(self.root, self.names[idx], im2)]
        # for idx in range(0, len(images_)):
        #     self.images.append(os.path.join(self.root, self.names[idx], images_[idx]))
        # self.labels = df['encoded_name'].astype(int).to_list()
        # self.classes = np.unique(self.labels)
        
    def __getitem__(self, index):
        image1 = pil_loader(self.images[index][0])
        image2 = pil_loader(self.images[index][1])
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2
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
    
class AgeDB(tordata.Dataset):
    def __init__(self, file, transform=None):
        self.transform = transform
        self.root = osp.join(osp.dirname(osp.dirname(__file__)), 'dataset', 'AgeDB')
        df = pandas.read_csv(osp.join(osp.dirname(osp.dirname(__file__)), 'dataset', file), delimiter=',')
        self.data = df.values
        self.paths = df['path'].to_list()
        self.ages = df['age'].astype(np.float32).to_list()
        self.images = []
        self.classes = self.paths
        for path in self.paths:
            path = os.path.join(self.root, path)
            self.images.append(path)
        
    def __getitem__(self, index):
        image = pil_loader(self.images[index])
        if self.transform is not None:
            image = self.transform(image)
        age = self.ages[index]
        return image, age
    def __len__(self):
        return len(self.images)
    
class UTK(tordata.Dataset):
    def __init__(self, file, transform=None):
        self.transform = transform
        self.root = osp.join(osp.dirname(osp.dirname(__file__)), 'dataset', 'UTK')
        df = pandas.read_csv(osp.join(osp.dirname(osp.dirname(__file__)), 'dataset', file), delimiter=',')
        self.paths = df['path'].to_list()
        self.ages = df['age'].to_list()
        self.genders = df['gender'].to_list()
        self.images = []
        self.classes = self.paths
        for path in self.paths:
            path = os.path.join(self.root, path)
            self.images.append(path)
        
    def __getitem__(self, index):
        image = pil_loader(self.images[index])
        if self.transform is not None:
            image = self.transform(image)
        age = self.ages[index]
        gender = self.genders[index]
        return image, age, gender
    def __len__(self):
        return len(self.images)

class TrainingDataAge(AgeDB):
    def __init__(self, file, transform):
        super().__init__(file, transform)

    def __getitem__(self, index):
        return super().__getitem__(index)
    
class EvaluationDataAge(AgeDB):
    def __init__(self, file, transform):
        super().__init__(file, transform)

    def __getitem__(self, index):
        return super().__getitem__(index)