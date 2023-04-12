# Time-Independent Block
# Multi-Task Learning based time-independent tasks estimation e.g. gender, race
# Author: Abbas Haider
# Date: 14 Jan 2023

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms

from . import BasicTask


class TIDTask():
    def __init__(self) -> None:
        self.gender_classifier = GenderClassifier()
        self.race_classifier = RaceClassifier()


class GenderClassifier(BasicTask):
    def set_loader(self):
        pass

    def set_model(self):
        opt = self.opt
        # Load a pre-trained ResNet-18 model
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            model.fc.parameters(), lr=0.001, momentum=0.9)
        self.gc = model
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
        # Apply the transform to the batch of data after loading
        # Use this step after loading the dataset in FR task, for pretrained Gender classifier
        # Gender classifier requires a different transform than FR and FAS tasks 
        # images_transformed = torch.stack([transform(image) for image in images])

    def adjust_learning_rate(self, step):
        pass

    def validate(self, n_iter):
        pass

    def train(self, inputs, n_iter):
        pass


class RaceClassifier(BasicTask):
    def set_loader(self):
        pass

    def set_model(self):
        opt = self.opt
        # Load a pre-trained ResNet-18 model
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        # Asian, Caucasian, Black
        model.fc = nn.Linear(num_ftrs, 3)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            model.fc.parameters(), lr=0.001, momentum=0.9)
        self.rc = model
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

    def adjust_learning_rate(self, step):
        pass

    def validate(self, n_iter):
        pass

    def train(self, inputs, n_iter):
        pass
