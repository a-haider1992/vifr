## Time-Independent Block
## Multi-Task Learning based time-independent tasks estimation e.g. gender, race
## Author: Abbas Haider
## Date: 14 Jan 2023

import torch
import torch.nn as nn
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
        pass
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
        pass
    def adjust_learning_rate(self, step):
        pass
    def validate(self, n_iter):
        pass
    def train(self, inputs, n_iter):
        pass