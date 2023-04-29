import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    def __init__(self, input_size, num_classes, s=30.0, m=0.50):
        super(ArcFace, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(input_size, num_classes))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels):
        # normalize feature
        x = F.normalize(x)

        # normalize weights
        weights = F.normalize(self.weight)

        # dot product
        logits = F.linear(x, weights)

        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)

        # add scale
        logits = logits * (1 - labels) + target_logits * labels
        logits *= self.s

        return logits
