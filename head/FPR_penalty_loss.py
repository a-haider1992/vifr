import torch
import torch.nn as nn
import torch.nn.functional as F


class FPRPenaltyLoss(nn.Module):
    r"""Implement of FPR Penalty Loss 
    (https://paperswithcode.com/paper/consistent-instance-false-positive-improves):
    Args:
        in_channels: size of output channels of backbone
        out_features: size of output classes
        s: norm of input feature
        kernel_size: kernel size
        stride: stride
        hidden_size: hidden dimensions
        alpha: control paramater for penalty
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, hidden_size=64, alpha=0.1):
        super(FPRPenaltyLoss, self).__init__()

        self.alpha = alpha

        self.conv = nn.Conv2d(in_channels, hidden_size, kernel_size, stride, padding)
        self.fc = nn.Linear(hidden_size * 7 * 7, out_channels)

    def forward(self, logits, labels, neg_mask, pos_mask):
        # Flatten the 4D tensor to 2D for the fully connected layer
        h = self.conv(logits)
        h = h.view(h.size(0), -1)
        logits_fc = self.fc(h)

        # Compute the cross-entropy loss for the positive samples
        pos_loss = F.cross_entropy(logits_fc[pos_mask], labels[pos_mask])

        # Compute the FPR Penalty loss for the negative samples
        neg_logits_fc = logits_fc[neg_mask]
        neg_labels = torch.zeros(neg_logits_fc.shape[0], dtype=torch.long).to(logits_fc.device)
        neg_loss = F.cross_entropy(neg_logits_fc, neg_labels)

        # Compute the FPR Penalty term
        fpr_penalty = torch.exp(self.alpha * neg_loss)

        # Compute the overall loss
        loss = pos_loss + fpr_penalty

        return loss
