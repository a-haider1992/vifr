import torch
import torch.nn as nn
import torch.nn.functional as F

class CircleLoss(nn.Module):
    def __init__(self, in_features, out_features, gamma=64, margin=0.25):
        super(CircleLoss, self).__init__()
        self.gamma = gamma
        self.margin = margin
        self.O_p = 1 + self.margin
        self.O_n = -self.margin
        self.Delta_p = 1 - self.margin
        self.Delta_n = self.margin
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings (torch.Tensor): Embeddings from the backbone network (batch_size, embedding_size)
            labels (torch.Tensor): Ground truth labels (batch_size)

        Returns:
            torch.Tensor: Circle Loss
        """
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        similarity_matrix = torch.matmul(embeddings_norm, weight_norm.t())

        one_hot = torch.zeros(similarity_matrix.size()).to(embeddings_norm)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        one_hot = one_hot.type(dtype=torch.bool)

        sp = similarity_matrix[one_hot]
        sn = similarity_matrix[~one_hot]

        ap = torch.clamp_min(-sp.detach() + 1 + self.margin, min=0.)
        an = torch.clamp_min(sn.detach() + self.margin, min=0.)

        delta_p = 1 - self.margin
        delta_n = self.margin

        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = F.softplus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss.mean()
