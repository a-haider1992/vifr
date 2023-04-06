# Time Dependent Task Block
# A transfromer based TD task estimation e.g. Age
# Author: Abbas Haider
# Date: 14 Jan 2023

import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
from . import BasicTask

import torch.nn.functional as F
from common.ops import convert_to_ddp

from common.sampler import RandomSampler
from common.data_prefetcher import DataPrefetcher
from common.dataset import TrainImageDataset, EvaluationImageDataset
from common.datasetV2 import TrainDataset, EvaluationDataset
from common.datasetV3 import TrainingData, EvaluationData


class TDTask(BasicTask):

    def set_loader(self):
        opt = self.opt
        # 1-100 Age labels
        self.train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize([opt.image_size, opt.image_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, ], std=[0.5, ])
            ])
        self.evaluation_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize([opt.image_size, opt.image_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, ], std=[0.5, ])
            ])
        lfw_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize([opt.image_size, opt.image_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, ], std=[0.5, ])
            ])
        if opt.dataset_name == "casia-webface" or opt.dataset_name == "scaf":
            train_dataset = TrainImageDataset(
                opt.dataset_name, self.train_transform)
            # evaluation_dataset = EvaluationImageDataset(
            #     opt.evaluation_dataset, self.evaluation_transform)
            weights = None
            sampler = RandomSampler(train_dataset, batch_size=opt.batch_size,
                                    num_iter=opt.num_iter, restore_iter=opt.restore_iter, weights=weights)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=opt.batch_size, sampler=sampler, pin_memory=True,
                                                       num_workers=opt.num_worker, drop_last=True)
            # evaluation_loader = torch.utils.data.DataLoader(evaluation_dataset,
            #                                                 batch_size=opt.eval_batch_size, pin_memory=True,
            #                                                 num_workers=opt.num_worker)

        elif opt.dataset_name == "lfw":
            # LFW dataset
            train_lfw_dataset = TrainingData('pairs.csv', lfw_transform)
            # test_lfw_dataset = EvaluationData('lfwTest.csv', lfw_transform)
            weights = None
            sampler_lfw = RandomSampler(
                train_lfw_dataset, batch_size=opt.batch_size, num_iter=opt.num_iter, weights=weights)
            train_loader = torch.utils.data.DataLoader(train_lfw_dataset,
                                                       batch_size=opt.batch_size,
                                                       sampler=sampler_lfw, num_workers=opt.num_worker,
                                                       drop_last=True)
            # evaluation_loader = torch.utils.data.DataLoader(
            #     test_lfw_dataset, num_workers=opt.num_worker)
        elif opt.dataset_name == "UTK":
            pass
        else:
            return Exception("Database doesn't exist.")

        # Train Prefetcher
        self.prefetcher = DataPrefetcher(train_loader)

        # # Evaluation prefetcher
        # self.eval_prefetcher = DataPrefetcher(evaluation_loader)

    def set_model(self):
        opt = self.opt
        # self.tdblock = MyViT((1, opt.image_size, opt.image_size), n_patches=8, n_blocks=2,
        #                           hidden_d=8, n_heads=2, out_d=len(self.prefetcher.__loader__.dataset.classes))
        # self.tdblock = convert_to_ddp(self.tdblock)
        # self.optimizer = torch.optim.SGD(list(self.tdblock.parameters()),
        #                                  momentum=self.opt.momentum, lr=self.opt.learning_rate)
        # self.criterion = nn.CrossEntropyLoss()

    def adjust_learning_rate(self, step):
        assert step > 0, 'batch index should large than 0'
        opt = self.opt
        if step > opt.warmup:
            lr = opt.learning_rate * \
                (opt.gamma ** np.sum(np.array(opt.milestone) < step))
        else:
            lr = step * opt.learning_rate / opt.warmup
        lr = max(1e-4, lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def validate(self, n_iter):
        pass

    def train(self, inputs, n_iter):
        opt = self.opt
        self.tdblock.train()
        images, labels, ages, genders = inputs
        age_loss = self.criterion(self.tdblock(images), ages)
        self.optimizer.zero_grad()
        age_loss.backward()
        self.optimizer.step()
        # apply_weight_decay(self.tdblock,
        #                    weight_decay_factor=opt.weight_decay, wo_bn=True)
        # id_loss = reduce_loss(id_loss)
        self.adjust_learning_rate(n_iter)
        lr = self.optimizer.param_groups[0]['lr']
        self.logger.msg([age_loss, lr], n_iter)

def patchify(images, n_patches):
    n, c, w, h = images.shape

    assert h == w, "Patchify method works for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size,
                            j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches

def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(
                i / (10000 ** (j/d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

class MyViT(nn.Module):
    def __init__(self, chw, n_patches, n_blocks, hidden_d, n_heads, out_d, age_group) -> None:
        super().__init__()

        self.chw = chw
        self.n_patches = n_patches
        self.hidden_d = hidden_d

        assert chw[1] % n_patches == 0, "Input shape not completely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not completely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        self.input_d = int(
            chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3) Positional embedding
        self.register_buffer('positional_embeddings', get_positional_embeddings(
            n_patches ** 2 + 1, hidden_d), persistent=False)

        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList(
            [MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # 5) Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )
        self.age_group_layer = nn.Linear(out_d, age_group)

    def forward(self, images):
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches)
        tokens = self.linear_mapper(patches)
        # Adding classification token to the tokens
        tokens = torch.cat(
            (self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Getting the classification token only
        out = out[:, 0]

        return self.mlp(out), self.age_group_layer(out)

class MSA(nn.Module):
    def __init__(self, d, n_heads=2) -> None:
        super(MSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"
        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head *
                            self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(
                    seq), v_mapping(seq)

                attention = self.softmax(
                    q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4) -> None:
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )
        ## To move the last layer to CUDA to resolve error `out` on GPU than CPU
        self.mlp = nn.ModuleList(self.mlp)

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out
