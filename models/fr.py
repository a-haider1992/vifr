import torch
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
import torch.cuda.amp as amp

from common.sampler import RandomSampler
from common.data_prefetcher import DataPrefetcher
from common.ops import convert_to_ddp, get_dex_age, age2group, apply_weight_decay, reduce_loss
from common.grl import GradientReverseLayer
from . import BasicTask
from backbone.aifr import backbone_dict, AgeEstimationModule
from head.cosface import CosFace
from common.dataset import TrainImageDataset, EvaluationImageDataset
from common.datasetV2 import TrainDataset, EvaluationDataset
from common.datasetV3 import TrainingData, EvaluationData
import pdb

class FR(BasicTask):

    def set_dataset(self):
        pass

    def set_loader(self):
        opt = self.opt

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
                transforms.ToTensor(),
                transforms.Resize([opt.image_size, opt.image_size])
            ])
        
        if opt.dataset_name == "casia-webface":
            train_dataset = TrainImageDataset(
                opt.dataset_name, self.train_transform)
            evaluation_dataset = EvaluationImageDataset(
                'casia-webface-eval', self.evaluation_transform)
            weights = None
            sampler = RandomSampler(train_dataset, batch_size=opt.batch_size,
                                    num_iter=opt.num_iter, restore_iter=opt.restore_iter, weights=weights)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=opt.batch_size, sampler=sampler, pin_memory=True,
                                                       num_workers=opt.num_worker, drop_last=True)
            evaluation_loader = torch.utils.data.DataLoader(evaluation_dataset,
                                                            batch_size=opt.eval_batch_size, pin_memory=True,
                                                            num_workers=opt.num_worker)

        elif opt.dataset_name == "lfw":
            # LFW dataset
            train_lfw_dataset = TrainingData('lfwTrain.csv', lfw_transform)
            test_lfw_dataset = EvaluationData('lfwTest.csv', lfw_transform)
            weights = None
            sampler_lfw = RandomSampler(
                train_lfw_dataset, batch_size=opt.batch_size, num_iter=opt.num_iter, weights=weights)
            train_loader = torch.utils.data.DataLoader(train_lfw_dataset,
                                                       batch_size=opt.batch_size,
                                                       sampler=sampler_lfw, num_workers=opt.num_worker,
                                                       drop_last=True)
            evaluation_loader = torch.utils.data.DataLoader(
                test_lfw_dataset, num_workers=opt.num_worker)
        elif opt.dataset_name == "UTK":
            pass
        else:
            return Exception("Database doesn't exist.")

        # Train Prefetcher
        self.prefetcher = DataPrefetcher(train_loader)

        # Evaluation prefetcher
        self.eval_prefetcher = DataPrefetcher(evaluation_loader)

    def set_model(self):
        opt = self.opt
        backbone = backbone_dict[opt.backbone_name](input_size=opt.image_size)
        head = CosFace(in_features=512, out_features=len(self.prefetcher.__loader__.dataset.classes),
                       s=opt.head_s, m=opt.head_m)

        estimation_network = AgeEstimationModule(
            input_size=opt.image_size, age_group=opt.age_group)

        da_discriminator = AgeEstimationModule(
            input_size=opt.image_size, age_group=opt.age_group)

        if opt.gfr:
            optimizer = torch.optim.SGD(list(head.parameters()),
                                        momentum=opt.momentum, lr=opt.learning_rate)
        else:
            optimizer = torch.optim.SGD(list(backbone.parameters()) +
                                        list(head.parameters()) +
                                        list(estimation_network.parameters()) +
                                        list(da_discriminator.parameters()),
                                        momentum=opt.momentum, lr=opt.learning_rate)

        backbone, head, estimation_network, da_discriminator = convert_to_ddp(backbone, head, estimation_network,
                                                                              da_discriminator)
        scaler = amp.GradScaler()
        self.optimizer = optimizer
        self.backbone = backbone
        self.head = head
        self.estimation_network = estimation_network
        self.da_discriminator = da_discriminator
        self.grl = GradientReverseLayer()
        self.scaler = scaler

        self.logger.modules = [optimizer, backbone, head,
                               estimation_network, da_discriminator, scaler]
        if opt.restore_iter > 0:
            self.logger.load_checkpoints(opt.restore_iter)

    def validate(self, n_iter):
        pass

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

    def compute_age_loss(self, x_age, x_group, ages):
        opt = self.opt
        age_loss = F.mse_loss(get_dex_age(x_age), ages) + \
            F.cross_entropy(x_group, age2group(
                ages, age_group=opt.age_group).long())
        return age_loss

    def forward_da(self, x_id, ages):
        x_age, x_group = self.da_discriminator(self.grl(x_id))
        loss = self.compute_age_loss(x_age, x_group, ages)
        return loss

    def train(self, inputs, n_iter):
        opt = self.opt

        if opt.gfr:
            # For LFW type datasets
            # A pre-trained backbone is used
            images, labels = inputs
            embedding = self.backbone(images)
        else:
            # For casia-webface type datasets
            self.backbone.train()
            self.head.train()
            images, labels, ages, genders = inputs
            self.da_discriminator.train()
            self.estimation_network.train()
            if opt.amp:
                with amp.autocast():
                    embedding, x_id, x_age = self.backbone(
                        images, return_age=True)
                embedding = embedding.float()
                x_id = x_id.float()
                x_age = x_age.float()
            else:
                embedding, x_id, x_age = self.backbone(images, return_age=True)

        if opt.gfr:
            # Train GFR only
            id_loss = F.cross_entropy(self.head(embedding, labels), labels)
            self.optimizer.zero_grad()
            id_loss.backward()
            self.optimizer.step()
            apply_weight_decay(self.backbone, self.head,
                               weight_decay_factor=opt.weight_decay, wo_bn=True)
            # id_loss = reduce_loss(id_loss)
            # lr = self.optimizer.param_groups[0]['lr']
            # self.logger.msg({'id_loss':id_loss, 'lr':lr}, n_iter)
        else:
            # Train Face Recognition with ages and genders
            id_loss = F.cross_entropy(self.head(embedding, labels), labels)
            x_age, x_group = self.estimation_network(x_age)
            age_loss = self.compute_age_loss(x_age, x_group, ages)
            da_loss = self.forward_da(x_id, ages)
            loss = id_loss + \
                age_loss * opt.fr_age_loss_weight + \
                da_loss * opt.fr_da_loss_weight

            total_loss = loss
            if opt.amp:
                total_loss = self.scaler.scale(loss)
            self.optimizer.zero_grad()
            total_loss.backward()
            apply_weight_decay(self.backbone, self.head, self.estimation_network,
                               weight_decay_factor=opt.weight_decay, wo_bn=True)
            if opt.amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            id_loss, da_loss, age_loss = reduce_loss(
                id_loss, da_loss, age_loss)
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.msg([id_loss, da_loss, age_loss, lr], n_iter)
