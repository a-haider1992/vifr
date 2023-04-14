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
from .td_block import ViT
from backbone.aifr import backbone_dict, AgeEstimationModule
from head.cosface import CosFace
from common.dataset import TrainImageDataset, EvaluationImageDataset
from common.datasetV2 import TrainDataset, EvaluationDataset
from common.datasetV3 import TrainingData, EvaluationData, TrainingDataAge, EvaluationDataAge
import pdb


class FR(BasicTask):

    def set_dataset(self):
        pass

    def set_loader(self):
        opt = self.opt
        if opt.dataset_name == "casia-webface" or opt.dataset_name == "scaf":
            print("Loading Casia-webface or SCAF dataset..")
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

            train_dataset = TrainImageDataset(
                opt.dataset_name, self.train_transform)
            age_db_dataset = TrainingDataAge(
                'AgeDB.csv', self.evaluation_transform)
            # evaluation_dataset = EvaluationImageDataset(
            #     opt.evaluation_dataset, self.evaluation_transform)
            weights = None
            sampler = RandomSampler(train_dataset, batch_size=opt.batch_size,
                                    num_iter=opt.num_iter, restore_iter=opt.restore_iter, weights=weights)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=opt.batch_size, sampler=sampler, pin_memory=True,
                                                       num_workers=opt.num_worker, drop_last=True)
            evaluation_loader = torch.utils.data.DataLoader(age_db_dataset,
                                                            batch_size=opt.eval_batch_size, pin_memory=True,
                                                            num_workers=opt.num_worker)

        elif opt.dataset_name == "lfw":
            # LFW dataset
            print("Loading LFW dataset..")
            lfw_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize([opt.image_size, opt.image_size]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, ], std=[0.5, ])
                ])

            torch.cuda.empty_cache()
            train_lfw_dataset = TrainingData('pairs.csv', lfw_transform)
            # test_lfw_dataset = EvaluationData('lfwTest.csv', lfw_transform)
            weights = None
            sampler_lfw = RandomSampler(
                train_lfw_dataset, batch_size=opt.batch_size, num_iter=opt.num_iter, weights=weights)
            train_loader = torch.utils.data.DataLoader(train_lfw_dataset,
                                                       batch_size=opt.batch_size,
                                                       sampler=sampler_lfw, pin_memory=True, num_workers=opt.num_worker,
                                                       drop_last=True)
            print(
                f"GPU memory allocated after loading data: {torch.cuda.memory_allocated()} bytes")
            # evaluation_loader = torch.utils.data.DataLoader(
            #     test_lfw_dataset, num_workers=opt.num_worker)
        elif opt.dataset_name == "UTK" or opt.dataset_name == "AgeDB":
            print("Loading AgeDB or UTK dataset..")
            agedb_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize([opt.image_size, opt.image_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, ], std=[0.5, ])
            ])

            torch.cuda.empty_cache()
            age_db_dataset = TrainingDataAge('agedb_train.csv', agedb_transform)
            agedb_evaluation_dataset = EvaluationDataAge('agedb_test.csv', agedb_transform)
            weights = None
            sampler = RandomSampler(
                age_db_dataset, batch_size=opt.batch_size, num_iter=opt.num_iter, weights=weights)
            train_loader = torch.utils.data.DataLoader(age_db_dataset,
                                                       batch_size=opt.batch_size,
                                                       sampler=sampler, pin_memory=True, num_workers=opt.num_worker,
                                                       drop_last=True)
            evaluation_loader = torch.utils.data.DataLoader(agedb_evaluation_dataset, batch_size=1, num_workers=opt.num_worker)

        else:
            return Exception("Database doesn't exist.")

        # Train Prefetcher
        self.prefetcher = DataPrefetcher(train_loader)

        # # Evaluation prefetcher
        self.eval_prefetcher = DataPrefetcher(evaluation_loader)

    def set_model(self):
        opt = self.opt
        backbone = backbone_dict[opt.backbone_name](input_size=opt.image_size)
        head = CosFace(in_features=512, out_features=len(self.prefetcher.__loader__.dataset.classes),
                       s=opt.head_s, m=opt.head_m)

        # if age estimation network to TD block VIT
        if opt.td_block:
            estimation_network = ViT(image_size=opt.image_size, patch_size=7, num_classes=101,
                                     hidden_features=32,
                                     num_heads=2, num_layers=2, age_group=opt.age_group)
            # estimation_network = PreTrainedVIT(image_size=opt.image_size)
            optimizer_new = torch.optim.Adam(list(backbone.parameters()) +
                                        list(head.parameters()) +
                                        list(estimation_network.parameters()),
                                        lr=opt.learning_rate, betas=(opt.momentum, 0.999))
           # with open('VIT_keys.txt', 'w') as f:
            #     for key in estimation_network.state_dict().keys():
            #         f.write(key + '\n')
            # with open('backbone_keys.txt', 'w') as f:
            #     for key in backbone.state_dict().keys():
            #         f.write(key + '\n')

            # estimation_network = MyViT((3, opt.image_size, opt.image_size), n_patches=7, n_blocks=5,
            #                       hidden_d=32, n_heads=10, out_d=101, age_group=opt.age_group)
        else:
            estimation_network = AgeEstimationModule(
                input_size=opt.image_size, age_group=opt.age_group)

        da_discriminator = AgeEstimationModule(
            input_size=opt.image_size, age_group=opt.age_group)
        has_backbone_params = False
        if opt.gfr:
            optimizer = torch.optim.Adam(list(estimation_network.parameters()), lr=0.001)
            # Freeze all layers except last
            # last_layer_name = list(backbone.named_modules())[-1][0]
            # for name, param in backbone.named_parameters():
            #     if last_layer_name is not name:   # Skip the last layer
            #         param.requires_grad = False
        else:
            optimizer = torch.optim.SGD(list(backbone.parameters()) +
                                        list(head.parameters()) +
                                        list(estimation_network.parameters()) +
                                        list(da_discriminator.parameters()),
                                        momentum=opt.momentum, lr=opt.learning_rate)
        # if not opt.evaluation_only:
        backbone, head, estimation_network, da_discriminator = convert_to_ddp(backbone, head, estimation_network,
                                                                              da_discriminator)
        # with open('VIT_keys_after_ddp.txt', 'w') as f:
        #         for key in estimation_network.state_dict().keys():
        #             f.write(key + '\n')
        scaler = amp.GradScaler()
        self.optimizer = optimizer
        # self.optimizer = optimizer_new
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
            # AgeDB
            # A pre-trained backbone is used
            self.da_discriminator.train()
            self.estimation_network.train()
            self.backbone.eval()
            images, ages = inputs
            embedding, x_id, x_age = self.backbone(images, return_age=True)
        else:
            # For casia-webface type datasets
            self.head.train()
            self.backbone.train()
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
            x_age, x_group = self.estimation_network(x_id)
            age_loss = self.compute_age_loss(x_age, x_group, ages)
            da_loss = self.forward_da(x_id, ages)
            loss = age_loss * opt.fr_age_loss_weight + \
                da_loss * opt.fr_da_loss_weight
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            apply_weight_decay(self.head,
                               weight_decay_factor=opt.weight_decay, wo_bn=True)
            age_loss, da_loss = reduce_loss(age_loss, da_loss)
            # self.adjust_learning_rate(n_iter)
            # lr = self.optimizer.param_groups[0]['lr']
            self.logger.msg([age_loss, da_loss, lr], n_iter)
        else:
            # Train Face Recognition with ages and genders
            id_loss = F.cross_entropy(self.head(embedding, labels), labels)

            # If using VIT then feed images directly to estimation network
            # if opt.td_block:

            # x_age, x_group = self.estimation_network(images)
            

            x_age, x_group = self.estimation_network(x_age)
            age_loss = self.compute_age_loss(x_age, x_group, ages)
            da_loss = self.forward_da(x_id, ages)
            loss = id_loss + \
                age_loss * opt.fr_age_loss_weight + \
                da_loss * opt.fr_da_loss_weight


            # loss = id_loss + opt.fr_age_loss_weight * age_loss
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
            self.adjust_learning_rate(n_iter)
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.msg([id_loss, da_loss, age_loss, lr], n_iter)

            # id_loss,  age_loss = reduce_loss(
            #     id_loss, age_loss)
            # self.adjust_learning_rate(n_iter)
            # lr = self.optimizer.param_groups[0]['lr']
            # self.logger.msg([id_loss, age_loss, lr], n_iter)
    
    def train_pretrained_eval(self):
        opt = self.opt
        # from sklearn.model_selection import KFold
        # # 10-fold cross-validation here
        # kfold = KFold(n_splits=10, shuffle=True)

        # self.backbone.eval()

        # # Define loss function and optimizer
        # criterion = torch.nn.MSELoss()
        # optimizer = torch.optim.SGD(list(self.estimation_network.parameters()),
        #                                 momentum=opt.momentum, lr=0.001)

        # for fold, (train_idx, test_idx) in enumerate(kfold.split(self.age_db_dataset)):
        #     train_dataset = torch.utils.data.Subset(self.age_db_dataset, train_idx)
        #     test_dataset = torch.utils.data.Subset(self.age_db_dataset, test_idx)

        #     train_loader = torch.utils.data.DataLoader(
        #         train_dataset, batch_size=opt.eval_batch_size, shuffle=True)
        #     test_loader = torch.utils.data.DataLoader(
        #         test_dataset, batch_size=1, shuffle=False)
            
        #     train_fetcher = DataPrefetcher(train_loader)
        #     test_fetcher = DataPrefetcher(test_loader)
        #     total_loss = 0.0
        #     self.estimation_network.train()
        #     for _ in range(opt.evaluation_num_iter):
        #         images, ages = train_fetcher.next()
        #         embedding, x_id, x_age = self.backbone(images, return_age=True)
        #         x_age, x_group = self.estimation_network(x_age)
        #         # print(get_dex_age(x_age).dtype)
        #         # print(ages.dtype)
        #         age_loss = self.compute_age_loss(x_age, x_group, ages)
        #         loss = age_loss
        #         # print(age_loss.dtype)
        #         # total_loss += age_loss
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #     print(f'Fold {fold + 1} training loss : {total_loss}')
        print("Age Estimation Model under evaluation.")
        self.estimation_network.eval()
        self.backbone.eval()
        total_correct_pred = 0
        total_incorrect_pred = 0
        with torch.no_grad():
            for iter in range(50):
                image, age = self.eval_prefetcher.next()
                embedding, x_id, x_age = self.backbone(
                        image, return_age=True)
                predicted_age, predicted_group = self.estimation_network(
                        x_age)
                    # print("The correct age tensor shape is : {}".format(age.shape))
                    # print("The predicted age tensor shape is : {}".format(predicted_age.shape))
                if age.item() == torch.argmax(predicted_age).item():
                    total_correct_pred += 1
                else:
                    total_incorrect_pred += 1
                        # print("The correct age is : {}".format(age.item()))
                        # print("The predicted age is : {}".format(
                        #     torch.argmax(predicted_age).item()))
            accuracy = total_correct_pred / (total_correct_pred+total_incorrect_pred)
            print(f'The correct predictions are {total_correct_pred}')
            print(f'The Incorrect predictions are {total_incorrect_pred}')
            print(f'The accuracy of Age estimation model : {accuracy}')
