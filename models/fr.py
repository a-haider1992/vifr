import pdb

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from torchvision import transforms

from backbone.aifr import (AgeEstimationModule, EthnicityFeatureExtractor,
                           GenderFeatureExtractor, backbone_dict)
from common.data_prefetcher import DataPrefetcher
from common.dataset import EvaluationImageDataset, TrainImageDataset
from common.datasetV2 import EvaluationDataset, TrainDataset
from common.datasetV3 import (UTK, EvaluationData, EvaluationDataAge,
                              TrainingData, TrainingDataAge, Casia)
from common.grl import GradientReverseLayer
from common.ops import (age2group, apply_weight_decay, convert_to_ddp,
                        get_dex_age, reduce_loss)
from common.sampler import RandomSampler
from head.arcface import ArcFace
from head.cosface import CosFace
from head.custom_loss import CrossEntropyLossCalculator

from . import BasicTask
from .td_block import ViT
import logging


class FR(BasicTask):

    def set_dataset(self):
        pass

    def set_loader(self):
        opt = self.opt
        logging.basicConfig(filename='fr.log', level=logging.INFO)
        if opt.dataset_name == "cvfr" or opt.dataset_name == "scaf":
            print("Loading CVFR or SCAF dataset..")
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
            self.total_gender_count = train_dataset.get_gender_counts()
            self.total_race_count = train_dataset.get_race_counts()
            # age_db_dataset = TrainingDataAge(
            #     'AgeDB.csv', self.evaluation_transform)
            # evaluation_dataset = EvaluationImageDataset(
            #     opt.evaluation_dataset, self.evaluation_transform)
            weights = None
            sampler = RandomSampler(train_dataset, batch_size=opt.batch_size,
                                    num_iter=opt.num_iter, restore_iter=opt.restore_iter, weights=weights)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=opt.batch_size, sampler=sampler, pin_memory=True,
                                                       num_workers=opt.num_worker, drop_last=True)
            # evaluation_loader = torch.utils.data.DataLoader(age_db_dataset,
            #                                                 batch_size=opt.eval_batch_size, pin_memory=True,
            #                                                 num_workers=opt.num_worker)

        elif opt.dataset_name == "lfw" or opt.dataset_name == "casia":
            # LFW dataset
            print("Loading LFW dataset or CASIA-WebFACE..")
            lfw_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize([opt.image_size, opt.image_size]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, ], std=[0.5, ])
                ])
            if opt.dataset_name == "lfw":
                torch.cuda.empty_cache()
                if opt.lfw_mode == 0:
                    train_lfw_dataset = TrainingData('pairs.csv', mode=0, transform=lfw_transform)
                elif opt.lfw_mode == 1:
                    train_lfw_dataset = TrainingData('pairs_mismatch.csv', mode=1, transform=lfw_transform)
                else:
                    print("Invalid LFW mode!! Defaulting to 0")
                    train_lfw_dataset = TrainingData('pairs.csv', mode=0, transform=lfw_transform)
                # test_lfw_dataset = EvaluationData('lfwTest.csv', lfw_transform)
                weights = None
                sampler_lfw = RandomSampler(
                    train_lfw_dataset, batch_size=opt.batch_size, num_iter=opt.num_iter, weights=weights)
                train_loader = torch.utils.data.DataLoader(train_lfw_dataset,
                                                        batch_size=opt.batch_size,
                                                        sampler=sampler_lfw, pin_memory=True, num_workers=opt.num_worker,
                                                        drop_last=True)
            elif opt.dataset_name == "casia":
                torch.cuda.empty_cache()
                casia_dataset = Casia('casia.csv', lfw_transform)
                weights = None
                sampler_casia = RandomSampler(
                    casia_dataset, batch_size=opt.batch_size, num_iter=opt.num_iter, weights=weights)
                train_loader = torch.utils.data.DataLoader(casia_dataset,
                                                        batch_size=opt.batch_size,
                                                        sampler=sampler_casia, pin_memory=True, num_workers=opt.num_worker,
                                                        drop_last=True)
        elif opt.dataset_name == "UTK" or opt.dataset_name == "AgeDB":
            print("Loading AgeDB or UTK dataset..")
            agedb_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize([opt.image_size, opt.image_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, ], std=[0.5, ])
            ])
            if opt.dataset_name == "AgeDB":
                torch.cuda.empty_cache()
                age_db_dataset = TrainingDataAge(
                    opt.age_protocol + ".csv", agedb_transform)
                # agedb_evaluation_dataset = EvaluationDataAge(
                #     'agedb_test.csv', agedb_transform)
                weights = None
                sampler = RandomSampler(
                    age_db_dataset, batch_size=opt.batch_size, num_iter=opt.num_iter, weights=weights)
                train_loader = torch.utils.data.DataLoader(age_db_dataset,
                                                           batch_size=opt.batch_size,
                                                           sampler=sampler, pin_memory=True, num_workers=opt.num_worker,
                                                           drop_last=True)
                # evaluation_loader = torch.utils.data.DataLoader(
                #     agedb_evaluation_dataset, batch_size=1, num_workers=opt.num_worker)
            else:
                utk_dataset = UTK(opt.utk_eval_file, agedb_transform)
                weights = None
                sampler = RandomSampler(
                    utk_dataset, batch_size=opt.batch_size, num_iter=opt.evaluation_num_iter, weights=weights)
                train_loader = torch.utils.data.DataLoader(utk_dataset,
                                                           batch_size=opt.batch_size,
                                                           sampler=sampler, pin_memory=True, num_workers=opt.num_worker,
                                                           drop_last=True)

        else:
            return Exception("Database doesn't exist.")

        # Train Prefetcher
        self.prefetcher = DataPrefetcher(train_loader)

        # # Evaluation prefetcher
        # self.eval_prefetcher = DataPrefetcher(evaluation_loader)

    def set_model(self):
        opt = self.opt
        backbone = backbone_dict[opt.backbone_name](input_size=opt.image_size)
        head = CosFace(in_features=512, out_features=len(self.prefetcher.__loader__.dataset.classes),
                       s=opt.head_s, m=opt.head_m)
        # Custom Loss
        class_based_loss = CrossEntropyLossCalculator()

        gender_estimation = GenderFeatureExtractor()
        race_estimation = EthnicityFeatureExtractor()
        if opt.td_block:
            da_discriminator = estimation_network = ViT(image_size=opt.image_size, patch_size=7, num_classes=101,
                                                        hidden_features=opt.vit_hidden_f,
                                                        num_heads=opt.vit_heads, num_layers=opt.vit_blocks, age_group=opt.age_group)
            # head = ArcFace(in_features=512, out_features=len(self.prefetcher.__loader__.dataset.classes))
            # optimizer = torch.optim.SGD(list(backbone.parameters()) +
            #                             list(head.parameters()) +
            #                             list(estimation_network.parameters()) +
            #                             list(gender_estimation.parameters()) +
            #                             list(da_discriminator.parameters()),
            #                             lr=opt.learning_rate, momentum=opt.momentum)
            optimizer = torch.optim.ASGD(list(backbone.parameters()) +
                                         list(head.parameters()) +
                                         list(estimation_network.parameters()) +
                                         list(gender_estimation.parameters()) +
                                         list(da_discriminator.parameters()) +
                                         list(race_estimation.parameters()),
                                         lr=opt.learning_rate)
        else:
            estimation_network = AgeEstimationModule(
                input_size=opt.image_size, age_group=opt.age_group)
            da_discriminator = AgeEstimationModule(
                input_size=opt.image_size, age_group=opt.age_group)
            optimizer = torch.optim.SGD(list(backbone.parameters()) +
                                        list(head.parameters()) +
                                        list(estimation_network.parameters()) +
                                        list(da_discriminator.parameters()),
                                        momentum=opt.momentum, lr=opt.learning_rate)
        # if not opt.evaluation_only:
        backbone, head, estimation_network, da_discriminator, gender_estimation, race_estimation = convert_to_ddp(backbone, head, estimation_network,
                                                                                                                  da_discriminator, gender_estimation, race_estimation)
        # with open('VIT_keys_after_ddp.txt', 'w') as f:
        #         for key in estimation_network.state_dict().keys():
        #             f.write(key + '\n')
        scaler = amp.GradScaler()
        self.optimizer = optimizer
        self.backbone = backbone
        self.head = head
        self.estimation_network = estimation_network
        self.gender_network = gender_estimation
        self.race_network = race_estimation
        self.da_discriminator = da_discriminator
        self.grl = GradientReverseLayer()
        self.scaler = scaler
        self.class_based_loss = class_based_loss

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

    def count_genders(self, genders):
        gender = {"Female": 0, "Male": 0}
        for item in genders:
            if item.item() == 0:
                gender["Female"] += 1
            else:
                gender["Male"] += 1
        return gender

    def count_races(self, races):
        print(f'Type of race:: {type(races)}')

    def train(self, inputs, n_iter):
        opt = self.opt
        # pdb.set_trace()
        images, labels, ages, genders, races = inputs
        # gender_counts = self.count_genders(genders)
        # race_counts = self.count_races(races)
        # print(f'Gender counts {gender_counts}')
        # print(f'Races counts {race_counts}')
        self.backbone.train()
        self.head.train()
        self.da_discriminator.train()
        self.estimation_network.train()
        self.gender_network.train()

        if opt.amp:
            with amp.autocast():
                embedding, x_id, x_age = self.backbone(images, return_age=True)
            embedding = embedding.float()
            x_id = x_id.float()
            x_age = x_age.float()
        else:
            embedding, x_id, x_age, x_residual = self.backbone(
                images, return_residual=True)

        # Train Face Recognition
        id_loss = F.cross_entropy(self.head(embedding, labels), labels)
        x_age, x_group = self.estimation_network(x_age)
        age_loss = self.compute_age_loss(x_age, x_group, ages)
        # da_loss = self.forward_da(x_id, ages)
        # gender_loss = F.cross_entropy(self.gender_network(x_residual), genders)
        predicted_gender = self.gender_network(x_residual)
        # pdb.set_trace()
        self.class_based_loss.calculate_losses(predicted_gender, genders)
        # print(f'Predicted gender {predicted_gender}')
        # print(f'Actual gender {genders}')
        # pdb.set_trace()
        # print(f'Gender class based losses {self.class_based_loss.get_loss_values()}')
        gender_based_loss = self.class_based_loss.get_loss_values()
        gender_loss = torch.tensor(max(gender_based_loss.values()), requires_grad=True).cuda()
        race_loss = F.cross_entropy(self.race_network(x_residual), races)
        loss = id_loss * opt.fr_id_loss_weight + \
            age_loss * opt.fr_age_loss_weight + \
        gender_loss * opt.fr_gender_loss_weight + \
        race_loss * opt.fr_race_loss_weight

        total_loss = loss
        if opt.amp:
            total_loss = self.scaler.scale(loss)
        self.optimizer.zero_grad()
        total_loss.backward()
        logging.info(f'Losses at iteration {n_iter} are {total_loss}')
        logging.info(f'Gender based losses at iteration {n_iter} is {gender_based_loss}')
        # self.logger.msg([total_loss, gender_loss], n_iter)
        # apply_weight_decay(self.backbone, self.head, self.estimation_network, self.gender_network, self.race_network,
        #                    weight_decay_factor=opt.weight_decay, wo_bn=True)
        if opt.amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # id_loss, da_loss, age_loss, gender_loss, race_loss = reduce_loss(
        #     id_loss, da_loss, age_loss, gender_loss, race_loss)
        lr = self.optimizer.param_groups[0]['lr']
        # self.logger.msg([id_loss, da_loss, age_loss, gender_loss, race_loss, lr], n_iter)
        return id_loss, age_loss, gender_loss, race_loss, total_loss

    def age_pretrained_eval(self):
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
            for _ in range(0, int(opt.evaluation_num_iter)):
                image, age, gender = self.prefetcher.next()
                target_age = age.item()
                if opt.age_group == 7:
                    target_group = 0
                    if target_age >= 10 and target_age <= 20:
                        target_group = 1
                    elif target_age >= 20 and target_age <= 30:
                        target_group = 2
                    elif target_age >= 30 and target_age <= 40:
                        target_group = 3
                    elif target_age >= 40 and target_age <= 50:
                        target_group = 4
                    elif target_age >= 50 and target_age <= 60:
                        target_group = 5
                    elif target_age > 60:
                        target_group = 6
                    elif target_age < 10:
                        target_group = 0
                elif opt.age_group == 4:
                    target_group = 0
                    if target_age >= 30 and target_age <= 40:
                        target_group = 1
                    elif target_age >= 40 and target_age <= 50:
                        target_group = 2
                    elif target_age > 50:
                        target_group = 3
                    elif target_age < 30:
                        target_group = 0
                embedding, x_id, x_age = self.backbone(
                    image, return_age=True)
                predicted_age, predicted_group = self.estimation_network(
                    x_age)
                # print("The correct age tensor shape is : {}".format(age.shape))
                # print("The predicted age tensor shape is : {}".format(predicted_age.shape))
                pred_group = 0
                pred_age = torch.argmax(predicted_age).item()
                predicted_group = torch.argmax(predicted_group).item()
                # if pred_age >= 10 and pred_age <= 20:
                #     pred_group = 1
                # elif pred_age >= 20 and pred_age <= 30:
                #     pred_group = 2
                # elif pred_age >= 30 and pred_age <= 40:
                #     pred_group = 3
                # elif pred_age >= 40 and pred_age <= 50:
                #     pred_group = 4
                # elif pred_age >=50 and pred_age <=60:
                #     pred_group = 5
                # elif pred_age > 60:
                #     pred_group = 6
                # elif pred_age < 10:
                #     pred_group = 0
                if target_age == pred_age or predicted_group == target_group:
                    total_correct_pred += 1
                else:
                    total_incorrect_pred += 1
                    print("The correct age is : {}".format(target_age))
                    print("The predicted age is : {}".format(pred_age))
                    print("The correct age group is : {}".format(target_group))
                    print("The predicted age group is : {}".format(predicted_group))
            accuracy = total_correct_pred / \
                (total_correct_pred+total_incorrect_pred)
            print(f'Total correct predictions are {total_correct_pred}')
            print(f'Total Incorrect predictions are {total_incorrect_pred}')
            print(f'Accuracy of Age estimation model : {accuracy}')

    def evaluate_gender_model(self):
        opt = self.opt
        print("Gender Estimation Model under evaluation.")
        self.gender_network.eval()
        self.backbone.eval()
        total_correct_pred = 0
        total_incorrect_pred = 0
        with torch.no_grad():
            for _ in range(0, int(opt.evaluation_num_iter)):
                if self.prefetcher.next() is None:
                    # print("Prefetcher next returns None.")
                    continue
                else:
                    image, age, gender, race = self.prefetcher.next()
                    embedding, x_id, x_age, x_residual = self.backbone(
                        image, return_residual=True)
                    if opt.eval_gender:
                        predicted_sex = self.gender_network(x_residual)
                        predicted_sex = torch.argmax(predicted_sex).item()
                        if predicted_sex == gender.item():
                            total_correct_pred += 1
                        else:
                            total_incorrect_pred += 1
                            # print(f'The predicted sex {predicted_sex}')
                            # print(f'The actual sex {gender.item()}')
                    else:
                        predicted_race = self.race_network(x_residual)
                        predicted_race = torch.argmax(predicted_race).item()
                        if predicted_race == race.item():
                            total_correct_pred += 1
                        else:
                            total_incorrect_pred += 1
                            print(f'The predicted race {predicted_race}')
                            print(f'The actual race {race.item()}')
            accuracy = total_correct_pred / \
                (total_correct_pred+total_incorrect_pred)
            print(f'Total correct predictions are {total_correct_pred}')
            print(f'Total Incorrect predictions are {total_incorrect_pred}')
            print(f'Accuracy : {accuracy}')
