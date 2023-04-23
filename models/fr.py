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
from backbone.aifr import backbone_dict, AgeEstimationModule, GenderFeatureExtractor
from head.cosface import CosFace
from common.dataset import TrainImageDataset, EvaluationImageDataset
from common.datasetV2 import TrainDataset, EvaluationDataset
from common.datasetV3 import TrainingData, EvaluationData, TrainingDataAge, EvaluationDataAge, UTK
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
                transforms.Resize([opt.image_size, opt.image_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, ], std=[0.5, ])
            ])
            if opt.dataset_name == "AgeDB":
                torch.cuda.empty_cache()
                age_db_dataset = TrainingDataAge('agedb_train.csv', agedb_transform)
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
                utk_dataset = UTK('UTK.csv', agedb_transform)
                weights = None
                sampler = RandomSampler(
                    utk_dataset, batch_size=opt.batch_size, num_iter=opt.num_iter, weights=weights)
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

        gender_estimation = GenderFeatureExtractor()

        # if age estimation network to TD block VIT
        if opt.td_block:
            da_discriminator = estimation_network = ViT(image_size=opt.image_size, patch_size=7, num_classes=101,
                                                        hidden_features=opt.vit_hidden_f,
                                                        num_heads=opt.vit_heads, num_layers=opt.vit_blocks, age_group=opt.age_group)
            # estimation_network = PreTrainedVIT(image_size=opt.image_size)
            optimizer_new = torch.optim.Adam(list(backbone.parameters()) +
                                             list(head.parameters()) +
                                             list(
                                                 estimation_network.parameters()),
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
            optimizer = torch.optim.Adam(
                list(estimation_network.parameters()), lr=0.001)
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
        backbone, head, estimation_network, da_discriminator, gender_estimation = convert_to_ddp(backbone, head, estimation_network,
                                                                                                 da_discriminator, gender_estimation)
        # with open('VIT_keys_after_ddp.txt', 'w') as f:
        #         for key in estimation_network.state_dict().keys():
        #             f.write(key + '\n')
        scaler = amp.GradScaler()
        self.optimizer = optimizer
        # self.optimizer = optimizer_new
        self.backbone = backbone
        self.head = head
        self.estimation_network = estimation_network
        self.gender_network = gender_estimation
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
            self.estimation_network.train()
            self.backbone.train()
            images, ages = inputs
            if opt.amp:
                with amp.autocast():
                    embedding, x_id, x_age = self.backbone(
                        images, return_age=True)
                embedding = embedding.float()
                x_id = x_id.float()
                x_age = x_age.float()
            else:
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
                embedding, x_id, x_age, x_gender = self.backbone(
                    images, return_gender=True)

        if opt.gfr:
            # Train GFR only
            x_age, x_group = self.estimation_network(x_age)
            # out = get_dex_age(x_age)
            # print(out)
            # print("-------------------------------------")
            # print(ages)
            # age_loss = F.mse_loss(torch.round(out * 10) / 10, ages)
            # age_group_loss = F.cross_entropy(x_group, age2group(
            #     ages, age_group=opt.age_group).long())
            age_loss = self.compute_age_loss(x_age, x_group, ages)
            # da_loss = self.forward_da(x_id, ages)
            # loss = age_loss * opt.fr_age_loss_weight + \
            #     da_loss * opt.fr_da_loss_weight
            if opt.amp:
                age_loss = self.scaler.scale(age_loss)
            self.optimizer.zero_grad()
            age_loss.backward()
            if opt.amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            apply_weight_decay(self.estimation_network,
                               weight_decay_factor=opt.weight_decay, wo_bn=True)
            # age_loss = reduce_loss(age_loss)
            # self.adjust_learning_rate(n_iter)
            # lr = self.optimizer.param_groups[0]['lr']
            self.logger.msg([age_loss, lr], n_iter)
        else:
            # Train Face Recognition with ages and genders
            id_loss = F.cross_entropy(self.head(embedding, labels), labels)

            # If using VIT then feed images directly to estimation network
            # if opt.td_block:

            # x_age, x_group = self.estimation_network(images)

            x_age, x_group = self.estimation_network(x_age)
            age_loss = self.compute_age_loss(x_age, x_group, ages)
            da_loss = self.forward_da(x_id, ages)

            # Gender
            x_genders = self.gender_network(x_gender)
            gender_loss = F.cross_entropy(x_genders, genders)

            loss = id_loss + \
                age_loss * opt.fr_age_loss_weight + \
                da_loss * opt.fr_da_loss_weight + gender_loss * opt.fr_gender_loss_weight

            # loss = id_loss + opt.fr_age_loss_weight * age_loss
            total_loss = loss

            if opt.amp:
                total_loss = self.scaler.scale(loss)
            self.optimizer.zero_grad()
            total_loss.backward()
            apply_weight_decay(self.backbone, self.head, self.estimation_network, self.gender_network,
                               weight_decay_factor=opt.weight_decay, wo_bn=True)
            if opt.amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            id_loss, da_loss, age_loss, gender_loss = reduce_loss(
                id_loss, da_loss, age_loss, gender_loss)
            self.adjust_learning_rate(n_iter)
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.msg(
                [id_loss, da_loss, age_loss, gender_loss, lr], n_iter)

            # id_loss,  age_loss = reduce_loss(
            #     id_loss, age_loss)
            # self.adjust_learning_rate(n_iter)
            # lr = self.optimizer.param_groups[0]['lr']
            # self.logger.msg([id_loss, age_loss, lr], n_iter)

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
                image, age = self.prefetcher.next()
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
                image, age, gender = self.prefetcher.next()
                embedding, x_id, x_age, x_gender = self.backbone(
                    image, return_gender=True)
                predicted_sex = self.gender_network(x_gender)
                predicted_sex = torch.argmax(predicted_sex).item()
                if predicted_sex == gender.item():
                    total_correct_pred += 1
                else:
                    total_incorrect_pred +=1
                    print(f'The predicted sex {predicted_sex}')
                    print(f'The actual sex {gender.item()}')
            accuracy = total_correct_pred / \
                (total_correct_pred+total_incorrect_pred)
            print(f'Total correct predictions are {total_correct_pred}')
            print(f'Total Incorrect predictions are {total_incorrect_pred}')
            print(f'Accuracy of Age estimation model : {accuracy}')
