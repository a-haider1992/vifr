import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from common.ops import load_network, load_network_1

from .fas import FAS
from .fr import FR
from .td_block import TDTask

'''
python -m torch.distributed.launch --nproc_per_node=8 --master_port=17647 main.py \
    --train_fr --backbone_name ir50 --head_s 64 --head_m 0.35 \
    --weight_decay 5e-4 --momentum 0.9 --fr_age_loss_weight 0.001 --fr_da_loss_weight 0.002 --age_group 7 \
    --gamma 0.1 --milestone 20000 23000 --warmup 1000 --learning_rate 0.1 \
    --dataset_name scaf --image_size 112 --num_iter 36000 --batch_size 64 --amp

python -m torch.distributed.launch --nproc_per_node=8 --master_port=17647 main.py \
    --train_fas --backbone_name ir50 --age_group 7 \
    --dataset_name scaf --image_size 112 --num_iter 36000 --batch_size 64 \
    --d_lr 1e-4 --g_lr 1e-4 --fas_gan_loss_weight 75 --fas_age_loss_weight 10 --fas_id_loss_weight 0.002
'''


class MTLFace(object):

    def __init__(self, opt):
        self.opt = opt
        self.fr = FR(opt)
        self.fr.set_loader()
        self.fr.set_model()
        if opt.train_fas:
            if opt.id_pretrained_path is not None and dist.get_rank() == 0:
                self.fr.backbone.load_state_dict(
                    torch.load(opt.id_pretrained_path))
            if opt.age_pretrained_path is not None and dist.get_rank() == 0:
                self.fr.estimation_network.load_state_dict(
                    torch.load(opt.age_pretrained_path))
            if opt.gender_pretrained_path is not None and dist.get_rank() == 0:
                self.fr.gender_network.load_state_dict(
                    torch.load(opt.gender_pretrained_path))
            if opt.race_pretrained_path is not None and dist.get_rank() == 0:
                self.fr.race_network.load_state_dict(
                    torch.load(opt.race_pretrained_path))
            if not opt.evaluation_only:
                self.fas = FAS(opt)
                self.fas.set_loader()
                self.fas.set_model()

    @staticmethod
    def parser():
        parser = argparse.ArgumentParser()

        parser.add_argument("--train_fr", help='train_fr', action='store_true')
        parser.add_argument("--train_fas", help='train_fas',
                            action='store_true')

        # BACKBONE, HEAD
        parser.add_argument("--backbone_name", help='backbone', type=str)
        parser.add_argument(
            "--head_s", help='s of cosface or arcface', type=float, default=64)
        parser.add_argument(
            "--head_m", help='m of cosface or arcface', type=float, default=0.35)

        # OPTIMIZED
        parser.add_argument(
            "--weight_decay", help='weight-decay', type=float, default=5e-4)
        parser.add_argument("--momentum", help='momentum',
                            type=float, default=0.9)

        # LOSS
        parser.add_argument("--fr_id_loss_weight",
                            help='id loss weight', type=float, default=1.0)
        parser.add_argument("--fr_age_loss_weight",
                            help='age loss weight', type=float, default=0.0)
        parser.add_argument("--fr_gender_loss_weight",
                            help='gender loss weight', type=float, default=0.0)
        parser.add_argument("--fr_race_loss_weight",
                            help='race loss weight', type=float, default=0.0)
        parser.add_argument("--fr_da_loss_weight", help='cross age domain adaption loss weight', type=float,
                            default=0.0)
        parser.add_argument("--age_group", help='age_group',
                            default=7, type=int)

        # LR
        parser.add_argument(
            "--gamma", help='learning-rate gamma', type=float, default=0.1)
        parser.add_argument("--milestone", help='milestones',
                            type=int, nargs='*', default=[20, 40, 60])
        parser.add_argument(
            "--warmup", help='learning rate warmup epoch', type=int, default=5)
        parser.add_argument("--learning_rate",
                            help='learning-rate', type=float, default=0.1)

        # TRAINING
        parser.add_argument("--dataset_name", "-d",
                            help='input image size', type=str)
        parser.add_argument(
            "--image_size", help='input image size', default=224, type=int)
        parser.add_argument(
            "--num_iter", help='total epochs', type=int, default=125)
        parser.add_argument(
            "--restore_iter", help='restore_iter', default=0, type=int)
        parser.add_argument(
            "--batch_size", help='batch-size', default=0, type=int)
        parser.add_argument(
            "--val_interval", help='val dataset interval iteration', type=int, default=1000)

        parser.add_argument('--seed', type=int, default=1,
                            metavar='S', help='random seed (default: 1)')
        parser.add_argument(
            "--num_worker", help='dataloader num-worker', default=32, type=int)
        parser.add_argument(
            "--local_rank", help='local process rank, not need to be set.', default=0, type=int)

        parser.add_argument("--amp", help='amp', action='store_true')

        parser.add_argument("--model_save", "-s",
                            help='save trained model path', type=str)
        parser.add_argument("--td_block", help='Use VIT', action='store_true')
        parser.add_argument(
            "--vit_hidden_f", help='number of hidden features in VIT', default=32, type=int)
        parser.add_argument(
            "--vit_heads", help='number of heads in VIT', default=4, type=int)
        parser.add_argument(
            "--vit_blocks", help='number of VIT blocks', default=2, type=int)

        # GENERAL FACE RECOGNITION
        parser.add_argument("--eval_gender", help='evaluate gender network True/False',
                            action='store_true')
        parser.add_argument("--age_protocol",
                            help='Age protocol', type=str)
        

        # TESTING
        parser.add_argument("--evaluation_dataset",
                            help='Evaluation Dataset', type=str)
        parser.add_argument("--evaluation_num_iter",
                            help='Number of evaluation epochs', default=5, type=int)
        parser.add_argument("--eval_batch_size",
                            help='Batch size of evaluation', default=1, type=int)
        parser.add_argument(
            "--evaluation_only", help='Evaluate the trained models', action='store_true')
        parser.add_argument(
            "--fine_tune", help='Fine tune the pre-trained models', action='store_true')
        parser.add_argument("--utk_eval_file",
                            help='UTK dataset evaluation file', type=str)
        

        # FAS
        parser.add_argument("--d_lr", help='learning-rate',
                            type=float, default=1e-4)
        parser.add_argument("--g_lr", help='learning-rate',
                            type=float, default=1e-4)
        parser.add_argument("--fas_gan_loss_weight",
                            help='gan_loss_weight', type=float)
        parser.add_argument("--fas_id_loss_weight",
                            help='id_loss_weight', type=float)
        parser.add_argument("--fas_age_loss_weight",
                            help='age_loss_weight', type=float)
        parser.add_argument("--id_pretrained_path",
                            help='id_pretrained_path', type=str)
        parser.add_argument("--age_pretrained_path",
                            help='age_pretrained_path', type=str)
        parser.add_argument("--gender_pretrained_path",
                            help='gender_pretrained_path', type=str)
        parser.add_argument("--race_pretrained_path",
                            help='race_pretrained_path', type=str)

        return parser

    def fit(self):
        opt = self.opt
        # training routine
        train_losses = open('training_loss.csv', 'w')
        train_losses.write("Id-Loss"+","+"Age-Loss"+","+"Gender-Loss"+","+"Race-Loss"+","+"MTL-Loss"+"\n")
        for n_iter in tqdm.trange(opt.restore_iter + 1, opt.num_iter + 1, disable=(opt.local_rank != 0)):
            # img, label, age, gender
            fr_inputs = self.fr.prefetcher.next()
            if opt.train_fr:
                l1, l2, gl, rl, l3 = self.fr.train(fr_inputs, n_iter)
                train_losses.write(str(l1)+","+str(l2)+","+str(gl)+","+str(rl)+","+str(l3.item())+"\n")
            if opt.train_fas:
                # target_img, target_label
                fas_inputs = self.fas.prefetcher.next()
                # backbone, age_estimation, source_img, target_img, source_label, target_label
                # You can also use other attributes for aligning
                _fas_inputs = [self.fr.backbone.module, self.fr.estimation_network,
                               fr_inputs[0], fas_inputs[0], fr_inputs[1], fas_inputs[1]]
                self.fas.train(_fas_inputs, n_iter)
            if n_iter % opt.val_interval == 0:
                if opt.train_fr:
                    self.fr.validate(n_iter)
                if opt.train_fas:
                    self.fas.validate(n_iter)
        train_losses.close()

    def save_model(self):
        opt = self.opt
        if dist.get_rank() == 0:
            root = os.path.dirname(__file__)
            PATH_BACKBONE = os.path.join(root, 'backbone.pt')
            PATH_AGE_ESTIMATION = os.path.join(
                root, 'age_estimation_model.pt')
            PATH_GENDER_MODEL = os.path.join(root, 'gender_model.pt')
            PATH_RACE_MODEL = os.path.join(root, 'race_model.pt')
            torch.save(self.fr.backbone.state_dict(), PATH_BACKBONE)
            torch.save(self.fr.estimation_network.state_dict(),
                        PATH_AGE_ESTIMATION)
            torch.save(self.fr.gender_network.state_dict(),
                        PATH_GENDER_MODEL)
            torch.save(self.fr.race_network.state_dict(),
                        PATH_RACE_MODEL)
                
    def calculateMetrics(self, a, b):
        # Compute Euclidean distance
        euclidean_distance = torch.norm(a - b, p=2)

        # Compute cosine similarity
        cosine_similarity = F.cosine_similarity(a, b).item()

        # Compute correlation coefficient
        cov = torch.mean((a - torch.mean(a)) * (b - torch.mean(b)))
        std_a = torch.std(a)
        std_b = torch.std(b)
        corr_coef = cov / (std_a * std_b)
        corr_coef = corr_coef.item()

        # Compute mean squared error
        mean_squared_error = F.mse_loss(a, b)
        return euclidean_distance, cosine_similarity, corr_coef, mean_squared_error

    def isSame(self, embed1, embed2):
        # result = torch.eq(embed1, embed2)
        # num_ones = torch.sum(result == 1).item()
        # if num_ones > (result.size()[0] * result.size()[1])/6.0:
        #     return True
        # else:
        #     return False
        # diff = torch.abs(embed1 - embed2)
        # # Check if the maximum difference is small enough
        # if diff.max() < 1e-2:
        #     return True
        # else:
        #     return False

        # cosine_sim = cosine_similarity(embed2.detach().cpu().numpy(), 
        #                                       embed2.detach().cpu().numpy())
        
        # sim_threshold = 0.5
        
        # num_correct = (cosine_sim >= sim_threshold).sum().item()
        # cosine_accuracy = num_correct / cosine_sim.shape[0]

        similarity_error = torch.mean(torch.abs(embed1 - embed2))
        # normalize the mean absolute difference between 0 and 1
        # max_value = torch.max(similarity_error)
        # if max_value != 0 and not torch.isnan(max_value):
        #     normalized_error = similarity_error / max_value
        # else:
        #     normalized_error = similarity_error
        cov = torch.mean((embed1 - torch.mean(embed1)) * (embed2 - torch.mean(embed2)))
        std_a = torch.std(embed1)
        std_b = torch.std(embed2)
        corr_coef = cov / (std_a * std_b)
        return similarity_error.item(), corr_coef.item()
    
    def checkEq(self, embed1, embed2):
        result = torch.eq(embed1, embed2)
        num_ones = torch.sum(result == 1).item()
        if num_ones > (result.size()[0] * result.size()[1])/4.0:
            return True
        else:
            return False

    def evaluate_age_estimation(self):
        self.fr.age_pretrained_eval()

    def evaluate_gender_estimation(self):
        self.fr.evaluate_gender_model()

    def evaluate_mtlface(self):
        # evaluate trained model
        from skimage.io import imsave
        from PIL import Image
        import torchvision.transforms as transforms
        opt = self.opt
        torch.cuda.empty_cache()
        print("MTL Face is under evaluation.")
        self.fr.backbone.eval()
        # total_correct_pred = 0
        # total_incorrect_pred = 0
        total_iter = int(opt.evaluation_num_iter)
        mean_euc_dis = mean_cos_sim = mean_corr_coeff = mean_mse = 0.0
        with torch.no_grad():
            for _ in range(0, total_iter):
                image1, image2 = self.fr.prefetcher.next()
                # embedding1, x_id1, x_age1, x_residual1 = self.fr.backbone(
                #     image1, return_residual=True)
                # embedding2, x_id2, x_age2, x_residual2 = self.fr.backbone(
                #     image2, return_residual=True)
                embedding2, x_id2, x_age2 = self.fr.backbone(
                    image1, return_age=True)
                x1, x2, x3, x4, x5, _, _ = self.fr.backbone(
                    image1, return_shortcuts=True)
                # embedding1 = self.fr.backbone(image1)
                # embedding2 = self.fr.backbone(image2)
                # if self.checkEq(embedding1, embedding2):
                #     total_correct_pred += 1
                # else:
                #     total_incorrect_pred += 1
                # similarity_error, mean_corr, cosine_acc = self.isSame(embedding1, embedding2)
                # euc_dis, cos_sim, corr_coeff, mse = self.calculateMetrics(embedding1, embedding2)
                # mean_euc_dis += euc_dis
                # mean_cos_sim += cos_sim
                # mean_corr_coeff += corr_coeff
                # mean_mse += mse
                # if similarity_error <= 1e-4 or mean_corr >= 0.5:
                #     total_correct_pred += 1
                # else:
                #     total_incorrect_pred += 1
            # mean_euc_dis, mean_cos_sim, mean_corr_coeff, mean_mse = mean_euc_dis / total_iter, mean_cos_sim / total_iter, mean_corr_coeff / total_iter, mean_mse / total_iter
            # print(f'The mean euclidean distance: {mean_euc_dis}')
            # print(f'The mean cosine similarity: {mean_cos_sim}')
            # print(f'The mean correlation: {mean_corr_coeff}')
            # print(f'The mean MSE: {mean_mse}')
            # print(f'Embedding1 shape : {embedding1.shape}')
            # print(f'Age shape : {x_age1.shape}')
            # print(f'Residual shape : {x_residual1.shape}')
            # print(f'Id shape : {x_id1.shape}')

            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((112, 112))
            ])
            x = x_id2
            for i in range(x.shape[1] // 3):
                # if i > 5:
                #     break
                start_channel = i * 3
                end_channel = start_channel + 3
                img = transform(x[:, start_channel:end_channel, :, :].squeeze())
                img.save(f"tensor_image_{i}.png")

            # x_age1 = x_age1.flatten()
            # encoded_array = x_age1.view(112, -1).cpu().numpy()
            # encoded_array = (encoded_array - np.min(encoded_array)) / (np.max(encoded_array) - np.min(encoded_array)) * 255
            # encoded_image = Image.fromarray(encoded_array.astype(np.uint8), mode='L')
            # encoded_image.save('age1.jpg')
            # imsave('age1.png', encoded_image)
            # imsave('embed2.jpg', embedding2)
            # imsave('age_1.jpg', x_age1)
            # imsave('age_2.jpg', x_age2)
            # imsave('id_1.jpg', x_id1)
            # imsave('id_2.jpg', x_id2)
            # imsave('residual_1.jpg', x_residual1)
            # imsave('residual_2.jpg', x_residual2)
            # # print("Model Accuracy:{}".format(1 - (total_eval_error / total_iter)))
            # print(f'The mean cosine similarity: {cosine_acc}')
            # print("Average Correlation between prediction and true labels :{}".format(avg_corr / total_iter))
            # print("During evaluation, the model corectly predicts {} number of classes.".format(
            #     total_correct_pred))
            # print("During evaluation, the model incorrectly predicts {} number of classes.".format(
            #     total_incorrect_pred))
        # with torch.no_grad():
        #     for _ in range(0, total_iter):
        #         image, label = self.fr.eval_prefetcher.next()
        #         embed = self.fr.backbone(image)
        #         pred_label = self.fr.head(embed, torch.tensor(0, dtype=torch.int32))
        #         id_loss = F.cross_entropy(pred_label, label)
        #         total_loss += id_loss
        #         if torch.argmax(pred_label).item() == label.item():
        #             total_correct_pred += 1
        #         else:
        #             total_incorrect_pred += 1
        #             print("Predicted label:{} and actual label {}".format(torch.argmax(pred_label).item(), label.item()))
        #     print("-----------------------------Summary------------------------------")
        #     print("The size of predicted tensor:{}".format(pred_label.size()))
        #     print("The MTLFace model is trained for {} iterations.".format(opt.num_iter))
        #     print("The MTLFace model is evaluated {} times.".format(total_iter))
        #     print("Average Identity loss in evaluation:{}".format(total_loss/total_iter))
        #     print("During evaluation, the model corectly predicts {} number of classes.".format(total_correct_pred))
        #     print("During evaluation, the model incorrectly predicts {} number of classes.".format(total_incorrect_pred))
        #     print("Model Accuracy:{}".format(total_correct_pred/(total_correct_pred+total_incorrect_pred)))
        #     print("-----------------------------oooooooooooooooo---------------------------------")
        #     # embed, id_ten, age_ten = self.fr.backbone(img.unsqueeze(0), return_age=True)
        #     # pred_label = self.fr.head(embed, torch.tensor(0, dtype=torch.int8))
        #     # with open('evaluation.txt', 'w') as f:
        #     #     f.write("The predicted class---------------------------")
        #     #     f.write(str(torch.argmax(pred_label)))
