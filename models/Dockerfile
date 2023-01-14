# syntax=docker/dockerfile:1

FROM python:3.7-slim-buster

WORKDIR /

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3 -m torch.distributed.launch --nproc_per_node=4 main.py --train_fas --backbone_name ir50 --age_group 7 --dataset_name casia-webface --image_size 112 --num_iter 36000 --batch_size 64 --d_lr 1e-4 --g_lr 1e-4 --fas_gan_loss_weight 75 --fas_age_loss_weight 10 --fas_id_loss_weight 0.002"]