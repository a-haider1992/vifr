FROM python:3

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "-m torch.distributed.launch --nproc_per_node=4 main.py --train_fas --backbone_name ir50 --age_group 7 --dataset_name casia-webface --image_size 112 --num_iter 36000 --batch_size 64 --d_lr 1e-4 --g_lr 1e-4 --fas_gan_loss_weight 75 --fas_age_loss_weight 10 --fas_id_loss_weight 0.002"]

# CMD [ "python3", "-m torch.distributed.launch --nproc_per_node=4 main.py --train_fas --backbone_name ir50 --age_group 7 --dataset_name casia-webface --image_size 112 --num_iter 36000 --batch_size 64 --d_lr 1e-4 --g_lr 1e-4 --fas_gan_loss_weight 75 --fas_age_loss_weight 10 --fas_id_loss_weight 0.002" ]