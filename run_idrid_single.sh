#!/bin/bash


gpu_id=0
dataset='idrid'
name='idrid'
data_root='./datasets/idrid'
cp='checkpoint/r18-checkpoint.pkl'


for w in {0.0,0.1,0.3,0.5,0.7,0.9,1.0}
do
m='AggdNet'
CUDA_VISIBLE_DEVICES=${gpu_id} python train_cls.py --name $name --dataset $dataset --data_root $data_root --method $m --pretrained_dir $cp --epochs 180 --learning_rate 0.0005 --train_batch_size 8 --loss_type 'aggd_loss' --weight $w --head_type 'one_layer' --var_type 'one_layer'
done