#!/bin/bash


gpu_id=0
dataset='kneeoa'
name='kneeoa_backbone3'
data_root='./datasets/kneeoa'
cp=None
# --pretrained_dir $cp



for w in {0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}
do
m='AggdNet'
CUDA_VISIBLE_DEVICES=${gpu_id} python train_cls.py --name $name --dataset $dataset --data_root $data_root --method $m --epochs 180 --learning_rate 5e-4 --train_batch_size 8 --loss_type 'aggd_loss' --weight $w --head_type 'one_layer' --var_type 'one_layer' --backbone 'efficientnet_b1'
done