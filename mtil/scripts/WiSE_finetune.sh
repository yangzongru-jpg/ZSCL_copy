#!/bin/bash
set -v
set -e
set -x

exp_no=app_004
GPU=2,3
dataset=(StanfordCars Food MNIST OxfordPet Flowers SUN397 Aircraft Caltech101 DTD EuroSAT CIFAR100)
lr=(1e-5 1e-5 5e-5 1e-5 1e-5 1e-5 5e-5 1e-5 1e-5 1e-5 1e-5)

# first dataset
CUDA_VISIBLE_DEVICES=${GPU} python -m src.main \
    --train-mode=whole \
    --train-dataset=${dataset[0]} \
    --lr=${lr[0]} \
    --ls 0.2 \
    --iterations 1000 \
    --method finetune \
    --wise_merge \
    --wise_ft_model n \
    --wise_ft_alpha 0.8 \
    --ref-dataset ImageNet \
    --save ckpt/exp_${exp_no}

for ((i = 1; i < ${#dataset[@]}; i++)); do
    dataset_cur=${dataset[i]}
    dataset_pre=${dataset[i - 1]}

    # continue training
    CUDA_VISIBLE_DEVICES=${GPU} python -m src.main \
        --train-mode=whole \
        --train-dataset=${dataset_cur} \
        --lr=${lr[i]} \
        --ls 0.2 \
        --method finetune \
        --wise_merge \
        --wise_ft_model n \
        --wise_ft_alpha 0.8 \
        --ref-dataset ImageNet \
        --iterations 1000 \
        --save ckpt/exp_${exp_no} \
        --load ckpt/exp_${exp_no}/${dataset_pre}.pth
done
