#!/bin/bash
#SBATCH --ntasks-per-node=4 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D . # working directory
#SBATCH -t 0-05:00 # Runtime in D-HH:MM
#SBATCH -p mlow # Partition to submit to
#SBATCH -q masterhigh  # This way will only requeue of dcc partition
#SBATCH --mem 32000 # 4GB memory
#SBATCH --gres gpu:1 # Request of 1 gpu
#SBATCH -o out/%j.out # File to which STDOUT will be written
#SBATCH -e out/%j.err # File to which STDERR will be written

sleep 5
/ghome/share/example/deviceQuery
nvidia-smi

# Set env vars
set -a # automatically export all variables
source /ghome/c5mcv07/C5_G7_MCV/.env
set +a

python train.py \
    --model_name_or_path facebook/mask2former-swin-tiny-coco-instance \
    --output_dir checkpoints/Mask2Former_KITTI_v1 \
    --dataset_name marcagve18/kitti-mots-instance-seg \
    --do_reduce_labels \
    --do_train \
    --fp16 \
    --num_train_epochs 40 \
    --learning_rate 1e-5 \
    --lr_scheduler_type constant \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 8 \
    --dataloader_persistent_workers \
    --dataloader_prefetch_factor 4 \
    --do_eval \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2 \
    --token $HF_MARC \
    --report_to wandb \
    --push_to_hub
