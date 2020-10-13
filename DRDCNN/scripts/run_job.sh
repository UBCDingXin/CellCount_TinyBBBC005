#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:v100l:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-08:00
#SBATCH --mail-user=xin.ding@stat.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=DR1_1

module load arch/avx512 StdEnv/2018.3
module load cuda/10.0.130
module load python/3.7.4
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements.req


ROOT_PATH="/scratch/dingx92/Annotation-free_Cell_Counting/DRDCNN"
VGG_DATA_PATH='/scratch/dingx92/Annotation-free_Cell_Counting/VGG_dataset.h5'
TINY_BBBC005_DATA_PATH='/scratch/dingx92/Annotation-free_Cell_Counting/Tiny-BBBC005_256x256.h5'
SEED=2020


echo "-------------------------------------------------------------------------------------------------"
echo "Experiment 1 SEED=$SEED"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp1 \
--root $ROOT_PATH \
--path_vgg_dataset $VGG_DATA_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--unet_epochs 200 --unet_resume_epoch 0 \
--unet_lr_base 1e-3 --unet_lr_decay_factor 0.01 --unet_lr_decay_epochs '100_150' \
--unet_batch_size_train 16 --unet_transform \
--cnn_epochs 120 --cnn_resume_epoch 0 \
--cnn_lr_base 1e-2 --cnn_lr_decay_factor 0.1  --cnn_lr_decay_epochs '20_50_80' \
--cnn_batch_size_train 64 --cnn_batch_size_test 64 --cnn_transform \
2>&1 | tee output_exp1_seed_${SEED}.txt
