#!/bin/bash
#SBATCH --account=def-jhchen
#SBATCH --gres=gpu:v100l:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-08:00
#SBATCH --mail-user=qiong.zhang@stat.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=DR1_1

module load arch/avx512 StdEnv/2018.3
module load cuda/10.0.130
module load python/3.7.4
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements.req


ROOT_PATH="/scratch/CellCount_cedar_MSE/FPNCNN"
TINY_BBBC005_DATA_PATH='/scratch/CellCount_cedar_MSE/dataset/Tiny-BBBC005_256x256.h5'
SEED=2020


echo "-------------------------------------------------------------------------------------------------"
echo "Experiment 1 SEED=$SEED"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp1 \
--cnn_name VGG19 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 1 \
--cnn_epochs 120 --cnn_resume_epoch 0 \
--cnn_lr_base 1e-3 --cnn_lr_decay_factor 0.1 --cnn_lr_decay_epochs '20_50_80' \
--cnn_batch_size_train 64 --cnn_batch_size_test 64 \
--fpn_epochs 50 --fpn_resume_epoch 0 \
--fpn_lr_base 1e-3 --fpn_lr_decay_factor 0.1 --fpn_lr_decay_epochs '20_30_40' --fpn_weight_decay 1e-5 \
--fpn_batch_size_train 2 \
2>&1 | tee output_exp1_seed_${SEED}.txt

