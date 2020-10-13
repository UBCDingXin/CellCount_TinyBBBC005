#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-12:00
#SBATCH --mail-user=xin.ding@stat.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=FP_2

module load arch/avx512 StdEnv/2018.3
module load cuda/10.0.130
module load python/3.7.4
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements.req


ROOT_PATH="/scratch/dingx92/Annotation-free_Cell_Counting/FPNCNN"
TINY_BBBC005_DATA_PATH='/scratch/dingx92/Annotation-free_Cell_Counting/Tiny-BBBC005_256x256.h5'
VGG_DATA_PATH='/scratch/dingx92/Annotation-free_Cell_Counting/VGG_dataset.h5'
SEED=2020


echo "-------------------------------------------------------------------------------------------------"
echo "Experiment 2 SEED=$SEED Round 1"
CUDA_VISIBLE_DEVICES=0 python3 main.py \
--experiment_name exp2_rd1 \
--cnn_name VGG19 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--deleted_counts '14_35_57_66_83' \
--cnn_epochs 120 --cnn_resume_epoch 0 \
--cnn_lr_base 1e-3 --cnn_lr_decay_factor 0.1 --cnn_lr_decay_epochs '20_50_80' \
--cnn_batch_size_train 64 --cnn_batch_size_test 64 \
--fpn_epochs 60 --fpn_resume_epoch 0 \
--fpn_lr_base 1e-3 --fpn_lr_decay_factor 0.1 --fpn_lr_decay_epochs '10_20_35' --fpn_weight_decay 1e-5 \
--fpn_batch_size_train 2 \
2>&1 | tee output_exp2_rd1_seed_${SEED}.txt


echo "-------------------------------------------------------------------------------------------------"
echo "Experiment 2 SEED=$SEED Round 2"
CUDA_VISIBLE_DEVICES=0 python3 main.py \
--experiment_name exp2_rd2 \
--cnn_name VGG19 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--deleted_counts '10_31_70_83_91' \
--cnn_epochs 120 --cnn_resume_epoch 0 \
--cnn_lr_base 1e-3 --cnn_lr_decay_factor 0.1 --cnn_lr_decay_epochs '20_50_80' \
--cnn_batch_size_train 64 --cnn_batch_size_test 64 \
--fpn_epochs 60 --fpn_resume_epoch 0 \
--fpn_lr_base 1e-3 --fpn_lr_decay_factor 0.1 --fpn_lr_decay_epochs '10_20_35' --fpn_weight_decay 1e-5 \
--fpn_batch_size_train 2 \
2>&1 | tee output_exp2_rd2_seed_${SEED}.txt


echo "-------------------------------------------------------------------------------------------------"
echo "Experiment 2 SEED=$SEED Round 3"
CUDA_VISIBLE_DEVICES=0 python3 main.py \
--experiment_name exp2_rd3 \
--cnn_name VGG19 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--deleted_counts '18_27_44_53_91' \
--cnn_epochs 120 --cnn_resume_epoch 0 \
--cnn_lr_base 1e-3 --cnn_lr_decay_factor 0.1 --cnn_lr_decay_epochs '20_50_80' \
--cnn_batch_size_train 64 --cnn_batch_size_test 64 \
--fpn_epochs 60 --fpn_resume_epoch 0 \
--fpn_lr_base 1e-3 --fpn_lr_decay_factor 0.1 --fpn_lr_decay_epochs '10_20_35' --fpn_weight_decay 1e-5 \
--fpn_batch_size_train 2 \
2>&1 | tee output_exp2_rd3_seed_${SEED}.txt
