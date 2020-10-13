#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:v100l:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-04:00
#SBATCH --mail-user=xin.ding@stat.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=Afr_2

module load arch/avx512 StdEnv/2018.3
module load cuda/10.0.130
module load python/3.7.4
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements.req


ROOT_PATH="/scratch/dingx92/Annotation-free_Cell_Counting/AfCCNN"
TINY_BBBC005_DATA_PATH='/scratch/dingx92/Annotation-free_Cell_Counting/Tiny-BBBC005_256x256.h5'
SEED=2021
BACKBONE_CNN='ResNet34'
NTRAIN=1200

echo "-------------------------------------------------------------------------------------------------"
echo "exp1: ${BACKBONE_CNN}(Reg)"
CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
--experiment_name exp1_reduced \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name $BACKBONE_CNN --predtype 'regre' \
--epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
--lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
--num_train $NTRAIN \
2>&1 | tee output_exp1_reduced_${BACKBONE_CNN}_Reg_seed_${SEED}.txt

echo "-------------------------------------------------------------------------------------------------"
echo "exp1: ${BACKBONE_CNN}(Cl)"
CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
--experiment_name exp1_reduced \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name $BACKBONE_CNN --predtype 'class' \
--epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
--lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
--num_train $NTRAIN \
2>&1 | tee output_exp1_reduced_${BACKBONE_CNN}_Cl_seed_${SEED}.txt


echo "-------------------------------------------------------------------------------------------------"
echo "exp1: ${BACKBONE_CNN}(Cl) Ensemble with ResNet34"
CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
--experiment_name exp1_reduced \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name $BACKBONE_CNN --predtype 'class' \
--epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
--lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
--ensemble --ensemble_regre_model 'ResNet34' --ensemble_regre_cnn_path 'ckpt_exp1_reduced_regre_ResNet34_epochs_100_transform_True_dataAugment_False_seed_2020.pth' \
--num_train $NTRAIN \
2>&1 | tee output_exp1_reduced_${BACKBONE_CNN}_Cl_EwResNet34_seed_${SEED}.txt
