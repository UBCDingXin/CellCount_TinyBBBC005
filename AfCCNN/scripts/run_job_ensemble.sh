#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:v100l:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00
#SBATCH --mail-user=xin.ding@stat.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=Af1_2

module load arch/avx512 StdEnv/2018.3
module load cuda/10.0.130
module load python/3.7.4
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements.req


ROOT_PATH="/scratch/dingx92/Annotation-free_Cell_Counting/AfCCNN"
TINY_BBBC005_DATA_PATH='/scratch/dingx92/Annotation-free_Cell_Counting/Tiny-BBBC005_256x256.h5'
SEED=2020
BACKBONE_CNN='ResNet34'


####################################################################################################################################
# Regression CNN
echo "-------------------------------------------------------------------------------------------------"
echo "exp2_rd1: ${BACKBONE_CNN}(Reg)"
CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
--experiment_name exp2_rd1 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name $BACKBONE_CNN --predtype 'regre' \
--epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
--lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
2>&1 | tee output_exp2_rd1_${BACKBONE_CNN}_Reg_seed_${SEED}.txt

echo "-------------------------------------------------------------------------------------------------"
echo "exp2_rd2: ${BACKBONE_CNN}(Reg)"
CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
--experiment_name exp2_rd2 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name $BACKBONE_CNN --predtype 'regre' \
--epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
--lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
2>&1 | tee output_exp2_rd2_${BACKBONE_CNN}_Reg_seed_${SEED}.txt

echo "-------------------------------------------------------------------------------------------------"
echo "exp2_rd3: ${BACKBONE_CNN}(Reg)"
CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
--experiment_name exp2_rd3 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name $BACKBONE_CNN --predtype 'regre' \
--epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
--lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
2>&1 | tee output_exp2_rd3_${BACKBONE_CNN}_Reg_seed_${SEED}.txt




####################################################################################################################################
# Classification CNN + Ensembling with LQReg
echo "-------------------------------------------------------------------------------------------------"
echo "exp2_rd1: ${BACKBONE_CNN}(Cl)+EwLQReg"
CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
--experiment_name exp2_rd1 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name $BACKBONE_CNN --predtype 'class' \
--epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
--lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
--ensemble --ensemble_regre_model 'LQReg' \
2>&1 | tee output_exp2_rd1_${BACKBONE_CNN}_Cl_EwLQReg_seed_${SEED}.txt

echo "-------------------------------------------------------------------------------------------------"
echo "exp2_rd2: ${BACKBONE_CNN}(Cl)+EwLQReg"
CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
--experiment_name exp2_rd2 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name $BACKBONE_CNN --predtype 'class' \
--epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
--lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
--ensemble --ensemble_regre_model 'LQReg' \
2>&1 | tee output_exp2_rd2_${BACKBONE_CNN}_Cl_EwLQReg_seed_${SEED}.txt

echo "-------------------------------------------------------------------------------------------------"
echo "exp2_rd3: ${BACKBONE_CNN}(Cl)+EwLQReg"
CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
--experiment_name exp2_rd3 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name $BACKBONE_CNN --predtype 'class' \
--epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
--lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
--ensemble --ensemble_regre_model 'LQReg' \
2>&1 | tee output_exp2_rd3_${BACKBONE_CNN}_Cl_EwLQReg_seed_${SEED}.txt




####################################################################################################################################
# Classification CNN + DA + Ensembling with LQReg
echo "-------------------------------------------------------------------------------------------------"
echo "exp2_rd1: ${BACKBONE_CNN}(Cl)+DA+EwLQReg"
CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
--experiment_name exp2_rd1 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name $BACKBONE_CNN --predtype 'class' \
--epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
--lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
--dataAugment --da_flip --da_filter \
--ensemble --ensemble_regre_model 'LQReg' \
2>&1 | tee output_exp2_rd1_${BACKBONE_CNN}_Cl_DA_EwLQReg_seed_${SEED}.txt

echo "-------------------------------------------------------------------------------------------------"
echo "exp2_rd2: ${BACKBONE_CNN}(Cl)+DA+EwLQReg"
CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
--experiment_name exp2_rd2 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name $BACKBONE_CNN --predtype 'class' \
--epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
--lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
--dataAugment --da_flip --da_filter \
--ensemble --ensemble_regre_model 'LQReg' \
2>&1 | tee output_exp2_rd2_${BACKBONE_CNN}_Cl_DA_EwLQReg_seed_${SEED}.txt

echo "-------------------------------------------------------------------------------------------------"
echo "exp2_rd3: ${BACKBONE_CNN}(Cl)+DA+EwLQReg"
CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
--experiment_name exp2_rd3 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name $BACKBONE_CNN --predtype 'class' \
--epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
--lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
--dataAugment --da_flip --da_filter \
--ensemble --ensemble_regre_model 'LQReg' \
2>&1 | tee output_exp2_rd3_${BACKBONE_CNN}_Cl_DA_EwLQReg_seed_${SEED}.txt
