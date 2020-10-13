ROOT_PATH="/home/xin/OneDrive/Working_directory/Annotation-free_Cell_Counting/AfCCNN"
TINY_BBBC005_DATA_PATH='/home/xin/OneDrive/Working_directory/datasets/Cell_Counting/Tiny-BBBC005/Tiny-BBBC005_256x256.h5'
SEED=2020
BACKBONE_CNN='ResNet34'

# ####################################################################################################################################
# # LQReg
# echo "-------------------------------------------------------------------------------------------------"
# echo "exp3_rd1: LQReg only"
# python3 main_LQReg.py \
# --experiment_name exp3_rd1 \
# --root $ROOT_PATH \
# --path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
# --seed $SEED --num_workers 8 \
# 2>&1 | tee output_exp3_rd1_LQReg_seed_${SEED}.txt
#
# echo "-------------------------------------------------------------------------------------------------"
# echo "exp3_rd2: LQReg only"
# python3 main_LQReg.py \
# --experiment_name exp3_rd2 \
# --root $ROOT_PATH \
# --path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
# --seed $SEED --num_workers 8 \
# 2>&1 | tee output_exp3_rd2_LQReg_seed_${SEED}.txt
#
# echo "-------------------------------------------------------------------------------------------------"
# echo "exp3_rd3: LQReg only"
# python3 main_LQReg.py \
# --experiment_name exp3_rd3 \
# --root $ROOT_PATH \
# --path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
# --seed $SEED --num_workers 8 \
# 2>&1 | tee output_exp3_rd3_LQReg_seed_${SEED}.txt



####################################################################################################################################
# Regression CNN
echo "-------------------------------------------------------------------------------------------------"
echo "exp3_rd1: ${BACKBONE_CNN}(Reg)"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp3_rd1 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name $BACKBONE_CNN --predtype 'regre' \
--epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
--lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
2>&1 | tee output_exp3_rd1_${BACKBONE_CNN}_Reg_seed_${SEED}.txt

echo "-------------------------------------------------------------------------------------------------"
echo "exp3_rd2: ${BACKBONE_CNN}(Reg)"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp3_rd2 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name $BACKBONE_CNN --predtype 'regre' \
--epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
--lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
2>&1 | tee output_exp3_rd2_${BACKBONE_CNN}_Reg_seed_${SEED}.txt

echo "-------------------------------------------------------------------------------------------------"
echo "exp3_rd3: ${BACKBONE_CNN}(Reg)"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp3_rd3 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name $BACKBONE_CNN --predtype 'regre' \
--epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
--lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
2>&1 | tee output_exp3_rd3_${BACKBONE_CNN}_Reg_seed_${SEED}.txt




####################################################################################################################################
# Classification CNN + Ensembling with LQReg
echo "-------------------------------------------------------------------------------------------------"
echo "exp3_rd1: ${BACKBONE_CNN}(Cl)+EwLQReg"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp3_rd1 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name $BACKBONE_CNN --predtype 'class' \
--epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
--lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
--ensemble --ensemble_regre_model 'LQReg' \
2>&1 | tee output_exp3_rd1_${BACKBONE_CNN}_Cl_EwLQReg_seed_${SEED}.txt

echo "-------------------------------------------------------------------------------------------------"
echo "exp3_rd2: ${BACKBONE_CNN}(Cl)+EwLQReg"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp3_rd2 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name $BACKBONE_CNN --predtype 'class' \
--epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
--lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
--ensemble --ensemble_regre_model 'LQReg' \
2>&1 | tee output_exp3_rd2_${BACKBONE_CNN}_Cl_EwLQReg_seed_${SEED}.txt

echo "-------------------------------------------------------------------------------------------------"
echo "exp3_rd3: ${BACKBONE_CNN}(Cl)+EwLQReg"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp3_rd3 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name $BACKBONE_CNN --predtype 'class' \
--epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
--lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
--ensemble --ensemble_regre_model 'LQReg' \
2>&1 | tee output_exp3_rd3_${BACKBONE_CNN}_Cl_EwLQReg_seed_${SEED}.txt





####################################################################################################################################
# Classification CNN + DA + Ensembling with LQReg
echo "-------------------------------------------------------------------------------------------------"
echo "exp3_rd1: ${BACKBONE_CNN}(Cl)+DA+EwLQReg"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp3_rd1 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name $BACKBONE_CNN --predtype 'class' \
--epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
--lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
--dataAugment --da_flip --da_filter \
--ensemble --ensemble_regre_model 'LQReg' \
2>&1 | tee output_exp3_rd1_${BACKBONE_CNN}_Cl_DA_EwLQReg_seed_${SEED}.txt

echo "-------------------------------------------------------------------------------------------------"
echo "exp3_rd2: ${BACKBONE_CNN}(Cl)+DA+EwLQReg"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp3_rd2 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name $BACKBONE_CNN --predtype 'class' \
--epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
--lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
--dataAugment --da_flip --da_filter \
--ensemble --ensemble_regre_model 'LQReg' \
2>&1 | tee output_exp3_rd2_${BACKBONE_CNN}_Cl_DA_EwLQReg_seed_${SEED}.txt

echo "-------------------------------------------------------------------------------------------------"
echo "exp3_rd3: ${BACKBONE_CNN}(Cl)+DA+EwLQReg"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp3_rd3 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name $BACKBONE_CNN --predtype 'class' \
--epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
--lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
--dataAugment --da_flip --da_filter \
--ensemble --ensemble_regre_model 'LQReg' \
2>&1 | tee output_exp3_rd3_${BACKBONE_CNN}_Cl_DA_EwLQReg_seed_${SEED}.txt






# ####################################################################################################################################
# # Classification CNN + DA + Ensembling with ${BACKBONE_CNN}
# echo "-------------------------------------------------------------------------------------------------"
# echo "exp3_rd1: ${BACKBONE_CNN}(Cl)+DA+EwResNet34"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
# --experiment_name exp3_rd1 \
# --root $ROOT_PATH \
# --path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
# --seed $SEED --num_workers 8 \
# --cnn_name $BACKBONE_CNN --predtype 'class' \
# --epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
# --lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
# --dataAugment --da_flip --da_filter \
# --ensemble --ensemble_regre_model 'ResNet34' --ensemble_regre_cnn_path '????' \
# 2>&1 | tee output_exp3_rd1_${BACKBONE_CNN}_Cl_DA_EwResNet34_seed_${SEED}.txt
#
# echo "-------------------------------------------------------------------------------------------------"
# echo "exp3_rd2: ${BACKBONE_CNN}(Cl)+DA+EwResNet34"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
# --experiment_name exp3_rd2 \
# --root $ROOT_PATH \
# --path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
# --seed $SEED --num_workers 8 \
# --cnn_name $BACKBONE_CNN --predtype 'class' \
# --epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
# --lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
# --dataAugment --da_flip --da_filter \
# --ensemble --ensemble_regre_model 'ResNet34' --ensemble_regre_cnn_path '????' \
# 2>&1 | tee output_exp3_rd2_${BACKBONE_CNN}_Cl_DA_EwResNet34_seed_${SEED}.txt
#
# echo "-------------------------------------------------------------------------------------------------"
# echo "exp3_rd3: ${BACKBONE_CNN}(Cl)+DA+EwResNet34"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
# --experiment_name exp3_rd3 \
# --root $ROOT_PATH \
# --path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
# --seed $SEED --num_workers 8 \
# --cnn_name $BACKBONE_CNN --predtype 'class' \
# --epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
# --lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
# --dataAugment --da_flip --da_filter \
# --ensemble --ensemble_regre_model 'ResNet34' --ensemble_regre_cnn_path '????' \
# 2>&1 | tee output_exp3_rd3_${BACKBONE_CNN}_Cl_DA_EwResNet34_seed_${SEED}.txt
