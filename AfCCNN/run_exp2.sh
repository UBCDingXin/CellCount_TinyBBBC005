ROOT_PATH="/home/xin/OneDrive/Working_directory/Annotation-free_Cell_Counting/AfCCNN"
TINY_BBBC005_DATA_PATH='/home/xin/OneDrive/Working_directory/datasets/Cell_Counting/Tiny-BBBC005/Tiny-BBBC005_256x256.h5'
SEED=2020

# ####################################################################################################################################
# # LQReg
# echo "-------------------------------------------------------------------------------------------------"
# echo "exp2_rd1: LQReg only"
# python3 main_LQReg.py \
# --experiment_name exp2_rd1 \
# --root $ROOT_PATH \
# --path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
# --seed $SEED --num_workers 8 \
# 2>&1 | tee output_exp2_rd1_LQReg_seed_${SEED}.txt
#
# echo "-------------------------------------------------------------------------------------------------"
# echo "exp2_rd2: LQReg only"
# python3 main_LQReg.py \
# --experiment_name exp2_rd2 \
# --root $ROOT_PATH \
# --path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
# --seed $SEED --num_workers 8 \
# 2>&1 | tee output_exp2_rd2_LQReg_seed_${SEED}.txt
#
# echo "-------------------------------------------------------------------------------------------------"
# echo "exp2_rd3: LQReg only"
# python3 main_LQReg.py \
# --experiment_name exp2_rd3 \
# --root $ROOT_PATH \
# --path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
# --seed $SEED --num_workers 8 \
# 2>&1 | tee output_exp2_rd3_LQReg_seed_${SEED}.txt



####################################################################################################################################
# Regression CNN
echo "-------------------------------------------------------------------------------------------------"
echo "exp2_rd1: ResNet34(Reg)"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp2_rd1 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name 'ResNet34' --predtype 'regre' \
--epochs 120 --resume_epoch 0 --batch_size_train 64 --batch_size_test 64 \
--lr_base 1e-2 --lr_decay_factor 0.1 --lr_decay_epochs '20_50_80' --weight_decay 1e-5 --transform \
2>&1 | tee output_exp2_rd1_ResNet_Reg_seed_${SEED}.txt

echo "-------------------------------------------------------------------------------------------------"
echo "exp2_rd2: ResNet34(Reg)"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp2_rd2 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name 'ResNet34' --predtype 'regre' \
--epochs 120 --resume_epoch 0 --batch_size_train 64 --batch_size_test 64 \
--lr_base 1e-2 --lr_decay_factor 0.1 --lr_decay_epochs '20_50_80' --weight_decay 1e-5 --transform \
2>&1 | tee output_exp2_rd2_ResNet_Reg_seed_${SEED}.txt

echo "-------------------------------------------------------------------------------------------------"
echo "exp2_rd3: ResNet34(Reg)"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp2_rd3 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name 'ResNet34' --predtype 'regre' \
--epochs 120 --resume_epoch 0 --batch_size_train 64 --batch_size_test 64 \
--lr_base 1e-2 --lr_decay_factor 0.1 --lr_decay_epochs '20_50_80' --weight_decay 1e-5 --transform \
2>&1 | tee output_exp2_rd3_ResNet_Reg_seed_${SEED}.txt


####################################################################################################################################
# Classification CNN + Ensembling with LQReg
echo "-------------------------------------------------------------------------------------------------"
echo "exp2_rd1: ResNet34(Cl)+EwLQReg"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp2_rd1 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name 'ResNet34' --predtype 'class' \
--epochs 120 --resume_epoch 0 --batch_size_train 64 --batch_size_test 64 \
--lr_base 1e-2 --lr_decay_factor 0.1 --lr_decay_epochs '20_50_80' --weight_decay 1e-5 --transform \
--ensemble --ensemble_regre_model 'LQReg' \
2>&1 | tee output_exp2_rd1_ResNet34_Cl_EwLQReg_seed_${SEED}.txt

echo "-------------------------------------------------------------------------------------------------"
echo "exp2_rd2: ResNet34(Cl)+EwLQReg"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp2_rd2 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name 'ResNet34' --predtype 'class' \
--epochs 120 --resume_epoch 0 --batch_size_train 64 --batch_size_test 64 \
--lr_base 1e-2 --lr_decay_factor 0.1 --lr_decay_epochs '20_50_80' --weight_decay 1e-5 --transform \
--ensemble --ensemble_regre_model 'LQReg' \
2>&1 | tee output_exp2_rd2_ResNet34_Cl_EwLQReg_seed_${SEED}.txt

echo "-------------------------------------------------------------------------------------------------"
echo "exp2_rd3: ResNet34(Cl)+EwLQReg"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp2_rd3 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name 'ResNet34' --predtype 'class' \
--epochs 120 --resume_epoch 0 --batch_size_train 64 --batch_size_test 64 \
--lr_base 1e-2 --lr_decay_factor 0.1 --lr_decay_epochs '20_50_80' --weight_decay 1e-5 --transform \
--ensemble --ensemble_regre_model 'LQReg' \
2>&1 | tee output_exp2_rd3_ResNet34_Cl_EwLQReg_seed_${SEED}.txt




####################################################################################################################################
# Classification CNN + DA + Ensembling with LQReg
echo "-------------------------------------------------------------------------------------------------"
echo "exp2_rd1: ResNet34(Cl)+DA+EwLQReg"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp2_rd1 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name 'ResNet34' --predtype 'class' \
--epochs 120 --resume_epoch 0 --batch_size_train 64 --batch_size_test 64 \
--lr_base 1e-2 --lr_decay_factor 0.1 --lr_decay_epochs '20_50_80' --weight_decay 1e-5 --transform \
--dataAugment --da_flip --da_filter \
--ensemble --ensemble_regre_model 'LQReg' \
2>&1 | tee output_exp2_rd1_ResNet34_Cl_DA_EwLQReg_seed_${SEED}.txt

echo "-------------------------------------------------------------------------------------------------"
echo "exp2_rd2: ResNet34(Cl)+DA+EwLQReg"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp2_rd2 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name 'ResNet34' --predtype 'class' \
--epochs 120 --resume_epoch 0 --batch_size_train 64 --batch_size_test 64 \
--lr_base 1e-2 --lr_decay_factor 0.1 --lr_decay_epochs '20_50_80' --weight_decay 1e-5 --transform \
--dataAugment --da_flip --da_filter \
--ensemble --ensemble_regre_model 'LQReg' \
2>&1 | tee output_exp2_rd2_ResNet34_Cl_DA_EwLQReg_seed_${SEED}.txt

echo "-------------------------------------------------------------------------------------------------"
echo "exp2_rd3: ResNet34(Cl)+DA+EwLQReg"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp2_rd3 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name 'ResNet34' --predtype 'class' \
--epochs 120 --resume_epoch 0 --batch_size_train 64 --batch_size_test 64 \
--lr_base 1e-2 --lr_decay_factor 0.1 --lr_decay_epochs '20_50_80' --weight_decay 1e-5 --transform \
--dataAugment --da_flip --da_filter \
--ensemble --ensemble_regre_model 'LQReg' \
2>&1 | tee output_exp2_rd3_ResNet34_Cl_DA_EwLQReg_seed_${SEED}.txt
