ROOT_PATH="./CellCount_TinyBBBC005/AfCCNN"
TINY_BBBC005_DATA_PATH='./CellCount_TinyBBBC005/Tiny-BBBC005_256x256.h5'
SEED=2020
BACKBONE_CNN='ResNet34'



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
# ours: Classification CNN + DA + Ensembling with ResNet34
echo "-------------------------------------------------------------------------------------------------"
echo "exp3_rd1: ${BACKBONE_CNN}(Cl)+DA+EwResNet34"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp3_rd1 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name $BACKBONE_CNN --predtype 'class' \
--epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
--lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
--dataAugment --da_flip --da_filter \
--ensemble --ensemble_regre_model 'ResNet34' --ensemble_regre_cnn_path 'ckpt_exp3_rd1_regre_ResNet34_epochs_100_transform_True_dataAugment_False_seed_2020.pth' \
2>&1 | tee output_exp3_rd1_${BACKBONE_CNN}_Cl_DA_EwResNet34_seed_${SEED}.txt

echo "-------------------------------------------------------------------------------------------------"
echo "exp3_rd2: ${BACKBONE_CNN}(Cl)+DA+EwResNet34"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp3_rd2 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name $BACKBONE_CNN --predtype 'class' \
--epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
--lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
--dataAugment --da_flip --da_filter \
--ensemble --ensemble_regre_model 'ResNet34' --ensemble_regre_cnn_path 'ckpt_exp3_rd2_regre_ResNet34_epochs_100_transform_True_dataAugment_False_seed_2020.pth' \
2>&1 | tee output_exp3_rd2_${BACKBONE_CNN}_Cl_DA_EwResNet34_seed_${SEED}.txt

echo "-------------------------------------------------------------------------------------------------"
echo "exp3_rd3: ${BACKBONE_CNN}(Cl)+DA+EwResNet34"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp3_rd3 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name $BACKBONE_CNN --predtype 'class' \
--epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
--lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
--dataAugment --da_flip --da_filter \
--ensemble --ensemble_regre_model 'ResNet34' --ensemble_regre_cnn_path 'ckpt_exp3_rd3_regre_ResNet34_epochs_100_transform_True_dataAugment_False_seed_2020.pth' \
2>&1 | tee output_exp3_rd3_${BACKBONE_CNN}_Cl_DA_EwResNet34_seed_${SEED}.txt
