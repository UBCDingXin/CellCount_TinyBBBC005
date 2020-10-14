ROOT_PATH="./AfCCNN"
TINY_BBBC005_DATA_PATH='./Tiny-BBBC005_256x256.h5'
SEED=2020
BACKBONE_CNN='ResNet34'


echo "-------------------------------------------------------------------------------------------------"
echo "exp1: ${BACKBONE_CNN}(Reg)"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp1 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name $BACKBONE_CNN --predtype 'regre' \
--epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
--lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
2>&1 | tee output_exp1_${BACKBONE_CNN}_Reg_seed_${SEED}.txt



echo "-------------------------------------------------------------------------------------------------"
echo "exp1: ${BACKBONE_CNN}(Cl)"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp1 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name $BACKBONE_CNN --predtype 'class' \
--epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
--lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
2>&1 | tee output_exp1_${BACKBONE_CNN}_Cl_seed_${SEED}.txt


echo "-------------------------------------------------------------------------------------------------"
echo "exp1: ${BACKBONE_CNN}(Cl) + EwResNet34"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp1 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name $BACKBONE_CNN --predtype 'class' \
--epochs 100 --resume_epoch 0 --batch_size_train 16 --batch_size_test 64 \
--lr_base 1e-3 --lr_decay_factor 0.1 --lr_decay_epochs '10_30_50_70' --weight_decay 1e-4 --transform \
--ensemble --ensemble_regre_model 'ResNet34' --ensemble_regre_cnn_path 'ckpt_exp1_regre_ResNet34_epochs_100_transform_True_dataAugment_False_seed_2020.pth' \
2>&1 | tee output_exp1_${BACKBONE_CNN}_Cl_EwResNet34_seed_${SEED}.txt
