ROOT_PATH="/home/xin/OneDrive/Working_directory/Annotation-free_Cell_Counting/AfCCNN"
TINY_BBBC005_DATA_PATH='/home/xin/OneDrive/Working_directory/datasets/Cell_Counting/Tiny-BBBC005/Tiny-BBBC005_256x256.h5'
SEED=2020


# echo "-------------------------------------------------------------------------------------------------"
# echo "exp1: LQReg only"
# python3 main_LQReg.py \
# --experiment_name exp1 \
# --root $ROOT_PATH \
# --path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
# --seed $SEED --num_workers 8 \
# 2>&1 | tee output_exp1_LQReg_seed_${SEED}.txt


echo "-------------------------------------------------------------------------------------------------"
echo "exp1: ResNet34(Reg)"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp1 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--cnn_name 'ResNet34' --predtype 'regre' \
--epochs 120 --resume_epoch 0 --batch_size_train 32 --batch_size_test 64 \
--lr_base 1e-2 --lr_decay_factor 0.1 --lr_decay_epochs '20_50_80' --weight_decay 1e-5 --transform \
2>&1 | tee output_exp1_ResNet34_Reg_seed_${SEED}.txt



# echo "-------------------------------------------------------------------------------------------------"
# echo "exp1: ResNet34"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
# --experiment_name exp1 \
# --root $ROOT_PATH \
# --path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
# --seed $SEED --num_workers 8 \
# --cnn_name 'ResNet34' --predtype 'class' \
# --epochs 120 --resume_epoch 0 --batch_size_train 32 --batch_size_test 64 \
# --lr_base 1e-2 --lr_decay_factor 0.1 --lr_decay_epochs '20_50_80' --weight_decay 1e-5 --transform \
# 2>&1 | tee output_exp1_seed_${SEED}.txt
