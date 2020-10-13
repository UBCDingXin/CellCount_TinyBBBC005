ROOT_PATH="/home/xin/OneDrive/Working_directory/Annotation-free_Cell_Counting/FPNCNN"
VGG_DATA_PATH='/home/xin/OneDrive/Working_directory/datasets/Cell_Counting/VGGDataset/VGG_dataset.h5'
TINY_BBBC005_DATA_PATH='/home/xin/OneDrive/Working_directory/datasets/Cell_Counting/Tiny-BBBC005/Tiny-BBBC005_256x256.h5'
SEED=2020


# echo "-------------------------------------------------------------------------------------------------"
# echo "Experiment 3 SEED=$SEED Round 1"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
# --experiment_name exp3_rd1 \
# --cnn_name VGG19 \
# --root $ROOT_PATH \
# --path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
# --seed $SEED --num_workers 8 \
# --deleted_counts '61_66_70_74_78' \
# --cnn_epochs 120 --cnn_resume_epoch 0 \
# --cnn_lr_base 1e-3 --cnn_lr_decay_factor 0.1 --cnn_lr_decay_epochs '20_50_80' \
# --cnn_batch_size_train 64 --cnn_batch_size_test 64 \
# --fpn_epochs 60 --fpn_resume_epoch 0 \
# --fpn_lr_base 1e-3 --fpn_lr_decay_factor 0.1 --fpn_lr_decay_epochs '10_20_35' --fpn_weight_decay 1e-5 \
# --fpn_batch_size_train 8 \
# 2>&1 | tee output_exp3_rd1_seed_${SEED}.txt


echo "-------------------------------------------------------------------------------------------------"
echo "Experiment 3 SEED=$SEED Round 2"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp3_rd2 \
--cnn_name VGG19 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--deleted_counts '70_74_78_83_87' \
--cnn_epochs 120 --cnn_resume_epoch 0 \
--cnn_lr_base 1e-3 --cnn_lr_decay_factor 0.1 --cnn_lr_decay_epochs '20_50_80' \
--cnn_batch_size_train 64 --cnn_batch_size_test 64 \
--fpn_epochs 60 --fpn_resume_epoch 0 \
--fpn_lr_base 1e-3 --fpn_lr_decay_factor 0.1 --fpn_lr_decay_epochs '10_20_35' --fpn_weight_decay 1e-5 \
--fpn_batch_size_train 8 \
2>&1 | tee output_exp3_rd2_seed_${SEED}.txt


echo "-------------------------------------------------------------------------------------------------"
echo "Experiment 3 SEED=$SEED Round 3"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp3_rd3 \
--cnn_name VGG19 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--deleted_counts '83_87_91_96_100' \
--cnn_epochs 120 --cnn_resume_epoch 0 \
--cnn_lr_base 1e-3 --cnn_lr_decay_factor 0.1 --cnn_lr_decay_epochs '20_50_80' \
--cnn_batch_size_train 64 --cnn_batch_size_test 64 \
--fpn_epochs 60 --fpn_resume_epoch 0 \
--fpn_lr_base 1e-3 --fpn_lr_decay_factor 0.1 --fpn_lr_decay_epochs '10_20_35' --fpn_weight_decay 1e-5 \
--fpn_batch_size_train 8 \
2>&1 | tee output_exp3_rd3_seed_${SEED}.txt
