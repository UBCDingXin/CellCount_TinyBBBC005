ROOT_PATH="/home/xin/OneDrive/Working_directory/Annotation-free_Cell_Counting/FPNCNN"
VGG_DATA_PATH='/home/xin/OneDrive/Working_directory/datasets/Cell_Counting/VGGDataset/VGG_dataset.h5'
TINY_BBBC005_DATA_PATH='/home/xin/OneDrive/Working_directory/datasets/Cell_Counting/Tiny-BBBC005/Tiny-BBBC005_256x256.h5'
NTRAIN=1200


SEED=2020
echo "-------------------------------------------------------------------------------------------------"
echo "Experiment 1 SEED=$SEED"
CUDA_VISIBLE_DEVICES=0 python3 main.py \
--experiment_name exp1_reduced \
--cnn_name VGG19 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 4 \
--cnn_epochs 120 --cnn_resume_epoch 0 \
--cnn_lr_base 1e-3 --cnn_lr_decay_factor 0.1 --cnn_lr_decay_epochs '20_50_80' \
--cnn_batch_size_train 64 --cnn_batch_size_test 64 \
--fpn_epochs 60 --fpn_resume_epoch 0 \
--fpn_lr_base 1e-3 --fpn_lr_decay_factor 0.1 --fpn_lr_decay_epochs '10_20_35' --fpn_weight_decay 1e-5 \
--fpn_batch_size_train 8 \
--num_train $NTRAIN \
2>&1 | tee output_exp1_reduced_seed_${SEED}.txt


SEED=2021
echo "-------------------------------------------------------------------------------------------------"
echo "Experiment 1 SEED=$SEED"
CUDA_VISIBLE_DEVICES=0 python3 main.py \
--experiment_name exp1_reduced \
--cnn_name VGG19 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 4 \
--cnn_epochs 120 --cnn_resume_epoch 0 \
--cnn_lr_base 1e-3 --cnn_lr_decay_factor 0.1 --cnn_lr_decay_epochs '20_50_80' \
--cnn_batch_size_train 64 --cnn_batch_size_test 64 \
--fpn_epochs 60 --fpn_resume_epoch 0 \
--fpn_lr_base 1e-3 --fpn_lr_decay_factor 0.1 --fpn_lr_decay_epochs '10_20_35' --fpn_weight_decay 1e-5 \
--fpn_batch_size_train 8 \
--num_train $NTRAIN \
2>&1 | tee output_exp1_reduced_seed_${SEED}.txt


SEED=2022
echo "-------------------------------------------------------------------------------------------------"
echo "Experiment 1 SEED=$SEED"
CUDA_VISIBLE_DEVICES=0 python3 main.py \
--experiment_name exp1_reduced \
--cnn_name VGG19 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 4 \
--cnn_epochs 120 --cnn_resume_epoch 0 \
--cnn_lr_base 1e-3 --cnn_lr_decay_factor 0.1 --cnn_lr_decay_epochs '20_50_80' \
--cnn_batch_size_train 64 --cnn_batch_size_test 64 \
--fpn_epochs 60 --fpn_resume_epoch 0 \
--fpn_lr_base 1e-3 --fpn_lr_decay_factor 0.1 --fpn_lr_decay_epochs '10_20_35' --fpn_weight_decay 1e-5 \
--fpn_batch_size_train 8 \
--num_train $NTRAIN \
2>&1 | tee output_exp1_reduced_seed_${SEED}.txt
