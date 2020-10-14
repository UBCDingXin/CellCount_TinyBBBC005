ROOT_PATH="./FPNCNN"
VGG_DATA_PATH='./VGG_dataset.h5'
TINY_BBBC005_DATA_PATH='./Tiny-BBBC005_256x256.h5'
SEED=2020


echo "-------------------------------------------------------------------------------------------------"
echo "Experiment 1 SEED=$SEED"
CUDA_VISIBLE_DEVICES=0 python3 main.py \
--experiment_name exp1 \
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
2>&1 | tee output_exp1_seed_${SEED}.txt
