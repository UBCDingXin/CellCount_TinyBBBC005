ROOT_PATH="./ERDCNN"
VGG_DATA_PATH='./VGG_dataset.h5'
TINY_BBBC005_DATA_PATH='./Tiny-BBBC005_256x256.h5'
NTRAIN=1200


SEED=2020
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp1_reduced \
--root $ROOT_PATH \
--path_vgg_dataset $VGG_DATA_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--unet1_epochs 200 --unet1_resume_epoch 0 \
--unet1_lr_base 1e-3 --unet1_lr_decay_factor 0.01 --unet1_lr_decay_epochs '100_150' \
--unet1_batch_size_train 16 --unet1_transform \
--unet2_epochs 200 --unet2_resume_epoch 0 \
--unet2_lr_base 1e-3 --unet2_lr_decay_factor 0.1 --unet2_lr_decay_epochs '50_100_150' --unet2_weight_decay 1e-5 \
--unet2_batch_size_train 32 --unet2_transform \
--cnn_epochs 120 --cnn_resume_epoch 0 \
--cnn_lr_base 1e-2 --cnn_lr_decay_factor 0.1  --cnn_lr_decay_epochs '20_50_80' \
--cnn_batch_size_train 64 --cnn_batch_size_test 64 --cnn_transform \
--num_train $NTRAIN \
2>&1 | tee output_exp1_reduced_seed_${SEED}.txt


SEED=2021
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp1_reduced \
--root $ROOT_PATH \
--path_vgg_dataset $VGG_DATA_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--unet1_epochs 200 --unet1_resume_epoch 0 \
--unet1_lr_base 1e-3 --unet1_lr_decay_factor 0.01 --unet1_lr_decay_epochs '100_150' \
--unet1_batch_size_train 16 --unet1_transform \
--unet2_epochs 200 --unet2_resume_epoch 0 \
--unet2_lr_base 1e-3 --unet2_lr_decay_factor 0.1 --unet2_lr_decay_epochs '50_100_150' --unet2_weight_decay 1e-5 \
--unet2_batch_size_train 32 --unet2_transform \
--cnn_epochs 120 --cnn_resume_epoch 0 \
--cnn_lr_base 1e-2 --cnn_lr_decay_factor 0.1  --cnn_lr_decay_epochs '20_50_80' \
--cnn_batch_size_train 64 --cnn_batch_size_test 64 --cnn_transform \
--num_train $NTRAIN \
2>&1 | tee output_exp1_reduced_seed_${SEED}.txt


SEED=2022
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp1_reduced \
--root $ROOT_PATH \
--path_vgg_dataset $VGG_DATA_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--unet1_epochs 200 --unet1_resume_epoch 0 \
--unet1_lr_base 1e-3 --unet1_lr_decay_factor 0.01 --unet1_lr_decay_epochs '100_150' \
--unet1_batch_size_train 16 --unet1_transform \
--unet2_epochs 200 --unet2_resume_epoch 0 \
--unet2_lr_base 1e-3 --unet2_lr_decay_factor 0.1 --unet2_lr_decay_epochs '50_100_150' --unet2_weight_decay 1e-5 \
--unet2_batch_size_train 32 --unet2_transform \
--cnn_epochs 120 --cnn_resume_epoch 0 \
--cnn_lr_base 1e-2 --cnn_lr_decay_factor 0.1  --cnn_lr_decay_epochs '20_50_80' \
--cnn_batch_size_train 64 --cnn_batch_size_test 64 --cnn_transform \
--num_train $NTRAIN \
2>&1 | tee output_exp1_reduced_seed_${SEED}.txt
