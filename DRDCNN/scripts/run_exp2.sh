ROOT_PATH="./CellCount_TinyBBBC005/DRDCNN"
VGG_DATA_PATH='./CellCount_TinyBBBC005/VGGDataset/VGG_dataset.h5'
TINY_BBBC005_DATA_PATH='./CellCount_TinyBBBC005/Tiny-BBBC005_256x256.h5'
SEED=2020


echo "-------------------------------------------------------------------------------------------------"
echo "Experiment 2 SEED=$SEED Round 1"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp2_rd1 \
--root $ROOT_PATH \
--path_vgg_dataset $VGG_DATA_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--deleted_counts '14_35_57_66_83' \
--unet_epochs 200 --unet_resume_epoch 0 \
--unet_lr_base 1e-3 --unet_lr_decay_factor 0.01 --unet_lr_decay_epochs '100_150' \
--unet_batch_size_train 16 --unet_transform \
--cnn_epochs 120 --cnn_resume_epoch 0 \
--cnn_lr_base 1e-2 --cnn_lr_decay_factor 0.1  --cnn_lr_decay_epochs '20_50_80' \
--cnn_batch_size_train 64 --cnn_batch_size_test 64 --cnn_transform \


echo "-------------------------------------------------------------------------------------------------"
echo "Experiment 2 SEED=$SEED Round 2"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp2_rd2 \
--root $ROOT_PATH \
--path_vgg_dataset $VGG_DATA_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--deleted_counts '10_31_70_83_91' \
--unet_epochs 200 --unet_resume_epoch 0 \
--unet_lr_base 1e-3 --unet_lr_decay_factor 0.01 --unet_lr_decay_epochs '100_150' \
--unet_batch_size_train 16 --unet_transform \
--cnn_epochs 120 --cnn_resume_epoch 0 \
--cnn_lr_base 1e-2 --cnn_lr_decay_factor 0.1  --cnn_lr_decay_epochs '20_50_80' \
--cnn_batch_size_train 64 --cnn_batch_size_test 64 --cnn_transform \


echo "-------------------------------------------------------------------------------------------------"
echo "Experiment 2 SEED=$SEED Round 3"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--experiment_name exp2_rd3 \
--root $ROOT_PATH \
--path_vgg_dataset $VGG_DATA_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 8 \
--deleted_counts '18_27_44_53_91' \
--unet_epochs 200 --unet_resume_epoch 0 \
--unet_lr_base 1e-3 --unet_lr_decay_factor 0.01 --unet_lr_decay_epochs '100_150' \
--unet_batch_size_train 16 --unet_transform \
--cnn_epochs 120 --cnn_resume_epoch 0 \
--cnn_lr_base 1e-2 --cnn_lr_decay_factor 0.1  --cnn_lr_decay_epochs '20_50_80' \
--cnn_batch_size_train 64 --cnn_batch_size_test 64 --cnn_transform \
