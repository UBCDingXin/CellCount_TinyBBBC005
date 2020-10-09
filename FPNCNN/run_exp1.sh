ROOT_PATH="/media/qiong/icecream/SSC_case_study/CellCount_TinyBBBC005/FPNCNN"
TINY_BBBC005_DATA_PATH='/media/qiong/icecream/SSC_case_study/CellCount_TinyBBBC005/dataset/Tiny-BBBC005_256x256.h5'
SEED=2020


python main.py \
--experiment_name exp1 \
--cnn_name VGG19 \
--root $ROOT_PATH \
--path_tinybbbc005 $TINY_BBBC005_DATA_PATH \
--seed $SEED --num_workers 1 \
--cnn_epochs 120 --cnn_resume_epoch 0 \
--cnn_lr_base 1e-5 --cnn_lr_decay_factor 0.1 --cnn_lr_decay_epochs '20_50_80' \
--cnn_batch_size_train 16 --cnn_batch_size_test 64 \
--fpn_epochs 50 --fpn_resume_epoch 0 \
--fpn_lr_base 1e-3 --fpn_lr_decay_factor 0.1 --fpn_lr_decay_epochs '20_30_40' --fpn_weight_decay 1e-5 \
--fpn_batch_size_train 2 \
2>&1 | tee output_exp1_seed_${SEED}.txt
