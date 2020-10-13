#!/bin/bash

#PBS -l walltime=8:00:00,select=1:ncpus=4:ompthreads=4:ngpus=1:gpu_mem=16gb:mem=64gb
#PBS -N FP_1_1
#PBS -A st-dingxin9-1-gpu
#PBS -m abe
#PBS -M xin.ding@stat.ubc.ca
#PBS -o output.txt
#PBS -e error.txt

################################################################################

module unuse /arc/software/spack/share/spack/lmod/linux-centos7-x86_64/Core
module use /arc/software/spack-0.14.0-110/share/spack/lmod/linux-centos7-x86_64/Core


module load gcc
module load cuda
module load openmpi/3.1.5
module load openblas/0.3.9
module load py-torch/1.4.1-py3.7.6
module load py-torchvision/0.5.0-py3.7.6
module load py-pyparsing/2.4.2-py3.7.6
module load py-tqdm/4.36.1-py3.7.6
module load py-pillow/7.0.0-py3.7.6
module load py-cycler/0.10.0-py3.7.6
module load freetype/2.10.1
module load libpng/1.6.37
module load py-setuptools/41.4.0-py3.7.6
module load py-python-dateutil/2.8.0-py3.7.6
module load py-kiwisolver/1.1.0-py3.7.6
module load py-matplotlib
module load py-h5py
module load py-scipy


cd $PBS_O_WORKDIR


ROOT_PATH="/scratch/st-dingxin9-1/Annotation-free_Cell_Counting/FPNCNN"
TINY_BBBC005_DATA_PATH='/scratch/st-dingxin9-1/Annotation-free_Cell_Counting/Tiny-BBBC005_256x256.h5'
VGG_DATA_PATH='/scratch/st-dingxin9-1/Annotation-free_Cell_Counting/VGG_dataset.h5'
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
--cnn_lr_base 1e-3 --cnn_lr_decay_factor 0.1 --cnn_lr_decay_epochs '20_50_80' --cnn_transform \
--cnn_batch_size_train 64 --cnn_batch_size_test 64 \
--fpn_epochs 50 --fpn_resume_epoch 0 \
--fpn_lr_base 1e-3 --fpn_lr_decay_factor 0.1 --fpn_lr_decay_epochs '20_30_40' --fpn_weight_decay 1e-5 \
--fpn_batch_size_train 2 --fpn_transform \
2>&1 | tee output_exp1_seed_${SEED}.txt
