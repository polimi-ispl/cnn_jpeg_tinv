#!/bin/bash

# Function to handle the SIGINT signal (CTRL+C)
function handle_sigint {
    echo "Stopping all background jobs"
    kill $(jobs -p)
    exit
}

# Set the trap
trap handle_sigint SIGINT

# TRAIN the various AAAABondiNets on JPEG compressed data with various factors
# The various scripts are run in parallel on a single GPU, but executed sequentially for each block size

DATA_ROOT='/nas/home/ecannas/jpeg_expl/code_release/data'  # PUT YOUR DATA ROOT HERE!
MODEL_DIR='../models'  # PUT THE DIRECTORY TO SAVING YOUR MODELS HERE!
LOGS_DIR='../logs'  # PUT THE DIRECTORY TO SAVING YOUR LOGS HERE!
GPU_ID=2  # PUT YOUR GPU ID HERE!

# --- POOL ONLY --- #

## --- JPEG block size = 7x7 --- #
#echo "Doing JPEG block size = 7x7 trainings"
## 1. Train a AABondiNet with stride 1
#nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 1 --jpeg_bs 7 --gpu $GPU_ID --aa_pool_only &
## 2. Train a AABondiNet with stride 2
#nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 2 --jpeg_bs 7 --gpu $GPU_ID --aa_pool_only &
## 3. Train a AABondiNet with stride 1
#nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 3 --jpeg_bs 7 --gpu $GPU_ID --aa_pool_only &
## 4. Train a AABondiNet with stride 1
#nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 4 --jpeg_bs 7 --gpu $GPU_ID --aa_pool_only &
#
## Wait for the jobs to finish (the GPUs can't handle more than one training at a time)
#wait
#
## --- JPEG block size = 8x8 --- #
#echo "Doing JPEG block size = 8x8 trainings"
## 1. Train a AABondiNet with stride 1
#nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 1 --jpeg_bs 8 --gpu $GPU_ID --aa_pool_only &
## 2. Train a AABondiNet with stride 2
#nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 2 --jpeg_bs 8 --gpu $GPU_ID --aa_pool_only &
## 3. Train a AABondiNet with stride 1
#nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 3 --jpeg_bs 8 --gpu $GPU_ID --aa_pool_only &
## 4. Train a AABondiNet with stride 1
#nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 4 --jpeg_bs 8 --gpu $GPU_ID --aa_pool_only &
#
## Wait for the jobs to finish (the GPUs can't handle more than one training at a time)
#wait
#
## --- JPEG block size = 9x9 --- #
#echo "Doing JPEG block size = 9x9 trainings"
## 1. Train a AABondiNet with stride 1
#nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 1 --jpeg_bs 9 --gpu $GPU_ID --aa_pool_only &
## 2. Train a AABondiNet with stride 2
#nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 2 --jpeg_bs 9 --gpu $GPU_ID --aa_pool_only &
## 3. Train a AABondiNet with stride 1
#nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 3 --jpeg_bs 9 --gpu $GPU_ID --aa_pool_only &
## 4. Train a AABondiNet with stride 1
#nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 4 --jpeg_bs 9 --gpu $GPU_ID --aa_pool_only &
#
## Wait for the jobs to finish (the GPUs can't handle more than one training at a time)
#wait
#
## --- JPEG block size = 12x12 --- #
#echo "Doing JPEG block size = 12x12 trainings"
## 1. Train a AABondiNet with stride 1
#nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 1 --jpeg_bs 12 --gpu $GPU_ID --aa_pool_only &
## 2. Train a AABondiNet with stride 2
#nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 2 --jpeg_bs 12 --gpu $GPU_ID --aa_pool_only &
## 3. Train a AABondiNet with stride 1
#nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 3 --jpeg_bs 12 --gpu $GPU_ID --aa_pool_only &
## 4. Train a AABondiNet with stride 1
#nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 4 --jpeg_bs 12 --gpu $GPU_ID --aa_pool_only &
#
## Wait for the jobs to finish (the GPUs can't handle more than one training at a time)
#wait

# --- FULL ANTIALIASING --- #

# --- JPEG block size = 7x7 --- #
echo "Doing JPEG block size = 7x7 trainings fully antialias"
# 1. Train a AABondiNet with stride 1
nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-4 --batch_size 32 --init ../models/net-BondiNet_lr-0.001_batch_size-32_split_train_test-0.75_split_train_val-0.75_split_seed-42_in_channels-1_init_period-10_jpeg_bs-7_fl_stride-1_random_crop-False/bestval.pth --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 1 --jpeg_bs 7 --gpu $GPU_ID &
# 2. Train a AABondiNet with stride 2
nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-4 --batch_size 32 --init ../models/net-BondiNet_lr-0.001_batch_size-32_split_train_test-0.75_split_train_val-0.75_split_seed-42_in_channels-1_init_period-10_jpeg_bs-7_fl_stride-2_random_crop-False/bestval.pth --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 2 --jpeg_bs 7 --gpu $GPU_ID &
# 3. Train a AABondiNet with stride 3
nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-4 --batch_size 32 --init ../models/net-BondiNet_lr-0.001_batch_size-32_split_train_test-0.75_split_train_val-0.75_split_seed-42_in_channels-1_init_period-10_jpeg_bs-7_fl_stride-3_random_crop-False/bestval.pth --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 3 --jpeg_bs 7 --gpu $GPU_ID &
# 4. Train a AABondiNet with stride 4
nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-4 --batch_size 32 --init ../models/net-BondiNet_lr-0.001_batch_size-32_split_train_test-0.75_split_train_val-0.75_split_seed-42_in_channels-1_init_period-10_jpeg_bs-7_fl_stride-4_random_crop-False/bestval.pth --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 4 --jpeg_bs 7 --gpu $GPU_ID &

# Wait for the jobs to finish (the GPUs can't handle more than one training at a time)
wait

# --- JPEG block size = 8x8 --- #
echo "Doing JPEG block size = 8x8 trainings fully antialias"
# 1. Train a AABondiNet with stride 1
nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-4 --batch_size 32 --init ../models/net-BondiNet_lr-0.001_batch_size-32_split_train_test-0.75_split_train_val-0.75_split_seed-42_in_channels-1_init_period-10_jpeg_bs-8_fl_stride-1_random_crop-False/bestval.pth --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 1 --jpeg_bs 8 --gpu $GPU_ID &
# 2. Train a AABondiNet with stride 2
nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-4 --batch_size 32 --init ../models/net-BondiNet_lr-0.001_batch_size-32_split_train_test-0.75_split_train_val-0.75_split_seed-42_in_channels-1_init_period-10_jpeg_bs-8_fl_stride-2_random_crop-False/bestval.pth --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 2 --jpeg_bs 8 --gpu $GPU_ID &
# 3. Train a AABondiNet with stride 3
nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-4 --batch_size 32 --init ../models/net-BondiNet_lr-0.001_batch_size-32_split_train_test-0.75_split_train_val-0.75_split_seed-42_in_channels-1_init_period-10_jpeg_bs-8_fl_stride-3_random_crop-False/bestval.pth --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 3 --jpeg_bs 8 --gpu $GPU_ID &
# 4. Train a AABondiNet with stride 4
nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-4 --batch_size 32 --init ../models/net-BondiNet_lr-0.001_batch_size-32_split_train_test-0.75_split_train_val-0.75_split_seed-42_in_channels-1_init_period-10_jpeg_bs-8_fl_stride-4_random_crop-False/bestval.pth --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 4 --jpeg_bs 8 --gpu $GPU_ID &

# Wait for the jobs to finish (the GPUs can't handle more than one training at a time)
wait

# --- JPEG block size = 9x9 --- #
echo "Doing JPEG block size = 9x9 trainings fully antialias"
# 1. Train a AABondiNet with stride 1
nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-4 --batch_size 32 --init ../models/net-BondiNet_lr-0.001_batch_size-32_split_train_test-0.75_split_train_val-0.75_split_seed-42_in_channels-1_init_period-10_jpeg_bs-9_fl_stride-1_random_crop-False/bestval.pth --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 1 --jpeg_bs 9 --gpu $GPU_ID &
# 2. Train a AABondiNet with stride 2
nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-4 --batch_size 32 --init ../models/net-BondiNet_lr-0.001_batch_size-32_split_train_test-0.75_split_train_val-0.75_split_seed-42_in_channels-1_init_period-10_jpeg_bs-9_fl_stride-2_random_crop-False/bestval.pth --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 2 --jpeg_bs 9 --gpu $GPU_ID &
# 3. Train a AABondiNet with stride 3
nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-4 --batch_size 32 --init ../models/net-BondiNet_lr-0.001_batch_size-32_split_train_test-0.75_split_train_val-0.75_split_seed-42_in_channels-1_init_period-10_jpeg_bs-9_fl_stride-3_random_crop-False/bestval.pth --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 3 --jpeg_bs 9 --gpu $GPU_ID &
# 4. Train a AABondiNet with stride 4
nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-4 --batch_size 32 --init ../models/net-BondiNet_lr-0.001_batch_size-32_split_train_test-0.75_split_train_val-0.75_split_seed-42_in_channels-1_init_period-10_jpeg_bs-9_fl_stride-4_random_crop-False/bestval.pth --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 4 --jpeg_bs 9 --gpu $GPU_ID &

# Wait for the jobs to finish (the GPUs can't handle more than one training at a time)
wait

# --- JPEG block size = 12x12 --- #
echo "Doing JPEG block size = 12x12 trainings fully antialias"
# 1. Train a AABondiNet with stride 1
nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-4 --batch_size 32 --init ../models/net-BondiNet_lr-0.001_batch_size-32_split_train_test-0.75_split_train_val-0.75_split_seed-42_in_channels-1_init_period-10_jpeg_bs-12_fl_stride-1_random_crop-False/bestval.pth --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 1 --jpeg_bs 12 --gpu $GPU_ID &
# 2. Train a AABondiNet with stride 2
nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-4 --batch_size 32 --init ../models/net-BondiNet_lr-0.001_batch_size-32_split_train_test-0.75_split_train_val-0.75_split_seed-42_in_channels-1_init_period-10_jpeg_bs-12_fl_stride-2_random_crop-False/bestval.pth --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 2 --jpeg_bs 12 --gpu $GPU_ID &
# 3. Train a AABondiNet with stride 3
nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-4 --batch_size 32 --init ../models/net-BondiNet_lr-0.001_batch_size-32_split_train_test-0.75_split_train_val-0.75_split_seed-42_in_channels-1_init_period-10_jpeg_bs-12_fl_stride-3_random_crop-False/bestval.pth --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 3 --jpeg_bs 12 --gpu $GPU_ID &
# 4. Train a AABondiNet with stride 4
nice python ../train.py --workers 8 --model AABondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-4 --batch_size 32 --init ../models/net-BondiNet_lr-0.001_batch_size-32_split_train_test-0.75_split_train_val-0.75_split_seed-42_in_channels-1_init_period-10_jpeg_bs-12_fl_stride-4_random_crop-False/bestval.pth --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 4 --jpeg_bs 12 --gpu $GPU_ID &

# Wait for the jobs to finish (the GPUs can't handle more than one training at a time)
wait