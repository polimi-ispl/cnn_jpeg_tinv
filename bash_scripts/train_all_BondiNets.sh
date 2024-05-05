#!/bin/bash

# Function to handle the SIGINT signal (CTRL+C)
function handle_sigint {
    echo "Stopping all background jobs"
    kill $(jobs -p)
    exit
}

# Set the trap
trap handle_sigint SIGINT

# TRAIN the various BondiNets on JPEG compressed data with various factors
# The various scripts are run in parallel on a single GPU, but executed sequentially for each block size

# --- Parameters --- #

DATA_ROOT='/nas/home/ecannas/jpeg_expl/code_release/data'  # PUT YOUR DATA ROOT HERE!
MODEL_DIR='../models'  # PUT THE DIRECTORY TO SAVING YOUR MODELS HERE!
LOGS_DIR='../logs'  # PUT THE DIRECTORY TO SAVING YOUR LOGS HERE!
GPU_ID=2  # PUT YOUR GPU ID HERE!

# --- No random crop --- #

# --- JPEG block size = 7x7 --- #
# 1. Train a BondiNet with stride 1
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 1 --jpeg_bs 7 --gpu $GPU_ID &
# 2. Train a BondiNet with stride 2
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 2 --jpeg_bs 7 --gpu $GPU_ID &
# 3. Train a BondiNet with stride 1
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 3 --jpeg_bs 7 --gpu $GPU_ID &
# 4. Train a BondiNet with stride 1
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 4 --jpeg_bs 7 --gpu $GPU_ID &

# Wait for the jobs to finish (the GPUs can't handle more than one training at a time)
wait

# --- JPEG block size = 8x8 --- #
# 1. Train a BondiNet with stride 1
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 1 --jpeg_bs 8 --gpu $GPU_ID &
# 2. Train a BondiNet with stride 2
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 2 --jpeg_bs 8 --gpu $GPU_ID &
# 3. Train a BondiNet with stride 1
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 3 --jpeg_bs 8 --gpu $GPU_ID &
# 4. Train a BondiNet with stride 1
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 4 --jpeg_bs 8 --gpu $GPU_ID &

# Wait for the jobs to finish (the GPUs can't handle more than one training at a time)
wait

# --- JPEG block size = 9x9 --- #
# 1. Train a BondiNet with stride 1
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 1 --jpeg_bs 9 --gpu $GPU_ID &
# 2. Train a BondiNet with stride 2
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 2 --jpeg_bs 9 --gpu $GPU_ID &
# 3. Train a BondiNet with stride 1
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 3 --jpeg_bs 9 --gpu $GPU_ID &
# 4. Train a BondiNet with stride 1
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 4 --jpeg_bs 9 --gpu $GPU_ID &

# Wait for the jobs to finish (the GPUs can't handle more than one training at a time)
wait

# --- JPEG block size = 12x12 --- #
# 1. Train a BondiNet with stride 1
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 1 --jpeg_bs 12 --gpu $GPU_ID &
# 2. Train a BondiNet with stride 2
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 2 --jpeg_bs 12 --gpu $GPU_ID &
# 3. Train a BondiNet with stride 1
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 3 --jpeg_bs 12 --gpu $GPU_ID &
# 4. Train a BondiNet with stride 1
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 4 --jpeg_bs 12 --gpu $GPU_ID &

# Wait for the jobs to finish (the GPUs can't handle more than one training at a time)
wait

# --- Random crop data augmentation --- #

# --- JPEG block size = 7x7 --- #
echo "Training BondiNets with JPEG block size 7x7 and random crop data augmentation..."
# 1. Train a BondiNet with stride 1
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 1 --jpeg_bs 7 --gpu $GPU_ID --random_crop &
# 2. Train a BondiNet with stride 2
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 2 --jpeg_bs 7 --gpu $GPU_ID --random_crop &
# 3. Train a BondiNet with stride 1
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 3 --jpeg_bs 7 --gpu $GPU_ID --random_crop &
# 4. Train a BondiNet with stride 1
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 4 --jpeg_bs 7 --gpu $GPU_ID --random_crop &

# Wait for the jobs to finish (the GPUs can't handle more than one training at a time)
wait

# --- JPEG block size = 8x8 --- #
echo "Training BondiNets with JPEG block size 8x8 and random crop data augmentation..."
# 1. Train a BondiNet with stride 1
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 1 --jpeg_bs 8 --gpu $GPU_ID --random_crop &
# 2. Train a BondiNet with stride 2
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 2 --jpeg_bs 8 --gpu $GPU_ID --random_crop &
# 3. Train a BondiNet with stride 1
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 3 --jpeg_bs 8 --gpu $GPU_ID --random_crop &
# 4. Train a BondiNet with stride 1
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 4 --jpeg_bs 8 --gpu $GPU_ID --random_crop &

# Wait for the jobs to finish (the GPUs can't handle more than one training at a time)
wait

# --- JPEG block size = 9x9 --- #
echo "Training BondiNets with JPEG block size 9x9 and random crop data augmentation..."
# 1. Train a BondiNet with stride 1
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 1 --jpeg_bs 9 --gpu $GPU_ID --random_crop &
# 2. Train a BondiNet with stride 2
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 2 --jpeg_bs 9 --gpu $GPU_ID --random_crop &
# 3. Train a BondiNet with stride 1
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 3 --jpeg_bs 9 --gpu $GPU_ID --random_crop &
# 4. Train a BondiNet with stride 1
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 4 --jpeg_bs 9 --gpu $GPU_ID --random_crop &

# Wait for the jobs to finish (the GPUs can't handle more than one training at a time)
wait

# --- JPEG block size = 12x12 --- #
echo "Training BondiNets with JPEG block size 12x12 and random crop data augmentation..."
# 1. Train a BondiNet with stride 1
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 1 --jpeg_bs 12 --gpu $GPU_ID --random_crop &
# 2. Train a BondiNet with stride 2
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 2 --jpeg_bs 12 --gpu $GPU_ID --random_crop &
# 3. Train a BondiNet with stride 1
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 3 --jpeg_bs 12 --gpu $GPU_ID --random_crop &
# 4. Train a BondiNet with stride 1
nice python ../train.py --workers 8 --model BondiNet --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --sched_patience 20 --epochs 200 --lr 1e-3 --batch_size 32 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --first_layer_stride 4 --jpeg_bs 12 --gpu $GPU_ID --random_crop &

# Wait for the jobs to finish (the GPUs can't handle more than one training at a time)
wait