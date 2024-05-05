# TRAIN the various ResNets on standard 8x8 JPEG compressed data
# The various scripts run sequentially on a single GPU due to memory requirements

# --- Execution params
DATA_ROOT='/nas/home/ecannas/jpeg_expl/code_release/data'  # PUT YOUR DATA ROOT HERE!
MODEL_DIR='../models'  # PUT THE DIRECTORY TO SAVING YOUR MODELS HERE!
LOGS_DIR='../logs'  # PUT THE DIRECTORY TO SAVING YOUR LOGS HERE!
GPU_ID=5  # PUT YOUR GPU ID HERE!

# --- Runs
# 1. Train a standard ResNet50
nice python ../train.py --workers 8 --model ResNet50 --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --init_period 10 --epochs 200 --lr 1e-3 --batch_size 2 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --gpu $GPU_ID
# 2. Train a full AAResNet50 (we need a slightly lower learning rate for it to converge)
nice python ../train.py --workers 8 --model AAResNet50 --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --init_period 10 --epochs 200 --lr 1e-4 --batch_size 2 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --gpu $GPU_ID
# 3. Train a pool only AAResNet50
nice python ../train.py --workers 8 --model AAResNet50 --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --init_period 10 --epochs 200 --lr 1e-3 --batch_size 2 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --aa_pool_only --gpu $GPU_ID
# 4. Train a standard ResNet50 with random cropping as data augmentation
nice python ../train.py --workers 8 --model ResNet50 --data_root $DATA_ROOT --scratch --grayscale --es_patience 35 --init_period 10 --epochs 200 --lr 1e-3 --batch_size 2 --scratch --models_dir $MODEL_DIR --log_dir $LOGS_DIR --random_crop --gpu $GPU_ID

