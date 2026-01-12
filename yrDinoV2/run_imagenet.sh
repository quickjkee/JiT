
export OMP_NUM_THREADS=1

export PYTHONPATH=$PYTHONPATH:"/home/dbaranchuk/dpms/"
export PYTHONPATH=$PYTHONPATH:"/home/dbaranchuk/dpms/ar-classifiers"

ACCELERATE_CONFIG="configs/default_config.yaml"
PORT=$(( ((RANDOM<<15)|RANDOM) % 27001 + 2000 ))
echo $PORT

DATASET_PATH="/mnt/data/imagenet/train"


# CUDA_VISIBLE_DEVICES=0,1,2,3 
# CUDA_VISIBLE_DEVICES=0,1,2,
CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv-id=10000 --nnode=1 --nproc-per-node=2 --master-port=$PORT \
    dinov2/train/train_noisy.py \
    --data-path $DATASET_PATH \
    --config-file dinov2/configs/train/vitb14_noisy_pretrained_dinov2.yaml \
    --output-dir results/debug_diffusion_mode \
    --exp-name "debug" \
    --batch-size-per-gpu 16
    
    # train.dataset_path=ImageNet:split=TRAIN:root=$DATASET_PATH:extra=$DATASET_PATH
