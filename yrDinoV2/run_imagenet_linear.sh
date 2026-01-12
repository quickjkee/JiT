
export OMP_NUM_THREADS=1

export PYTHONPATH=$PYTHONPATH:"/home/dbaranchuk/dpms/"
export PYTHONPATH=$PYTHONPATH:"/home/dbaranchuk/dpms/ar-classifiers"

ACCELERATE_CONFIG="configs/default_config.yaml"
PORT=$(( ((RANDOM<<15)|RANDOM) % 27001 + 2000 ))
echo $PORT

DATASET="fmow"
DATASET_PATH="/mnt/data/imagenet/train"


CUDA_VISIBLE_DEVICES=0,1,6,7 torchrun --rdzv-id=10000 --nnode=1 --nproc-per-node=4 --master-port=$PORT \
    dinov2/eval/linear.py \
    --config-file /home/dbaranchuk/dpms/ar-classifiers/yrDinoV2/dinov2/configs/train/vitl16_short.yaml \
    --pretrained-weights results/vit_l16_imagenet_ckpt_final/teacher_checkpoint.pth \
    --output-dir results/vit_l16_imagenet_ckpt_final/debug \
    --train-dataset "/mnt/data/imagenet/train" \
    --val-dataset "/mnt/data/imagenet/val" \
    --batch-size 256
