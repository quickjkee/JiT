# export OMP_NUM_THREADS=1
export PYTHONPATH=$PYTHONPATH:"/home/dbaranchuk/dpms/"

PORT=$(( ((RANDOM<<15)|RANDOM) % 27001 + 2000 ))
echo $PORT

MODEL_NAME="JiT-B/16"
SNAPSHOT_PATH="results"


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --master_port=${PORT} --nproc_per_node=4  main_jit.py \
    --model $MODEL_NAME \
    --proj_dropout 0.0 \
    --P_mean -0.8 --P_std 0.8 \
    --img_size 256 --noise_scale 1.0 \
    --batch_size 128 --blr 5e-5 \
    --epochs 600 --warmup_epochs 5 \
    --gen_bsz 128 --num_images 5000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
    --output_dir $SNAPSHOT_PATH --resume $SNAPSHOT_PATH \
    --online_eval \
    --in_context_len 32 \
    --in_context_start 4