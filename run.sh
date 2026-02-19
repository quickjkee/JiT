export PYTHONPATH=$PYTHONPATH:"/home/dbaranchuk/dpms/"

export OMP_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_port=29501 \
    main_jit.py \
    --model JiT-B/16 \
    --img_size 256 --noise_scale 1.0 \
    --gen_bsz 4 --num_images 50000 --cfg 3.0 --interval_min 0.0 --interval_max 1.0 \
    --output_dir tut \
    --data_path None \
    --in_context_len 24 \
    --reg_len 8 \
    --in_context_start 4