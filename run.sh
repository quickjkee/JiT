#!/bin/bash
# SIMPLE combined-guidance sweep:  v = C + cfg*(A-C) + reg*(A-B)
# Just runs each option and prints FID. No CSV, no logs, no resume, no saved outputs.

# ---------------- fixed config ----------------
GPUS=8
MODEL="JiT-B/16"
IMG=256
GEN_BSZ=256
NUM_IMAGES=50000
CLASS_MIN=0.1                       # class-term interval
CLASS_MAX=1.0
CKPT=/home/quickjkee/projects/CUR/registers/checkpoints/jit/tmp/new/checkpoint-420.pth
SCRATCH=here      # reused + wiped each run; nothing persisted
PORT=29570
# ----------------------------------------------

# REG_LIST="1.5 2.0 2.5 2.8 3 3.2 3.5 3.8 4.0"     
# ---------------- sweep grids -----------------
CFG_LIST="2.0"                       # class weights w_c
REG_LIST="3.5"                     # register weights w_r (reg=0 baseline added per cfg)
BAND_LIST="0.03:0.9"  # interval_min_reg:interval_max_reg
# ----------------------------------------------

# ---------------- CLI overrides ---------------
# Override any setting above as KEY=VALUE, e.g.:
#   ./sweep_reg_simple.sh CKPT=/path/checkpoint-520.pth
#   ./sweep_reg_simple.sh CKPT=... NUM_IMAGES=10000 CFG_LIST="2.5 3.0" GPUS=4
for arg in "$@"; do
  case "$arg" in
    *=*) eval "${arg%%=*}=\"\${arg#*=}\"" ;;
    *)   echo "ignoring arg (use KEY=VALUE): $arg" ;;
  esac
done
echo "CKPT=$CKPT | NUM_IMAGES=$NUM_IMAGES | GPUS=$GPUS"
echo "CFG_LIST=[$CFG_LIST] REG_LIST=[$REG_LIST] BAND_LIST=[$BAND_LIST]"
# ----------------------------------------------

run_one () {   # args: CFG REG RMIN RMAX
  local CFG=$1 REG=$2 RMIN=$3 RMAX=$4
  rm -rf "$SCRATCH"; mkdir -p "$SCRATCH"; PORT=$((PORT+1))
  local FID
  FID=$(torchrun --nproc_per_node=$GPUS --nnodes=1 --node_rank=0 --master_port=$PORT main_jit.py \
        --model "$MODEL" --img_size $IMG --noise_scale 1.0 \
        --gen_bsz $GEN_BSZ --num_images $NUM_IMAGES \
        --cfg $CFG --rg $REG \
        --interval_min $CLASS_MIN --interval_max $CLASS_MAX \
        --interval_min_rg $RMIN --interval_max_rg $RMAX \
        --output_dir "$SCRATCH" --resume "$CKPT" \
        --data_path None --evaluate_gen 2>&1 | grep -aoP 'FID:\s*\K[0-9.]+' | tail -1)
  printf 'cfg=%-4s reg=%-4s band=%s-%s  FID=%s\n' "$CFG" "$REG" "$RMIN" "$RMAX" "${FID:-NA}"
}

for CFG in $CFG_LIST; do
  run_one "$CFG" 0 "$CLASS_MIN" "$CLASS_MAX"          # per-cfg baseline (reg=0)
  for REG in $REG_LIST; do
    for BAND in $BAND_LIST; do
      run_one "$CFG" "$REG" "${BAND%:*}" "${BAND#*:}"
    done
  done
done

rm -rf "$SCRATCH"
