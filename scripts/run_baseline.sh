#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODEL=${1:?model required}
shift
PYTHON=/home/dbaranchuk/miniconda3/envs/qwen35/bin/python
GPUS=1
GEN_BSZ=8
NUM_IMAGES=50000
SEED=0
CKPT=
CONFIG=
REPO=
ASSETS_DIR=external
FDR_BSZ=32
FDR_MODELS=
FDR_WEIGHTS_DIR=fd_encoders
FDR_STATS_DIR=fid_stats/fd_repr
EVAL_FDR=1
SCORE_ONLY=0
DRY_RUN=0
OUTPUT_ROOT=baseline_outputs
TAG=$(date +%Y%m%d-%H%M%S)
case "$MODEL" in
  repa) CFG_LIST=4.0; STEPS=250; BAND_LIST=0.0:1.0 ;;
  pixelflow) CFG_LIST=4.0; STEPS=10; BAND_LIST=0.0:1.0 ;;
  pixnerd) CFG_LIST=3.5; STEPS=100; BAND_LIST=0.1:1.0 ;;
  rae) CFG_LIST=1.5; STEPS=50; BAND_LIST=0.0:1.0 ;;
  *) echo "Unknown model: $MODEL" >&2; exit 2 ;;
esac
for arg in "$@"; do
  key=${arg%%=*}
  [[ "$arg" == *=* ]] || { echo 'Use KEY=VALUE overrides' >&2; exit 2; }
  case "$key" in
    PYTHON|GPUS|GEN_BSZ|NUM_IMAGES|SEED|CKPT|CONFIG|REPO|ASSETS_DIR|FDR_BSZ|FDR_MODELS|FDR_WEIGHTS_DIR|FDR_STATS_DIR|EVAL_FDR|SCORE_ONLY|DRY_RUN|OUTPUT_ROOT|TAG|CFG_LIST|STEPS|BAND_LIST)
      printf -v "$key" '%s' "${arg#*=}" ;;
    *) echo "Unknown setting: $key" >&2; exit 2 ;;
  esac
done
# Bound CPU threads, including BLAS used for covariance square roots.
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
for CFG in $CFG_LIST; do
  for BAND in $BAND_LIST; do
    OUT="$OUTPUT_ROOT/$MODEL/$TAG-cfg$CFG-band$BAND-steps$STEPS-seed$SEED"
    CMD=("$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$GPUS"
      --log-dir "$OUT/workers" --tee 3
      eval_baseline.py --model "$MODEL" --assets-dir "$ASSETS_DIR" --output-dir "$OUT"
      --batch-size "$GEN_BSZ" --num-images "$NUM_IMAGES" --steps "$STEPS"
      --cfg "$CFG" --interval-min "${BAND%:*}" --interval-max "${BAND#*:}" --seed "$SEED"
      --fdr-bsz "$FDR_BSZ" --fdr-weights-dir "$FDR_WEIGHTS_DIR" --fdr-stats-dir "$FDR_STATS_DIR")
    [[ -z "$CKPT" ]] || CMD+=(--ckpt "$CKPT")
    [[ -z "$CONFIG" ]] || CMD+=(--config "$CONFIG")
    [[ -z "$REPO" ]] || CMD+=(--repo "$REPO")
    [[ "$EVAL_FDR" == 1 ]] || CMD+=(--skip-fdr)
    [[ "$SCORE_ONLY" == 0 ]] || CMD+=(--score-only)
    if [[ -n "$FDR_MODELS" ]]; then
      read -r -a SPACES <<< "$FDR_MODELS"
      CMD+=(--fdr-models "${SPACES[@]}")
    fi
    printf '%q ' "${CMD[@]}"; printf '\n'
    if [[ "$DRY_RUN" == 0 ]]; then
      mkdir -p "$OUT"
      "${CMD[@]}" 2>&1 | tee -a "$OUT/eval.log"
    fi
  done
done
