# Official ImageNet baselines for FD_r^6

## Sources and checkpoints

| Model | Resolution | Official source | Released weights |
|---|---:|---|---|
| SiT-XL/2 + REPA | 512 | [sihyun-yu/REPA](https://github.com/sihyun-yu/REPA) | [sit_xl_2_repa_512.pt](https://huggingface.co/kyungmnlee/DMF/blob/main/sit_xl_2_repa_512.pt) |
| PixelFlow | 256 | [ShoufaChen/PixelFlow](https://github.com/ShoufaChen/PixelFlow) | [PixelFlow-Class2Image](https://huggingface.co/ShoufaChen/PixelFlow-Class2Image) |
| PixNerd-XL/16 | 512 | [MCG-NJU/PixNerd](https://github.com/MCG-NJU/PixNerd) | [res512_ft200k_epoch=325-step=1800000_emainit.ckpt](https://huggingface.co/MCG-NJU/PixNerd-XL-P16-C2I/blob/main/res512_ft200k_epoch%3D325-step%3D1800000_emainit.ckpt) |
| RAE DiT-DH-XL + autoguidance | 512 | [bytetriper/RAE](https://github.com/bytetriper/RAE) | [RAE-collections](https://huggingface.co/nyu-visionx/RAE-collections/tree/main) |

**Skipped: plain SiT-512.** The [original SiT release](https://github.com/willisma/SiT)
provides only a 256 checkpoint; `sample.py` explicitly rejects automatic 512 checkpoint download.
No original-author 512 checkpoint was found. A 256 checkpoint is not substituted.

REPA does have a 512 training setup in its README. Its default download is 256, but the
[author's reply in issue #10](https://github.com/sihyun-yu/REPA/issues/10) points to the
`kyungmnlee/DMF` collection, which also contains the 512 REPA checkpoint used here.
The `dmf_*` weights in that collection are a different method and are not used.

RAE requires four files from its collection: `DiTDH-XL_ep400/stage2_model.pt`,
`DiTDH-S_ep20/stage2_model.pt` under `DiTs/Dinov2/wReg_base/ImageNet512/`,
`decoders/dinov2/wReg_base/ViTXL_n08_i512/model.pt`, and
`stats/dinov2/wReg_base/imagenet1k_512/stat.pt`. Its DINOv2 encoder/config is also prepared.
REPA uses `stabilityai/sd-vae-ft-ema` for decoding.

## Prepare

```bash
cd /home/dbaranchuk/dpms/jit
PY=/home/dbaranchuk/miniconda3/envs/qwen35/bin/python
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 "$PY" prepare_baselines.py --install-deps
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 "$PY" prepare_fd_stats.py
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 "$PY" prepare_fd_encoders.py
```

Preparation pins source revisions and downloads only inference artifacts under `external/`.
Extra packages (`diffusers==0.32.2`, `torchdiffeq==0.2.4`) are isolated in
`external/python_deps`; shared qwen35 packages are unchanged. The environment also needs
PyTorch, torchvision, timm, transformers, scipy, numpy, Pillow, einops, OmegaConf and
huggingface_hub (already present in the tested qwen35 environment).
Downloaded sources, weights, encoders, stats and generated images are git-ignored.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 bash run_repa.sh
CUDA_VISIBLE_DEVICES=0 bash run_pixelflow.sh
CUDA_VISIBLE_DEVICES=0 bash run_pixnerd.sh
CUDA_VISIBLE_DEVICES=0 bash run_rae.sh
```

Each launcher defaults to 50,000 samples, one GPU, batch size 8, and all six FDr spaces.
Choose available devices; GPU 3 was occupied during local validation.
For multiple GPUs or a guidance sweep:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash run_repa.sh GPUS=2 GEN_BSZ=8 CFG_LIST="3.0 4.0"
CUDA_VISIBLE_DEVICES=0,1 bash run_pixnerd.sh GPUS=2 CFG_LIST="3.0 3.5" BAND_LIST="0.1:1.0 0.0:1.0"
```

| Launcher | Default sampler | Guidance |
|---|---|---|
| `run_repa.sh` | Official Euler-Maruyama SDE, 250 steps | CFG 4.0, full interval |
| `run_pixelflow.sh` | Official Dopri5 cascade, atol 1e-6, rtol 1e-3, shift 1 | Stage-wise CFG up to 2.4, full interval only |
| `run_pixnerd.sh` | Official Euler ODE, 100 steps | CFG 3.5, interval (0.1, 1.0) |
| `run_rae.sh` | Official shifted Euler ODE, 50 time points | Autoguidance 1.5, full interval |

These are starting recipes, not a claim of optimal FD_r^6. `STEPS` follows each official
sampler's convention; it is not a common NFE budget. `CFG_LIST` controls RAE autoguidance
strength. REPA intervals use its official noise-to-data time convention; PixNerd uses
its own increasing time convention. PixelFlow rejects interval overrides. Its
`PIXELFLOW_SOLVER=dopri5` default uses `STEPS=30` as requested output time points per
stage, not the number of adaptive model evaluations. `PIXELFLOW_SOLVER=euler` uses
`STEPS` model evaluations per stage. PixelFlow output directories include the solver name.

Overrides: `PYTHON`, `GPUS`, `GEN_BSZ`, `NUM_IMAGES`, `SEED`, `CKPT`, `CONFIG`, `REPO`,
`CFG_LIST`, `STEPS`, `PIXELFLOW_SOLVER` (PixelFlow only), `BAND_LIST`, `EVAL_FDR`, `FDR_MODELS`, `FDR_BSZ`,
`FDR_STATS_DIR`, `FDR_WEIGHTS_DIR`, `OUTPUT_ROOT`, `TAG`, `SCORE_ONLY`, `DRY_RUN`.
`CKPT` is a file except for PixelFlow, where it is a directory containing `config.yaml`
and `model.pt`. RAE `CONFIG` replaces the full official sampling config; its relative
asset paths are interpreted under its source repository.

For a quick execution check:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_pixnerd.sh NUM_IMAGES=2 GEN_BSZ=1 STEPS=4 EVAL_FDR=0 TAG=smoke
bash run_repa.sh DRY_RUN=1 GPUS=2
```

Samples, `run.json`, `eval.log`, and `metrics.json` are retained under
`baseline_outputs/<model>/<tag>-cfg...`. Existing sample directories are refused, preventing
mixed runs. Use a new `TAG` to repeat generation. To score existing samples, repeat the
same settings and `TAG` with `SCORE_ONLY=1 EVAL_FDR=1`. Large sweeps require space for all
retained images; use `OUTPUT_ROOT` on an appropriate disk.

## Metric protocol

All methods feed the existing JiT `util/fd_repr.py` implementation. PNG conversion clips
to [0,1] and rounds to uint8, as JiT does. This differs from upstream REPA/RAE PNG truncation.
Classes cycle over 0–999, giving exactly 50 images per class for 50k samples. Small smoke
runs cover only the first few classes. Samples remain at native resolution; each FD encoder
resizes them to its reference input size. The released 256 reference statistics are also
used for 512 output, following FD-loss's current `eval_all_fds.py` policy.
The Inception term is recomputed against the ADM reference; no JiT FID is substituted.

All six spaces are required for FD_r^6. If `FDR_MODELS` selects a subset, the printed
`FDr^N` is an N-space average, not FD_r^6 (the existing metric JSON retains its `fdr6` key).
Tiny-sample smoke scores are not quality estimates: the covariance is singular and class
coverage is incomplete. Full 50k evaluations are separate runs.

## Local validation (2026-09-27)

Environment: `/home/dbaranchuk/miniconda3/envs/qwen35/bin/python`, PyTorch
`2.11.0+cu130`, timm `1.0.16`, NVIDIA A100 80 GB. CPU thread pools were limited to 8.

- Strict pretrained checkpoint loading and native-resolution finite output passed for
  REPA-512, PixelFlow-256, PixNerd-512 and RAE-512.
- Full default sampler schedules passed with two samples for REPA, PixNerd and RAE;
  PixelFlow passed with four samples on two GPUs using the initial Euler demo recipe.
- Independent review checked label ordering, CFG/autoguidance, EMA selection, latent
  scaling, time shift and decoder normalization against the official implementations.
- RAE's `patch_size: "SHOULD BE RELOADED"` config placeholder is resolved to the exact
  runtime value (16) in temporary JSON before loading. This handles Transformers 5 type
  validation without changing upstream files or the decoder geometry.
- Python compilation, shell syntax, all launcher dry runs, and pinned artifact preparation
  passed. No full 50,000-image benchmark has been run.

Smoke artifacts and logs are in `baseline_outputs/`. The six-space scoring checks use
only two samples and must not be interpreted as reported model performance.

Final checks: all six FDr spaces completed with finite results for all four methods.
Two-GPU generation passed for PixelFlow (four samples, initial Euler schedule) and PixNerd
(16 samples, batch 8 per GPU, four steps). RAE also passed batch 8 on one A100 80 GB.
REPA's metric calculation completed; its first shell wrapper was affected by a concurrent
launcher edit after scoring, so the final wrapper was rerun and verified to exit cleanly.

## Nirvana checkpoint archives

Created under `/home/dbaranchuk/nirvana_data/`:

| Archive | Extracted directory | Size (decimal GB) |
|---|---|---:|
| `sit_repa_512.tar` | `sit_repa_512/` | 3.087 |
| `pixelflow_256.tar` | `pixelflow_256/` | 2.722 |
| `pixnerd_512.tar` | `pixnerd_512/` | 2.845 |
| `rae_512.tar` | `rae_512/` | 6.182 |

Each archive includes the pinned official source, checkpoint, required auxiliary weights,
and isolated `diffusers`/`torchdiffeq` dependencies. Model files retain their original bytes.
Each archive has a `.tar.sha256` sidecar and an internal file manifest; all archived files
were checked against their SHA-256 hashes. Rebuild with `python package_baselines.py`;
existing archives are never overwritten.

Extract the desired archive into `$INPUT_PATH`, preserving its top-level directory.
Run from the updated JiT directory, with the intended Python environment activated.
`ASSETS_DIR` relocates the official source, dependencies, and auxiliary checkpoint paths;
`CKPT` selects the primary checkpoint. The common FD encoders/statistics are separate
inputs, using the names supplied below. `PYTHON=python3` uses the activated job environment.
`GPUS=8` assumes an eight-GPU job; reduce it to match the allocation.

### SiT + REPA 512

```bash
. run_repa.sh PYTHON=python3 GPUS=8 ASSETS_DIR="$INPUT_PATH/sit_repa_512" CKPT="$INPUT_PATH/sit_repa_512/weights/repa/sit_xl_2_repa_512.pt" BAND_LIST="0.0:1.0" CFG_LIST="4.0" EVAL_FDR=1 FDR_WEIGHTS_DIR="$INPUT_PATH/fd_encoders_v2" FDR_STATS_DIR="$INPUT_PATH/fd_encoders_stats"
```

### PixelFlow 256

```bash
. run_pixelflow.sh PYTHON=python3 GPUS=8 ASSETS_DIR="$INPUT_PATH/pixelflow_256" CKPT="$INPUT_PATH/pixelflow_256/weights/pixelflow" BAND_LIST="0.0:1.0" CFG_LIST="2.4" PIXELFLOW_SOLVER=dopri5 STEPS=30 EVAL_FDR=1 FDR_WEIGHTS_DIR="$INPUT_PATH/fd_encoders_v2" FDR_STATS_DIR="$INPUT_PATH/fd_encoders_stats"
```

### PixNerd 512

```bash
. run_pixnerd.sh PYTHON=python3 GPUS=8 ASSETS_DIR="$INPUT_PATH/pixnerd_512" CKPT="$INPUT_PATH/pixnerd_512/weights/pixnerd/res512_ft200k_epoch=325-step=1800000_emainit.ckpt" BAND_LIST="0.1:1.0" CFG_LIST="3.5" EVAL_FDR=1 FDR_WEIGHTS_DIR="$INPUT_PATH/fd_encoders_v2" FDR_STATS_DIR="$INPUT_PATH/fd_encoders_stats"
```

### RAE 512

```bash
. run_rae.sh PYTHON=python3 GPUS=8 ASSETS_DIR="$INPUT_PATH/rae_512" CKPT="$INPUT_PATH/rae_512/RAE/models/DiTs/Dinov2/wReg_base/ImageNet512/DiTDH-XL_ep400/stage2_model.pt" BAND_LIST="0.0:1.0" CFG_LIST="1.5" EVAL_FDR=1 FDR_WEIGHTS_DIR="$INPUT_PATH/fd_encoders_v2" FDR_STATS_DIR="$INPUT_PATH/fd_encoders_stats"
```

These launchers select their own model and sampler. JiT-specific `FORWARD_TYPE`,
`REG_LIST`, and `MODEL` arguments do not apply. Defaults remain 50,000 images and
batch size 8 per GPU. RAE's `CFG_LIST` controls autoguidance strength.

Archive validation: all four archives were extracted into fresh temporary directories and
passed two-image CUDA generation through the documented `ASSETS_DIR`/`CKPT` interface,
with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` in qwen35. This confirms that
checkpoint loading and generation use the relocated bundles. Logs:
`baseline_outputs/packaged_check_gelnj9zt/`. Temporary extracted copies were removed.

## Native worker crash diagnostics

The evaluator uses Gloo for its two CPU barriers. Model generation stays on each worker's
GPU; the evaluator does not exchange model tensors between GPUs. This avoids initializing
an unused NCCL communicator during checkpoint loading.

Launchers retain each worker's stdout/stderr under `<run>/workers/`, enable Python fault
tracebacks, and print runtime versions. RAE reports the loading stages before GPU transfer,
decoder verification, and stage-2/autoguidance loading.

A reported H100 job terminated with SIGSEGV on ranks 6 and 7 after stage-1 normalization
loaded. Its supplied stack showed UCX signal handlers without the faulting application
frame. This does not identify the root cause. Switching the barriers is a mitigation;
confirmation on that remote runtime requires a rerun. Existing model archives remain valid.

Local validation of this mitigation: two A100 GPUs in qwen35 completed the same four-image
RAE smoke case before and after the backend change. All four PNGs were byte-identical;
both workers recorded separate stdout/stderr logs. The H100/UCX crash was not reproduced
locally.

## PixelFlow paper sampling recipe (2026-09-28 correction)

The bundled model is XL/4: 28 blocks, width 1152, 16 attention heads, patch size 4,
and 676,611,120 checkpoint tensor elements. The checkpoint was correct, but the initial
launcher used the Gradio demo's Euler-10-per-stage / CFG-max-4.0 settings. Those do not
match the paper's reported 1.98 FID recipe.

[Table 3](https://arxiv.org/html/2504.07963v1#S4.T3) reports 1.98 with Dopri5,
absolute tolerance 1e-6 and stage-wise guidance with maximum 2.4. The official pipeline
sets relative tolerance 1e-3 and guidance to [1, 1.233333, 1.933333, 2.4] over four stages.
The corrected launcher selects this official Dopri5 path and matches the official sampler's
BF16 autocast and TF32 settings. The tar and weights do not need replacement.
The upstream `sample_ddp.py` requires `--use-ode-dopri5` to select that path; its README's
bare sampling command otherwise selects Euler.

Validation in qwen35 on A100: the corrected launcher completed two samples. With the
same seed and class label, the adapter's output exactly matched a separate official
pipeline call using Dopri5, CFG 2.4, BF16 and TF32. A negative control using the old
Euler-10 / CFG-4.0 recipe produced different output. Independent review checked stage
guidance, label ordering, solver tolerances and the output-time-point convention.

A paper FID comparison also requires its ADM TensorFlow evaluator on 50,000 generated
samples. JiT's FD_r pipeline currently uses its existing PyTorch Inception implementation
and the ADM reference statistics. Numerical equivalence of those evaluators has not been
established. The raw `FD[inception]` is distinct from its normalized `FDr[inception]`
and the six-space `FDr^6` average. No reproduction of FID 1.98 is claimed from smoke tests.
