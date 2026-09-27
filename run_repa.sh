#!/usr/bin/env bash
# ImageNet evaluation with official repa weights and FD_r^6.
bash "$(dirname "${BASH_SOURCE[0]}")/scripts/run_baseline.sh" repa "$@"
