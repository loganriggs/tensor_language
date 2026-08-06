#!/bin/bash
# Overnight chain, GPU 1. Waits for shrink3e5 (~06:30 UTC), then the
# param-matched shared-source values arm (~4h).
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
while pgrep -f 'qk_s_muon_run.py shrink3e[5]$' > /dev/null; do sleep 60; done
CUDA_VISIBLE_DEVICES=1 python qk_s_muon_run.py combo3e5svpb \
    > qk_s_combo3e5svpb.out 2>&1
echo "night gpu1 chain done"
