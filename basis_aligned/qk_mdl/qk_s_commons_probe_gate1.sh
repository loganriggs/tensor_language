#!/bin/bash
# gate: wait for the share chain (S4 last) to exit, then run commons probes on GPU 1
cd /workspace/tensor_language/basis_aligned/qk_mdl
while pgrep -f 'qk_s_share_run.p[y]' > /dev/null; do sleep 60; done
sleep 30
source /venv/main/bin/activate
CUDA_VISIBLE_DEVICES=1 nohup python qk_s_commons_probe.py > qk_s_commons_probe.out 2>&1
