#!/bin/bash
# gate: wait for the commons192 training process to exit, then start the
# effective-param-matched w1344 recipe on GPU 0
cd /workspace/tensor_language/basis_aligned/qk_mdl
while pgrep -f 'qk_s_muon_run.py commons3e[5]' > /dev/null; do sleep 60; done
sleep 30
source /venv/main/bin/activate
CUDA_VISIBLE_DEVICES=0 nohup python qk_s_w1344_run.py > qk_s_w1344.out 2>&1
