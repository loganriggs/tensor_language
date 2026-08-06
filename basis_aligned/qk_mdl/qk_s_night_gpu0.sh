#!/bin/bash
# Overnight chain, GPU 0 (Logan 2026-08-06: larger param-matched versions).
# Waits for combo3e5sv, then: funnel smoke -> funnelsv (~4h) -> funnel
# control (~4h) -> light probes for the finished sv arms.
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
while pgrep -f 'qk_s_muon_run.py combo3e5s[v]$' > /dev/null; do sleep 60; done
CUDA_VISIBLE_DEVICES=0 python qk_s_funnel1152_run.py \
    > qk_s_funnel_smoke.out 2>&1 || { echo "SMOKE FAILED"; exit 1; }
CUDA_VISIBLE_DEVICES=0 python qk_s_muon_run.py funnelsv \
    > qk_s_funnelsv.out 2>&1
echo "funnelsv arm done"
CUDA_VISIBLE_DEVICES=0 python qk_s_muon_run.py funnel \
    > qk_s_funnelplain.out 2>&1
echo "funnel control arm done"
CUDA_VISIBLE_DEVICES=0 python qk_s_probe_run.py > qk_s_night_probes.out 2>&1
echo "night gpu0 chain done"
