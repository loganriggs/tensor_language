#!/bin/bash
# Small-model V10/V11 set: train (control, V10, V11, V11nl, V11lr) then probe.
# Runs alongside the width-768 window job; each python script has its own
# shared-GPU guard (never demands more than 5200 MiB free, 60 s polls).
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
echo "=== chain start $(date) ==="
python qk_v10v11_train.py >> qk_v10v11_train.out 2>&1
echo "=== train done $(date) exit $? ==="
python qk_v10v11_probe.py >> qk_v10v11_probe.out 2>&1
echo "=== probe done $(date) exit $? ==="
