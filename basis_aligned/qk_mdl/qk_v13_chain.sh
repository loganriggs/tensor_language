#!/bin/bash
# V13 chain: waits for the V10/V11 chain to finish, then trains + probes V13/V13r1.
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
until grep -q "probe done" qk_v10v11_chain.log 2>/dev/null; do sleep 60; done
echo "=== v13 chain start $(date) ==="
python qk_v13_run.py >> qk_v13_run.out 2>&1
echo "=== v13 done $(date) exit $? ==="
