#!/bin/bash
# V14 chain: waits for the V13 chain to finish, then trains + probes V14a/V14b.
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
until grep -q "v13 done" qk_v13_chain.log 2>/dev/null; do sleep 60; done
echo "=== v14 chain start $(date) ==="
python qk_v14_run.py >> qk_v14_run.out 2>&1
echo "=== v14 done $(date) exit $? ==="
