#!/bin/bash
# E5 chain: wait for the Muon rerun to finish, then smoke-test the E5
# cost-decomposition runner on CPU (the smoke harness's import chain polls the
# GPU guard, so smoke must wait for a free GPU too), then run it for real.
cd /workspace/tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate
until ! pgrep -f "python qk_e0m_muon_run" > /dev/null; do sleep 60; done
echo "=== muon done, smoke-testing e5 $(date) ==="
QK_SMOKE=1 timeout 900 python qk_e5_costdecomp_run.py > qk_e5_smoke.out 2>&1
if grep -q "e5 costdecomp run done" qk_e5_smoke.out; then
    echo "=== smoke passed, e5 real run start $(date) ==="
    python qk_e5_costdecomp_run.py >> qk_e5_run.out 2>&1
    echo "=== e5 done $(date) exit $? ==="
else
    echo "=== SMOKE FAILED -- not launching real run; see qk_e5_smoke.out ==="
fi
