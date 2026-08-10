#!/bin/bash
# Chain wrapper for the planted-modular DGP experiment (qk_dgp_modular.py;
# aggregate list #4).  Registered predictions: qk_dgp_modular_predictions.json
# (committed before the build).  Gates until the GPU is COMPLETELY free (zero
# compute processes AND >=10000 MiB free) AND no tiny-models tf_* chain is
# running, for 3 consecutive 60-second checks.  Bracket-trick pgrep patterns:
# substring pgrep self-matches killed us twice (AGENT_BRIEF).
set -u
cd /workspace/tensor_language/basis_aligned/qk_mdl || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
LOG=qk_dgp_chain.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

gate() {
  local ok=0 napps free tfchains
  say "gating: need 0 GPU apps AND >=10000 MiB free AND no tf_* chain, 3 consecutive 60s checks"
  while true; do
    napps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    tfchains=$(pgrep -fc 'tf_v2_transition_chain[.]sh|tf_v4096_seeds35_chain[.]sh' || true)
    if [ "$napps" -eq 0 ] && [ "${free:-0}" -ge 10000 ] && [ "${tfchains:-0}" -eq 0 ]; then
      ok=$((ok + 1))
    else
      ok=0
    fi
    say "  gate: gpu_apps=$napps free=${free}MiB tf_chains=${tfchains:-0} ok=$ok"
    [ "$ok" -ge 3 ] && break
    sleep 60
  done
  say "gate OPEN"
}

say "=== planted-modular DGP chain (pid $$) ==="
gate
say "running qk_dgp_modular.py"
if python qk_dgp_modular.py >> qk_dgp_modular.out 2>&1; then
  say "qk_dgp_modular.py finished OK"
else
  say "qk_dgp_modular.py FAILED (see qk_dgp_modular.out) — committing whatever landed"
fi
git add qk_dgp_modular.json qk_dgp_modular.out "$LOG" >/dev/null 2>&1
git commit -q -m "planted-modular DGP run complete, numbers in qk_dgp_modular.json

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UkpvpG8hGeMVyGrk8er2uc" >/dev/null 2>&1 \
  && say "committed" || say "nothing to commit / commit failed"
git push -q origin main >/dev/null 2>&1 && say "pushed" || say "PUSH FAILED (fix with fetch+rebase, never stash)"
say "=== chain done ==="
