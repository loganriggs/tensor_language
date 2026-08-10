#!/bin/bash
# Chain wrapper for the realised-interface ledger (qk_iface_ledger.py; aggregate list #2).
# Waits for the GPU to be COMPLETELY free (zero compute processes AND >=10000 MiB free,
# three consecutive 60-second checks) before running.
# Registered predictions: qk_iface_ledger_predictions.json (written before the code).
set -u
cd /workspace/tensor_language/basis_aligned/qk_mdl || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
LOG=qk_iface_chain.log
# NOTE: the older chains used an unquoted 'date -u +%Y-%m-%d %H:%M:%S' whose space
# split the format string, so say() printed EMPTY timestamps. Space-free format here.
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

gate() {
  local ok=0 napps free
  say "gating on GPU compute processes (need 0 apps and >=10000 MiB free, 3 consecutive 60s checks)"
  while true; do
    napps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$napps" -eq 0 ] && [ "${free:-0}" -ge 10000 ]; then ok=$((ok + 1)); else ok=0; fi
    say "  gate: gpu_apps=$napps free=${free}MiB ok=$ok"
    [ "$ok" -ge 3 ] && break
    sleep 60
  done
  say "gate OPEN"
}

say "=== realised-interface ledger chain (pid $$) ==="
gate
say "running qk_iface_ledger.py"
if python qk_iface_ledger.py >> qk_iface_ledger.out 2>&1; then
  say "qk_iface_ledger.py finished OK"
else
  say "qk_iface_ledger.py FAILED (see qk_iface_ledger.out) — committing whatever landed"
fi
git add qk_iface_ledger.json qk_iface_ledger.out "$LOG" >/dev/null 2>&1
git commit -q -m "realised-interface ledger run complete, numbers in qk_iface_ledger.json

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UkpvpG8hGeMVyGrk8er2uc" >/dev/null 2>&1 \
  && say "committed" || say "nothing to commit / commit failed"
git push -q origin main >/dev/null 2>&1 && say "pushed" || say "PUSH FAILED (fix with fetch+rebase, never stash)"
say "=== chain done ==="
