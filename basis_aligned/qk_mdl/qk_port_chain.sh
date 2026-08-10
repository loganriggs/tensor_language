#!/bin/bash
# Chain wrapper for the static-fraction-by-depth port profile (qk_port_profile.py).
# Modelled on tiny_full_interp/tf_cells35.sh. Gates until the GPU is genuinely
# free — zero compute processes AND >= 10000 MiB free AND no qk_scalar_chain.sh
# running — for 3 consecutive minutes, then runs the full profile, then commits
# and pushes the result JSON + this log.
set -u
cd /workspace/tensor_language/basis_aligned/qk_mdl || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
LOG=qk_port_chain.log
say() { echo "[$(date -u +%Y-%m-%d %H:%M:%S)] $*" | tee -a "$LOG"; }

gate() {
  local ok=0 napps free scal
  say "gating on GPU (0 compute apps, >=10000 MiB free, no qk_scalar_chain.sh) x3 min"
  while true; do
    napps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    scal=$(pgrep -fc "qk_scalar_chain.sh" || true)
    if [ "${napps:-0}" -eq 0 ] && [ "${free:-0}" -ge 10000 ] && [ "${scal:-0}" -eq 0 ]; then
      ok=$((ok + 1))
    else
      ok=0
    fi
    say "  gate: gpu_apps=${napps:-0} free=${free:-?}MiB scalar_chain=${scal:-0} ok=$ok"
    [ "$ok" -ge 3 ] && break
    sleep 60
  done
  say "gate OPEN"
}

push() {
  git add -- qk_port_profile.json qk_port_chain.log qk_port_profile.py >/dev/null 2>&1
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UkpvpG8hGeMVyGrk8er2uc" >/dev/null 2>&1 || return 0
  git push -q origin main >/dev/null 2>&1 && say "  pushed" || say "  PUSH FAILED"
}

say "=== port profile chain (pid $$) ==="
gate
say "running qk_port_profile.py (full)"
if python qk_port_profile.py >> "$LOG" 2>&1; then
  say "profile complete"
  push "port profile by depth complete, numbers in qk_port_profile.json"
else
  say "PROFILE FAILED (see log; partial JSON may exist)"
  push "port profile by depth FAILED (gate or runtime); partial numbers in qk_port_profile.json"
fi
say "=== chain done ==="
