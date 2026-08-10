#!/bin/bash
# VOCABULARY-SIZE CHECK of FINDING 21's central cell: the cap-off 2x2 at
# depth 2 width 128 retrained at V=4096, seeds 0-2 (12 cells).
#
# Written directly, not derived from another chain (three derivation-inheritance
# bugs on 2026-08-10 all came from reusing arm lists written for other runs).
# The full arm list is spelled out below so it can be read at a glance.
# Predictions: tf_vocab4096_predictions.json, registered before this file.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
LOG=tf_vocab4096_chain.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

gate() {
  local ok=0 napps free
  say "gating on GPU compute processes"
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

push() {
  shopt -s nullglob
  local f=( tff_*_d2_w128_b4096_s[012]_noqknorm*.json tfb_std7_d2_w128_b4096_s[012]_noqknorm*.json \
            out_*b4096*.txt RESULTS.md )
  shopt -u nullglob
  [ ${#f[@]} -eq 0 ] && return 0
  git add -- "${f[@]}" >/dev/null 2>&1
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UkpvpG8hGeMVyGrk8er2uc" >/dev/null 2>&1 || return 0
  git push -q origin main >/dev/null 2>&1 && say "  pushed" || say "  PUSH FAILED"
}

fac() {  # ATTN MLP SEED
  local A=$1 M=$2 S=$3
  local ST="tff_${A}_${M}_d2_w128_b4096_s${S}_noqknorm"
  if [ -f "${ST}.pt" ]; then say "$ST exists -- skip"; else
    say "training $ST"
    python tf_factorial.py cell --attn "$A" --mlp "$M" --depth 2 --width 128 \
      --vocab 4096 --seed "$S" --suffix _noqknorm --no-qk-norm \
      >> "out_${ST}.txt" 2>&1 \
      && say "  trained" || { say "  TRAIN FAILED $ST"; return; }
  fi
  [ -f "${ST}_induction.json" ] || python tf_factorial_probe.py --stem "$ST" \
    >> "out_${ST}_probe.txt" 2>&1 && say "  probe OK" || say "  PROBE FAILED $ST"
}

std() {  # SEED
  local S=$1 ST="tfb_std7_d2_w128_b4096_s${S}_noqknorm"
  if [ -f "${ST}.pt" ]; then say "$ST exists -- skip"; else
    say "training $ST"
    python tf_baseline_std.py cell --exp 7 --depth 2 --width 128 --vocab 4096 \
      --seed "$S" --suffix _noqknorm --no-qk-norm >> "out_${ST}.txt" 2>&1 \
      && say "  trained" || { say "  TRAIN FAILED $ST"; return; }
  fi
  [ -f "${ST}_induction.json" ] || python tf_baseline_probe.py --stem "$ST" \
    >> "out_${ST}_probe.txt" 2>&1 && say "  probe OK" || say "  PROBE FAILED $ST"
}

say "=== vocab-4096 check, cap-off 2x2 at depth 2 width 128 (pid $$) ==="
gate
for S in 0 1 2; do fac bilin bilin "$S"; done
for S in 0 1 2; do fac bilin gelu "$S"; done
for S in 0 1 2; do fac softmax bilin "$S"; done
for S in 0 1 2; do std "$S"; done
push "vocab-4096 check: all four cap-off corners at depth 2 width 128, seeds 0-2"
say "=== vocab-4096 check done ==="
