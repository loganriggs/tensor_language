#!/bin/bash
# FIX-ROUND MEASUREMENTS (18:10 review): seeds 3-5 for all four cap-off corners
# at both V=4096 cells, plus the co-tenancy control (sequential retrain of the
# conventional d3 arm, seed 0, suffix _seqctl).
#
# ALL SEQUENTIAL, deliberately: objection 4 is about co-tenancy, so its
# settlement and the seed extension must not be co-tenant themselves.
# Written directly; arm list spelled out.
# Predictions: tf_v4096_seeds35_predictions.json (committed before launch).
# Gates behind the V2 transition sweep.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
LOG=tf_v4096_seeds35_chain.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

gate() {
  local ok=0 napps free
  say "gating on GPU + no V2 transition chain"
  while true; do
    napps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    pgrep -f "tf_v2_transition_chain.sh" > /dev/null && napps=$((napps + 1))
    if [ "$napps" -eq 0 ] && [ "${free:-0}" -ge 10000 ]; then ok=$((ok + 1)); else ok=0; fi
    say "  gate: busy=$napps free=${free}MiB ok=$ok"
    [ "$ok" -ge 3 ] && break
    sleep 60
  done
  say "gate OPEN"
}

push() {
  shopt -s nullglob
  local f=( tff_*_b4096_s[345]_noqknorm*.json tfb_std7_*_b4096_s[345]_noqknorm*.json \
            tfb_std7_d3_w128_b4096_s0_noqknorm_seqctl*.json out_v4096s35_*.txt RESULTS.md )
  shopt -u nullglob
  [ ${#f[@]} -eq 0 ] && return 0
  git add -- "${f[@]}" >/dev/null 2>&1
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UkpvpG8hGeMVyGrk8er2uc" >/dev/null 2>&1 || return 0
  git push -q origin main >/dev/null 2>&1 && say "  pushed" || say "  PUSH FAILED"
}

fac() {  # ATTN MLP DEPTH SEED
  local A=$1 M=$2 D=$3 S=$4
  local ST="tff_${A}_${M}_d${D}_w128_b4096_s${S}_noqknorm"
  if [ -f "${ST}.pt" ]; then say "$ST exists -- skip"; else
    say "training $ST (sequential)"
    python tf_factorial.py cell --attn "$A" --mlp "$M" --depth "$D" --width 128 \
      --vocab 4096 --seed "$S" --suffix _noqknorm --no-qk-norm \
      >> "out_v4096s35_${ST}.txt" 2>&1 \
      && say "  trained" || { say "  TRAIN FAILED $ST"; return; }
  fi
  [ -f "${ST}_induction.json" ] || python tf_factorial_probe.py --stem "$ST" \
    >> "out_v4096s35_${ST}_probe.txt" 2>&1 && say "  probe OK" || say "  PROBE FAILED $ST"
}

std() {  # DEPTH SEED [SUFFIX]
  local D=$1 S=$2 SUF="${3:-_noqknorm}"
  local ST="tfb_std7_d${D}_w128_b4096_s${S}${SUF}"
  if [ -f "${ST}.pt" ]; then say "$ST exists -- skip"; else
    say "training $ST (sequential)"
    python tf_baseline_std.py cell --exp 7 --depth "$D" --width 128 --vocab 4096 \
      --seed "$S" --suffix "$SUF" --no-qk-norm >> "out_v4096s35_${ST}.txt" 2>&1 \
      && say "  trained" || { say "  TRAIN FAILED $ST"; return; }
  fi
  [ -f "${ST}_induction.json" ] || python tf_baseline_probe.py --stem "$ST" \
    >> "out_v4096s35_${ST}_probe.txt" 2>&1 && say "  probe OK" || say "  PROBE FAILED $ST"
}

say "=== v4096 seeds 3-5 + co-tenancy control (pid $$) ==="
gate
say "--- co-tenancy control first (objection 4) ---"
std 3 0 _noqknorm_seqctl
push "co-tenancy control trained: conventional d3 w128 V=4096 seed 0, sequential twin of the concurrent arm"
say "--- depth 3 corners, seeds 3-5 ---"
for S in 3 4 5; do fac bilin bilin 3 "$S"; done
for S in 3 4 5; do fac bilin gelu 3 "$S"; done
for S in 3 4 5; do fac softmax bilin 3 "$S"; done
for S in 3 4 5; do std 3 "$S"; done
push "v4096 depth-3 corners at six seeds"
say "--- depth 2 corners, seeds 3-5 ---"
for S in 3 4 5; do fac bilin bilin 2 "$S"; done
for S in 3 4 5; do fac bilin gelu 2 "$S"; done
for S in 3 4 5; do fac softmax bilin 2 "$S"; done
for S in 3 4 5; do std 2 "$S"; done
push "v4096 depth-2 corners at six seeds -- both vocabulary cells now six seeds per corner"
say "=== v4096 seeds 3-5 done ==="
