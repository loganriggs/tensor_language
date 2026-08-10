#!/bin/bash
# PHASE V2 TRANSITION SWEEP: slots w32/w64/w256, predicate w32, vanilla w192,
# depth 2, seeds 0-2, V=8192. Train (concurrent seed groups), then the FULL
# tf_interp3 ladder per checkpoint (sequential; it is the deliverable and
# carries the induction battery that scores the predictions).
#
# Written directly; arm list spelled out. Predictions:
# tf_v2_transition_predictions.json, registered before this file.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
LOG=tf_v2_transition_chain.log
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
  local f=( tf_slots_d2_w32_* tf_slots_d2_w64_* tf_slots_d2_w256_* \
            tf_predicate_d2_w32_* tf_vanilla_d2_w192_* out_v2t_*.txt RESULTS.md )
  local keep=() x
  for x in "${f[@]}"; do [[ "$x" == *.pt ]] || keep+=("$x"); done
  shopt -u nullglob
  [ ${#keep[@]} -eq 0 ] && return 0
  git add -- "${keep[@]}" >/dev/null 2>&1
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UkpvpG8hGeMVyGrk8er2uc" >/dev/null 2>&1 || return 0
  git push -q origin main >/dev/null 2>&1 && say "  pushed" || say "  PUSH FAILED"
}

# group VARIANT WIDTH -- seeds 0-2 concurrent, then ladders sequential
group() {
  local VAR=$1 W=$2 t0=$SECONDS pids=() S rc=0 p
  for S in 0 1 2; do
    local ST="tf_${VAR}_d2_w${W}_b8192_s${S}"
    if [ -f "${ST}.pt" ]; then say "$ST exists -- skip train"; continue; fi
    say "training $ST (concurrent)"
    python tf_train.py cell --variant "$VAR" --depth 2 --width "$W" --seed "$S" \
      --no-sweep >> "out_v2t_${ST}.txt" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]:-}"; do [ -n "$p" ] && { wait "$p" || rc=1; }; done
  say "  group ${VAR} w${W} wall $((SECONDS - t0)) s"
  [ "$rc" -ne 0 ] && say "  TRAIN FAILED in ${VAR} w${W}"
  for S in 0 1 2; do
    local ST="tf_${VAR}_d2_w${W}_b8192_s${S}"
    [ -f "${ST}.pt" ] || continue
    if [ -f "${ST}_interp3.json" ] && python -c "
import json,sys
d=json.load(open('${ST}_interp3.json'))
sys.exit(0 if 'bits_per_byte_ladder' in d else 1)" 2>/dev/null; then
      say "  ladder complete $ST -- skip"
    else
      say "  ladder $ST"
      python tf_interp3.py --stem "$ST" >> "out_v2t_ladder_${ST}.txt" 2>&1 \
        && say "  ladder OK" || say "  LADDER FAILED $ST"
    fi
  done
}

say "=== V2 transition sweep (pid $$) ==="
gate
group slots 64
group slots 32
group predicate 32
group slots 256
group vanilla 192
push "V2 transition sweep: slots w32/64/256, predicate w32, vanilla w192, three seeds each, full ladders"
say "=== V2 transition sweep done ==="
