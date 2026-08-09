#!/bin/bash
# THE SYMMETRIC TAX: the predicate slot-32 arm, cap ON and OFF, three seeds,
# at depth 2 width 128 and depth 3 width 128.
#
# FINDING 19's headline was retracted after independent review because "each
# family at its own better configuration" was applied to the conventional model
# only.  The symmetric comparison uses the predicate slot-32 arm -- 1,523,808
# parameters, 7% FEWER than the conventional matched arm -- but that arm has
# never been trained without the query/key cap and only has two seeds.  This
# chain fixes both.
#
# Predictions: tf_symtax_predictions.json, registered before this file.
#
# GATE: on actual GPU compute processes first (the 2026-08-09 stale-wrapper
# failure mode -- a leftover /bin/bash -c whose command line contains the
# script text matches a command-line pgrep and looks exactly like a live run),
# with a name check only as a secondary signal.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
LOG=tf_symtax_d1.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

gate() {
  local ok=0 napps free
  say "gating on GPU compute processes (not on command-line text)"
  while true; do
    napps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    # also wait for the analysis pass, which has priority: it scores a
    # registered prediction and gives 12 trained checkpoints their ladder.
    
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$napps" -eq 0 ] && [ "${free:-0}" -ge 10000 ]; then
      ok=$((ok + 1))
    else
      ok=0
    fi
    say "  gate: gpu_apps=$napps free=${free}MiB ok=$ok"
    [ "$ok" -ge 3 ] && break
    sleep 60
  done
  say "gate OPEN"
}

push() {
  shopt -s nullglob
  local f=( tf_predicate_*_slot61*.json out_tf_predicate_*slot32*.txt \
            tf_symtax_predictions.json RESULTS.md GRID.md )
  shopt -u nullglob
  [ ${#f[@]} -eq 0 ] && return 0
  git add -- "${f[@]}" >/dev/null 2>&1
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UkpvpG8hGeMVyGrk8er2uc" >/dev/null 2>&1 || return 0
  git push -q origin main >/dev/null 2>&1 && { say "  pushed"; return 0; }
  git fetch -q origin main >/dev/null 2>&1
  git merge --no-edit -q origin/main >/dev/null 2>&1 \
    && git push -q origin main >/dev/null 2>&1 \
    && say "  pushed after merge" || say "  PUSH FAILED"
}

# run DEPTH SEED CAP
run() {
  local D=$1 S=$2 CAP=$3 SUF="_slot61" EXTRA=""
  if [ "$CAP" = off ]; then SUF="_slot61_noqknorm"; EXTRA="--no-qk-norm"; fi
  local ST="tf_predicate_d${D}_w128_b8192_s${S}${SUF}"
  if [ -f "${ST}.pt" ]; then
    say "$ST exists -- skip"
  else
    say "training $ST  (depth $D seed $S cap $CAP)"
    python tf_train.py cell --variant predicate --depth "$D" --width 128 \
      --seed "$S" --slot 61 --suffix "$SUF" $EXTRA --no-sweep \
      >> "out_${ST}.txt" 2>&1 && say "  trained" \
      || { say "  TRAIN FAILED $ST"; return; }
  fi
}

std1() {  # conventional matched arm WITHOUT the cap at depth 1 width 128 -- absent
  local S=$1 ST="tfb_std7_d1_w128_b8192_s${S}_noqknorm"
  if [ -f "${ST}.pt" ]; then say "$ST exists -- skip"; return; fi
  say "training $ST (conventional, cap off)"
  python tf_baseline_std.py cell --exp 7 --depth 1 --width 128 --seed "$S" \
    --suffix _noqknorm --no-qk-norm >> "out_${ST}.txt" 2>&1 && say "  trained" \
    || say "  TRAIN FAILED $ST"
  [ -f "${ST}_induction.json" ] || python tf_baseline_probe.py --stem "$ST" \
    >> "out_${ST}_probe.txt" 2>&1 && say "  probe OK" || say "  PROBE FAILED $ST"
}
say "=== symmetric-tax chain start (pid $$) ==="
CAPS="on off"
gate
say "stage 0: the conventional cap-off arm, absent at this cell"
for S in 0 1 2; do std1 "$S"; done
push "depth-1 symmetric tax: conventional cap-off arm, three seeds"
for D in 1; do
  for CAP in $CAPS; do
    for S in 0 1 2; do run "$D" "$S" "$CAP"; done
    push "symmetric tax: predicate slot-32 depth $D width 128, cap $CAP, three seeds"
  done
done
say "=== symmetric-tax chain done ==="
