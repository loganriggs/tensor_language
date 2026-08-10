#!/bin/bash
# SECOND FOLDABLE ARCHITECTURE: does the copying-availability sign split
# reproduce with the bandwidth variant, or is FINDING 22 a predicate-variant
# result?
#
# Written directly rather than derived by substitution from another chain --
# two derivation bugs today (a dropped factorial corner, and a width that never
# changed) both came from sed-editing a script whose structure had moved on.
#
#   no-copying cell: depth 1 width 128, bandwidth slot 61 (1,241,533 params)
#                    vs the existing conventional cap-off arm (1,343,616)
#                    -> our handicap 7.60%
#   copying cell:    depth 3 width 64, bandwidth slot 11 (716,001 params)
#                    vs the existing conventional expansion-8 cap-off arm
#                    (770,240) -> our handicap 7.04%
# Both opponents already trained. Predictions in tf_secondarch_predictions.json.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
LOG=tf_secondarch.log
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
  local f=( tf_bandwidth_d*_slot*.json out_tf_bandwidth_d*_slot*.txt RESULTS.md )
  shopt -u nullglob
  [ ${#f[@]} -eq 0 ] && return 0
  git add -- "${f[@]}" >/dev/null 2>&1
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UkpvpG8hGeMVyGrk8er2uc" >/dev/null 2>&1 || return 0
  git push -q origin main >/dev/null 2>&1 && say "  pushed" || say "  PUSH FAILED"
}

# train DEPTH WIDTH SLOT SEED CAP
train_one() {
  local D=$1 W=$2 SLOT=$3 S=$4 CAP=$5 SUF="_slot${3}" FLAG=""
  if [ "$CAP" = off ]; then SUF="_slot${3}_noqknorm"; FLAG="--no-qk-norm"; fi
  local ST="tf_bandwidth_d${D}_w${W}_b8192_s${S}${SUF}"
  if [ -f "${ST}.pt" ]; then
    say "$ST exists -- skip"
  else
    say "training $ST  (depth $D width $W slot $SLOT cap $CAP)"
    python tf_train.py cell --variant bandwidth --depth "$D" --width "$W" \
      --seed "$S" --slot "$SLOT" --suffix "$SUF" $FLAG --no-sweep \
      >> "out_${ST}.txt" 2>&1 && say "  trained" \
      || { say "  TRAIN FAILED $ST"; return; }
  fi
}

say "=== second foldable architecture (pid $$) ==="
gate
for SPEC in "1:128:61" "3:64:11"; do
  IFS=: read -r D W SLOT <<< "$SPEC"
  say "--- depth $D width $W slot $SLOT ---"
  for CAP in on off; do
    for S in 0 1 2; do
      train_one "$D" "$W" "$SLOT" "$S" "$CAP"
    done
    push "second architecture: bandwidth depth $D width $W slot $SLOT, cap $CAP, three seeds"
  done
done
say "=== second architecture done ==="
