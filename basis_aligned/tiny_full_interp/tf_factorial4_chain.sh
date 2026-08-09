#!/bin/bash
# DEPTH-3 FACTORIAL, REORDERED so the analysis-unlocking cells run FIRST.
#
# The original order put the row-L1 diagnostic's remaining nine cells ahead of
# the twelve cells the 2x2x2 actually needs.  That diagnostic is now measured
# NULL at both cells (depth 2 width 128: -0.0078 and +0.0251; depth 3 width 64:
# +0.0011 at t = 0.55), so nine more of its cells were buying nothing while the
# decomposition stayed uncomputable.  Nine cells already on disk are kept and
# skipped; nothing is retrained.
#
# Order:
#   1. the two CORNERS with the cap off -- our model and the conventional one.
#      Without these there is no baseline and no total move, so no share.
#   2. the two remaining 2x2x2 arms with the cap on.
#   3. the row-L1 remainder, last, for completeness of the 3x2x2 table.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
LOG=tf_factorial4_chain.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
push() {
  shopt -s nullglob
  local f=( tff_*d3_w128*.json tfb_std7_d3_w128*.json out_*d3_w128*.txt RESULTS.md )
  shopt -u nullglob
  [ ${#f[@]} -eq 0 ] && return 0
  git add -- "${f[@]}" >/dev/null 2>&1
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UkpvpG8hGeMVyGrk8er2uc" >/dev/null 2>&1 || return 0
  git push -q origin main >/dev/null 2>&1 && { say "  pushed"; return 0; }
  git fetch -q origin main >/dev/null 2>&1
  git merge --no-edit -q origin/main >/dev/null 2>&1 && git push -q origin main >/dev/null 2>&1 \
    && say "  pushed after merge" || say "  PUSH FAILED"
}
fac() {  # ATTN MLP CAP SEED
  local A=$1 M=$2 CAP=$3 S=$4 SUF="" FLAG=""
  [ "$CAP" = off ] && { SUF=_noqknorm; FLAG=--no-qk-norm; }
  local ST="tff_${A}_${M}_d3_w128_b8192_s${S}${SUF}"
  if [ -f "${ST}.pt" ]; then say "$ST exists -- skip"; else
    say "training $ST"
    python tf_factorial.py cell --attn "$A" --mlp "$M" --depth 3 --width 128 \
      --seed "$S" --suffix "$SUF" $FLAG >> "out_${ST}.txt" 2>&1 \
      && say "  trained" || { say "  TRAIN FAILED $ST"; return; }
  fi
  [ -f "${ST}_induction.json" ] || python tf_factorial_probe.py --stem "$ST" \
    >> "out_${ST}_probe.txt" 2>&1 && say "  probe OK" || say "  PROBE FAILED $ST"
}
std() {  # SEED   (conventional, matched params, cap off)
  local S=$1 ST="tfb_std7_d3_w128_b8192_s${S}_noqknorm"
  if [ -f "${ST}.pt" ]; then say "$ST exists -- skip"; else
    say "training $ST"
    python tf_baseline_std.py cell --exp 7 --depth 3 --width 128 --seed "$S" \
      --suffix _noqknorm --no-qk-norm >> "out_${ST}.txt" 2>&1 \
      && say "  trained" || { say "  TRAIN FAILED $ST"; return; }
  fi
  [ -f "${ST}_induction.json" ] || python tf_baseline_probe.py --stem "$ST" \
    >> "out_${ST}_probe.txt" 2>&1 && say "  probe OK" || say "  PROBE FAILED $ST"
}
gate() {
  local ok=0 busy free
  say "gating on tf_factorial3_reordered.sh"
  while true; do
    busy=0
    pgrep -f -x "/bin/bash ./tf_factorial3_reordered.sh" > /dev/null && busy=1
    pgrep -f -- 'python tf_[f]actorial\.py'    > /dev/null && busy=1
    pgrep -f -- 'python tf_[b]aseline_std\.py' > /dev/null && busy=1
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1)
    if [ "$busy" -eq 0 ] && [ "${free:-0}" -ge 10000 ]; then ok=$((ok+1)); else ok=0; fi
    say "  gate: busy=$busy free=${free}MiB ok=$ok"
    [ "$ok" -ge 3 ] && break
    sleep 120
  done
  say "gate OPEN"
}
say "=== DEPTH 3 WIDTH 128 -- depth-vs-width separator (pid $$) ==="
gate
say "stage 1: the two cap-off corners -- nothing is computable without them"
for S in 0 1 2; do fac bilin bilin off "$S"; done
push "depth-3 width-128 factorial: our model with the cap off, three seeds"
for S in 0 1 2; do std "$S"; done
push "depth-3 width-128 factorial: conventional model with the cap off, three seeds -- the 2x2x2 baseline and total move now exist"
say "stage 2: the two remaining 2x2x2 arms, cap on"
for S in 0 1 2; do fac softmax bilin on "$S"; done
push "depth-3 width-128 factorial: softmax + our feed-forward, cap on, three seeds"
for S in 0 1 2; do fac bilin gelu on "$S"; done
push "depth-3 width-128 factorial: our attention + GELU, cap on, three seeds -- THE 2x2x2 AT DEPTH 3 WIDTH 64 IS COMPLETE"
say "stage 3: the row-L1 remainder, last (already null at both cells)"
for S in 0 1 2; do fac bilinnorm gelu off "$S"; done
for S in 0 1 2; do fac bilinnorm bilin on "$S"; done
for S in 0 1 2; do fac bilinnorm gelu on "$S"; done
push "depth-3 width-128 factorial: row-L1 diagnostic remainder, three seeds"
say "=== depth 3 width 128 done ==="
