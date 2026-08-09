#!/bin/bash
# The two CAP-OFF corners the depth-3 factorial needs and does not have.
#
# The 2x2x2 at depth 3 width 64 needs eight configurations. tf_factorial3_chain.sh
# runs the four middle ones (our-attention+GELU and softmax+our-feed-forward, both
# caps) plus the four bilinnorm diagnostics.  The two CORNERS -- our model and the
# conventional model -- exist at this cell only WITH the cap
# (tf_vanilla_d3_w64_*, tfb_std7_d3_w64_*, three seeds each).  Their cap-off
# versions do not exist and the decomposition cannot be computed without them.
#
# Gated behind tf_factorial3_chain.sh so the two never compete for the card.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
LOG=tf_factorial3_corners.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
gate() {
  local ok=0 busy free
  say "gating on tf_factorial3_chain.sh"
  while true; do
    busy=0
    pgrep -f -x "/bin/bash ./tf_factorial3_chain.sh" > /dev/null && busy=1
    pgrep -f -- 'python tf_[f]actorial\.py'          > /dev/null && busy=1
    pgrep -f -- 'python tf_[b]aseline_std\.py'       > /dev/null && busy=1
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1)
    if [ "$busy" -eq 0 ] && [ "${free:-0}" -ge 10000 ]; then ok=$((ok+1)); else ok=0; fi
    say "  gate: busy=$busy free=${free}MiB ok=$ok"
    [ "$ok" -ge 3 ] && break
    sleep 120
  done
  say "gate OPEN"
}
push() {
  shopt -s nullglob
  local f=( tff_*d3_w64*.json tfb_std7_d3_w64*noqknorm*.json out_*d3_w64*.txt RESULTS.md )
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
say "=== depth-3 cap-off corners start (pid $$) ==="
gate
for S in 0 1 2; do
  STEM="tff_bilin_bilin_d3_w64_b8192_s${S}_noqknorm"
  if [ -f "${STEM}.pt" ]; then say "$STEM exists"; else
    say "training ${STEM} (our model, cap off)"
    python tf_factorial.py cell --attn bilin --mlp bilin --depth 3 --width 64 \
      --seed "$S" --suffix _noqknorm --no-qk-norm >> "out_${STEM}.txt" 2>&1 \
      && say "  trained" || { say "  TRAIN FAILED"; continue; }
  fi
  [ -f "${STEM}_induction.json" ] || python tf_factorial_probe.py --stem "$STEM" \
    >> "out_${STEM}_probe.txt" 2>&1 && say "  probe OK" || say "  PROBE FAILED"
done
push "depth-3 factorial corners: our model with the cap OFF, three seeds"
for S in 0 1 2; do
  STEM="tfb_std7_d3_w64_b8192_s${S}_noqknorm"
  if [ -f "${STEM}.pt" ]; then say "$STEM exists"; else
    say "training ${STEM} (conventional, cap off)"
    python tf_baseline_std.py cell --exp 7 --depth 3 --width 64 --seed "$S" \
      --suffix _noqknorm --no-qk-norm >> "out_${STEM}.txt" 2>&1 \
      && say "  trained" || { say "  TRAIN FAILED"; continue; }
  fi
  [ -f "${STEM}_induction.json" ] || python tf_baseline_probe.py --stem "$STEM" \
    >> "out_${STEM}_probe.txt" 2>&1 && say "  probe OK" || say "  PROBE FAILED"
done
push "depth-3 factorial corners: conventional with the cap OFF, three seeds -- the 2x2x2 at depth 3 width 64 is now computable"
say "=== depth-3 cap-off corners done ==="
