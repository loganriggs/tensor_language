#!/bin/bash
# DEPTH 3 WIDTH 128 -- CORRECTED COVERAGE.
#
# tf_factorial4_chain.sh was derived from the depth-3 width-64 REORDERED script,
# which had skipped four cap-off arms because they were already on disk from the
# run before it.  At THIS cell nothing was pre-run, so that derivation silently
# dropped two of the eight corners the 2x2x2 needs -- softmax+our-feed-forward
# and our-attention+GELU, both cap-off.  Caught by auditing the chain against
# what is actually on disk, before the decomposition was attempted.
#
# Order, same principle as before: the cells that unlock the analysis first, the
# known-dead row-L1 diagnostic last.  Everything skips if its checkpoint exists,
# so the 8 cells already trained at this cell are not repeated.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
LOG=tf_factorial5_chain.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

push() {
  shopt -s nullglob
  local f=( tff_*d2_w64*.json tfb_std7_d2_w64*.json out_*d2_w64*.txt \
            RESULTS.md GRID.md )
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

fac() {  # ATTN MLP CAP SEED
  local A=$1 M=$2 CAP=$3 S=$4 SUF="" FLAG=""
  if [ "$CAP" = off ]; then SUF=_noqknorm; FLAG=--no-qk-norm; fi
  local ST="tff_${A}_${M}_d2_w64_b8192_s${S}${SUF}"
  if [ -f "${ST}.pt" ]; then
    say "$ST exists -- skip"
  else
    say "training $ST"
    python tf_factorial.py cell --attn "$A" --mlp "$M" --depth 2 --width 64 \
      --seed "$S" --suffix "$SUF" $FLAG >> "out_${ST}.txt" 2>&1 \
      && say "  trained" || { say "  TRAIN FAILED $ST"; return; }
  fi
  if [ -f "${ST}_induction.json" ]; then
    say "  already probed"
  else
    python tf_factorial_probe.py --stem "$ST" >> "out_${ST}_probe.txt" 2>&1 \
      && say "  probe OK" || say "  PROBE FAILED $ST"
  fi
}

say "=== depth 2 width 64, CORRECTED coverage (pid $$) ==="
gate() {
  local ok=0 busy free
  say "gating on tf_factorial4b_chain.sh"
  while true; do
    busy=0
    pgrep -f -x "/bin/bash ./tf_factorial4b_chain.sh" > /dev/null && busy=1
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
std() {  # conventional, matched params, cap off -- not present at this cell
  local S=$1 ST="tfb_std7_d2_w64_b8192_s${S}_noqknorm"
  if [ -f "${ST}.pt" ]; then say "$ST exists -- skip"; else
    say "training $ST"
    python tf_baseline_std.py cell --exp 7 --depth 2 --width 64 --seed "$S" \
      --suffix _noqknorm --no-qk-norm >> "out_${ST}.txt" 2>&1 \
      && say "  trained" || { say "  TRAIN FAILED $ST"; return; }
  fi
  [ -f "${ST}_induction.json" ] || python tf_baseline_probe.py --stem "$ST" \
    >> "out_${ST}_probe.txt" 2>&1 && say "  probe OK" || say "  PROBE FAILED $ST"
}
gate
say "stage A0: the conventional cap-off corner (absent at this cell)"
for S in 0 1 2; do std "$S"; done
push "depth-2 width-64 factorial: conventional cap-off corner, three seeds"
say "stage A: our cap-off corner and the two cap-off middle arms"
for S in 0 1 2; do fac bilin bilin off "$S"; done
push "depth-2 width-64 factorial: our model cap off, three seeds"
for S in 0 1 2; do fac softmax bilin off "$S"; done
push "depth-2 width-64 factorial: softmax + our feed-forward, cap OFF, three seeds"
for S in 0 1 2; do fac bilin gelu off "$S"; done
push "depth-2 width-64 factorial: our attention + GELU, cap OFF, three seeds -- the cap-off 2x2 is complete"
say "stage B: the cap-on arms"
for S in 0 1 2; do fac softmax bilin on "$S"; done
for S in 0 1 2; do fac bilin gelu on "$S"; done
push "depth-2 width-64 factorial: cap-on arms, three seeds -- THE FULL 2x2x2 AT THIS CELL IS COMPLETE"
say "stage C: the row-L1 diagnostic, last"
for S in 0 1 2; do fac bilinnorm bilin off "$S"; done
for S in 0 1 2; do fac bilinnorm gelu off "$S"; done
for S in 0 1 2; do fac bilinnorm bilin on "$S"; done
for S in 0 1 2; do fac bilinnorm gelu on "$S"; done
push "depth-2 width-64 factorial: row-L1 diagnostic, three seeds"
say "=== done ==="
