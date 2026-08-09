#!/bin/bash
# THE ATTENTION x FEED-FORWARD FACTORIAL CHAIN.
#
# The conventional baseline chain established that a softmax+GELU transformer
# inducts one octave of width earlier than our foldable family.  It changed two
# things at once.  This chain changes them one at a time.
#
# GATED behind tf_baseline_chain.sh, which owns the card right now.  The gate
# uses an exact-name pgrep and a [c]haracter class so it cannot self-match --
# substring pgrep self-matches have killed runs in this programme twice.
#
# Stages, ordered so the discriminating cell lands first and every stage leaves
# a scored table on disk:
#   0  gates G1-G4 and the probe corner control (cheap, re-run anyway)
#   1  depth 2 width 128 -- family null, conventional +0.189.  4 new arms
#   2  depth 3 width 64  -- family null, conventional +0.103.  4 new arms
#   3  depth 1 width 128 -- the negative control; nothing may induct
#   4  depth 2 width 256 and depth 3 width 128 -- both families induct
#   5  seeds 1 and 2 at depth 2 width 128
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
LOG=tf_factorial_chain.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

gate() {
  local ok=0 busy free
  say "gating on tf_baseline_chain.sh + a quiet card"
  while true; do
    busy=0
    pgrep -f -x "/bin/bash ./tf_baseline_chain.sh"  > /dev/null && busy=1
    pgrep -f -- 'python tf_[b]aseline_std\.py'      > /dev/null && busy=1
    pgrep -f -- 'python tf_[b]aseline_probe\.py'    > /dev/null && busy=1
    pgrep -f -- 'python tf_[t]rain\.py'             > /dev/null && busy=1
    pgrep -f -- 'python tf_[i]nterp3\.py'           > /dev/null && busy=1
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
           | head -1)
    if [ "$busy" -eq 0 ] && [ "${free:-0}" -ge 10000 ]; then
      ok=$((ok + 1))
    else
      ok=0
    fi
    say "  gate: busy=$busy free=${free}MiB consecutive_ok=$ok"
    [ "$ok" -ge 3 ] && break
    sleep 120
  done
  say "gate OPEN"
}

push() {
  shopt -s nullglob
  local files=( tff_*.json tf_factorial*.json tf_factorial_table.md \
                out_tff_*.txt GRID.md RESULTS.md )
  shopt -u nullglob
  [ ${#files[@]} -eq 0 ] && { say "  nothing to stage"; return 0; }
  git add -- "${files[@]}" > /dev/null 2>&1
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UkpvpG8hGeMVyGrk8er2uc" \
    > /dev/null 2>&1 || { say "  nothing to commit"; return 0; }
  git push -q origin main > /dev/null 2>&1 && { say "  pushed"; return 0; }
  git fetch -q origin main > /dev/null 2>&1
  git merge --no-edit -q origin/main > /dev/null 2>&1 \
    && git push -q origin main > /dev/null 2>&1 \
    && { say "  pushed after merge"; return 0; }
  say "  PUSH FAILED (next stage retries)"
}

report() {
  python tf_factorial_report.py >> tf_factorial_report_stdout.txt 2>&1 \
    && say "  report OK" || say "  REPORT FAILED"
  push "$1"
}

# run_arm ATTN MLP DEPTH WIDTH SEED
run_arm() {
  local A=$1 M=$2 D=$3 W=$4 S=$5
  local STEM="tff_${A}_${M}_d${D}_w${W}_b8192_s${S}"
  if [ -f "${STEM}.pt" ]; then
    say "$STEM exists -- skip training"
  else
    say "training ${STEM}"
    python tf_factorial.py cell --attn "$A" --mlp "$M" --depth "$D" \
      --width "$W" --seed "$S" >> "out_${STEM}.txt" 2>&1 \
      || { say "  TRAIN FAILED ${STEM}"; return; }
    say "  trained"
  fi
  if [ -f "${STEM}_induction.json" ]; then
    say "  $STEM already probed -- skip"
  else
    python tf_factorial_probe.py --stem "$STEM" \
      >> "out_${STEM}_probe.txt" 2>&1 && say "  probe OK" \
      || say "  PROBE FAILED ${STEM}"
  fi
}

# The four arms that are NOT already on disk.  (bilin,bilin) is the published
# family model and (softmax,gelu) is the published conventional baseline; the
# report reads those two from their existing files rather than retraining them,
# which is also a free consistency check on the corner claim.
NEW_ARMS="bilin:gelu bilinnorm:bilin softmax:bilin bilinnorm:gelu"

cell_pass() {
  local D=$1 W=$2 S=$3
  for A in $NEW_ARMS; do
    run_arm "${A%%:*}" "${A##*:}" "$D" "$W" "$S"
  done
}

say "=== factorial chain start (pid $$) ==="

# ---- stage 0: gates ----
python tf_factorial.py controls >> out_tff_controls.txt 2>&1 \
  && say "gates OK" || { say "GATES FAILED -- stopping"; exit 1; }
python -c "
import json,sys
sys.exit(0 if json.load(open('tf_factorial_controls.json'))['all_pass'] else 1)
" || { say "GATES DID NOT ALL PASS -- stopping"; exit 1; }
if [ ! -f tf_factorial_probe_control.json ]; then
  python tf_factorial_probe.py --control >> out_tff_controls.txt 2>&1 \
    && say "probe corner control OK" || say "PROBE CORNER CONTROL FAILED"
fi
push "factorial: gates G1-G4 and the probe corner control"

gate

# ---- stage 1: the discriminating cell ----
say "stage 1: depth 2 width 128 -- family null, conventional inducts"
cell_pass 2 128 0
report "factorial: depth 2 width 128 seed 0, all four new arms"

# ---- stage 2: the other discriminating cell ----
say "stage 2: depth 3 width 64"
cell_pass 3 64 0
report "factorial: depth 3 width 64 seed 0"

# ---- stage 3: the negative control ----
say "stage 3: depth 1 width 128 -- nothing may induct"
cell_pass 1 128 0
report "factorial: depth 1 width 128 negative control"

# ---- stage 4: cells where both families already induct ----
say "stage 4: depth 2 width 256 and depth 3 width 128"
cell_pass 2 256 0
report "factorial: depth 2 width 256 seed 0"
cell_pass 3 128 0
report "factorial: depth 3 width 128 seed 0"

# ---- stage 5: seeds ----
say "stage 5: seeds 1 and 2 at the discriminating cell"
for S in 1 2; do
  cell_pass 2 128 "$S"
  report "factorial: depth 2 width 128 seed $S"
done

report "factorial: FINAL -- all cells, scored against the predictions registered before the code was written"
say "=== factorial chain done ==="
