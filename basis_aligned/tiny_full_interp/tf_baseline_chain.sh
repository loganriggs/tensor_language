#!/bin/bash
# THE FOLDABILITY TAX CHAIN — conventional softmax+GELU baseline.
#
# GRID.md has listed a same-size conventional baseline as unclaimed since the
# programme started, and STANDALONE_RESULTS.md section 8 lists it as open
# question 1.  Everything this programme has measured is relative to another
# FOLDABLE model; this chain measures what the foldability costs.
#
# GATED, per AGENT_BRIEF, on the geometry-control chain finishing: exact-name
# pgrep on tf_geom_control_chain.sh AND on each of its runners (the [c]haracter
# class keeps pgrep from matching itself — substring pgrep self-matches have
# killed runs in this programme twice), plus >= 10000 MiB of free card for
# THREE consecutive checks.  Do not compete for the card.
#
# Stages, ordered so the headline cell and two seeds of everything land first:
#   0  controls (model, harness, probe shim)
#   1  seed 0, nominal 4x then matched 7x, nine cells each
#   2  seed 1, same
#   3  query/key-norm control at depth 2 width 128, three seeds
#   4  learning-rate fairness bound at width 128, depths 1-3
#   5  seed 2, same nine cells
#   6  final report
# The report is rebuilt and pushed after every stage, so a chain killed
# half-way still leaves a scored, seed-counted table on disk.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
LOG=tf_baseline_chain.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ------------------------------------------------------------------ the gate
gate() {
  local ok=0 busy free
  say "gating on tf_geom_control_chain.sh + a quiet card"
  while true; do
    busy=0
    pgrep -f -x "/bin/bash ./tf_geom_control_chain.sh" > /dev/null && busy=1
    pgrep -f -- 'python tf_[t]rain\.py'          > /dev/null && busy=1
    pgrep -f -- 'python tf_[i]nterp3\.py'        > /dev/null && busy=1
    pgrep -f -- 'python tf_[d]epth_addendum\.py' > /dev/null && busy=1
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
           | head -1)
    if [ "$busy" -eq 0 ] && [ "${free:-0}" -ge 10000 ]; then
      ok=$((ok + 1))
    else
      ok=0
    fi
    say "  gate: busy=$busy free=${free}MiB consecutive_ok=$ok"
    [ "$ok" -ge 3 ] && break
    sleep 60
  done
  say "gate OPEN"
}

push() {
  # nullglob, because `git add` on a pathspec that matches nothing fails
  # WHOLESALE and would silently stage none of the results.
  shopt -s nullglob
  local files=( tfb_*.json tf_baseline_std.json tf_baseline_controls.json \
                tf_baseline_probe_control.json tf_baseline_table.md \
                tf_baseline_predictions.json out_tfb_*.txt )
  shopt -u nullglob
  [ ${#files[@]} -eq 0 ] && { say "  nothing to stage yet"; return 0; }
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
  python tf_baseline_report.py >> tf_baseline_report_stdout.txt 2>&1 \
    && say "  report OK" || say "  REPORT FAILED"
  push "$1"
}

# run_cell EXP DEPTH WIDTH SEED SUFFIX [extra args to tf_baseline_std.py]
run_cell() {
  local EXP=$1 D=$2 W=$3 S=$4 SUF=$5
  shift 5
  local STEM="tfb_std${EXP}_d${D}_w${W}_b8192_s${S}${SUF}"
  if [ -f "${STEM}.pt" ]; then
    say "$STEM checkpoint exists -- skip training"
  else
    say "training ${STEM}"
    python tf_baseline_std.py cell --exp "$EXP" --depth "$D" --width "$W" \
      --seed "$S" --suffix "$SUF" "$@" >> "out_${STEM}.txt" 2>&1 \
      || { say "  TRAIN FAILED ${STEM}"; return; }
    say "  trained"
  fi
  if [ -f "${STEM}_induction.json" ]; then
    say "  $STEM already probed -- skip"
  else
    python tf_baseline_probe.py --stem "$STEM" \
      >> "out_${STEM}_probe.txt" 2>&1 && say "  probe OK" \
      || say "  PROBE FAILED ${STEM}"
  fi
}

# the nine cells, ordered so the headline (depth 2 width 128) lands first
CELLS="2:128 1:128 3:128 2:64 2:256 1:64 1:256 3:64 3:256"

seed_pass() {
  local S=$1
  for EXP in 4 7; do
    for C in $CELLS; do
      run_cell "$EXP" "${C%%:*}" "${C##*:}" "$S" ""
    done
    report "conventional baseline: seed $S, expansion ${EXP}x, cells done"
  done
}

say "=== foldability-tax chain start (pid $$) ==="
gate

# ---- stage 0: controls ----
if [ -f tf_baseline_controls.json ]; then
  say "controls already run -- skip"
else
  say "stage 0: controls"
  python tf_baseline_std.py controls >> out_tfb_controls.txt 2>&1 \
    && say "  model/harness controls OK" || say "  CONTROLS FAILED"
fi
if [ -f tf_baseline_probe_control.json ]; then
  say "probe control already run -- skip"
else
  python tf_baseline_probe.py --control >> out_tfb_controls.txt 2>&1 \
    && say "  probe-shim control OK" || say "  PROBE CONTROL FAILED"
fi
push "conventional baseline: positive controls (parameter identity, naive-reference forward, causality, data identity, training-loop equivalence, probe shim)"

# ---- stages 1-2: seeds 0 and 1 ----
seed_pass 0
seed_pass 1

# ---- stage 3: the query/key-norm control ----
say "stage 3: query/key-norm control, depth 2 width 128, three seeds"
for S in 0 1 2; do
  run_cell 4 2 128 "$S" "_noqknorm" --no-qk-norm
done
report "conventional baseline: query/key-norm control at depth 2 width 128"

# ---- stage 4: the learning-rate fairness bound ----
say "stage 4: learning-rate bound at width 128"
for D in 2 1 3; do
  for LR in 0.01 0.04; do
    run_cell 4 "$D" 128 0 "_lr${LR}" --lr "$LR"
  done
done
for LR in 0.01 0.04; do
  run_cell 7 2 128 0 "_lr${LR}" --lr "$LR"
done
report "conventional baseline: learning-rate fairness bound (0.01/0.02/0.04 at full length, width 128)"

# ---- stage 5: the third seed ----
seed_pass 2

# ---- stage 6 ----
report "conventional baseline: FINAL — all cells, all seeds, scored against the predictions registered before training"
say "=== foldability-tax chain done ==="
