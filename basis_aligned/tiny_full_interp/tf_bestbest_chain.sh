#!/bin/bash
# BEST-AGAINST-BEST AT MATCHED PARAMETERS.
#
# The query/key-norm control settled the asymmetry: removing the per-head cap
# HELPS the conventional model by 0.175 nats and HURTS the foldable family by
# 0.039.  The cap is load-bearing for us and a handicap for them, so the only
# symmetric way to quote a foldability tax is each family at ITS OWN better
# configuration.
#
# That gives +0.2071 nats at depth 2 width 128 -- but the un-capped
# conventional number on disk is the 4x arm, which has ~12% FEWER parameters
# than the family.  This chain fills the missing cell: the MATCHED-parameter
# (7x) conventional arm without the cap, three seeds.  Until it lands, the
# best-against-best tax is quoted at a parameter DISadvantage to the
# conventional model and is therefore itself a lower bound.
#
# Predictions are in tf_qknorm_predictions.json (the decision rule fixed in
# advance there covers this) plus the registered expectation below:
#   the matched un-capped arm beats the 4x un-capped arm, so the
#   best-against-best tax at matched parameters EXCEEDS +0.2071.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
LOG=tf_bestbest_chain.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

push() {
  shopt -s nullglob
  local files=( tfb_std7_*_noqknorm*.json out_tfb_std7_*noqknorm*.txt \
                tf_qknorm.json tf_qknorm_table.md RESULTS.md GRID.md )
  shopt -u nullglob
  [ ${#files[@]} -eq 0 ] && return 0
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

say "=== best-against-best, matched parameters, un-capped (pid $$) ==="
for S in 0 1 2; do
  STEM="tfb_std7_d2_w128_b8192_s${S}_noqknorm"
  if [ -f "${STEM}.pt" ]; then
    say "$STEM exists -- skip"
  else
    say "training ${STEM}"
    python tf_baseline_std.py cell --exp 7 --depth 2 --width 128 --seed "$S" \
      --suffix _noqknorm --no-qk-norm >> "out_${STEM}.txt" 2>&1 \
      || { say "  TRAIN FAILED ${STEM}"; continue; }
    say "  trained"
  fi
  [ -f "${STEM}_induction.json" ] && say "  already probed" || {
    python tf_baseline_probe.py --stem "$STEM" \
      >> "out_${STEM}_probe.txt" 2>&1 && say "  probe OK" \
      || say "  PROBE FAILED ${STEM}"; }
  push "best-against-best: matched-parameter un-capped conventional, seed $S"
done
python tf_qknorm_report.py >> tf_qknorm_report_stdout.txt 2>&1 \
  && say "report OK" || say "REPORT FAILED"
push "best-against-best: matched-parameter un-capped conventional, all three seeds"
say "=== done ==="
