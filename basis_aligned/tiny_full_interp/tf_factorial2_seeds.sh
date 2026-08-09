#!/bin/bash
# THE THREE-FACTOR FACTORIAL: attention x feed-forward x query/key cap.
#
# The two-factor version was stopped before writing a checkpoint because it
# held the cap at ON for every arm, including its softmax arms.  FINDING 19
# shows the cap moves the conventional model -0.1745 nats and the foldable
# family +0.0392 -- OPPOSITE SIGNS -- so it is not independent of the attention
# factor and cannot be a fixed setting.  It is varied here as a third factor.
#
# 12 arms; 4 are already on disk (both families x both cap settings, 3 seeds
# each), so this chain runs the 8 that are missing.
# Predictions: tf_factorial2_predictions.json, registered before this file.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
LOG=tf_factorial2_seeds.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

push() {
  shopt -s nullglob
  local files=( tff_*.json out_tff_*.txt tf_factorial2_predictions.json \
                tf_factorial_table.md tf_factorial.json RESULTS.md GRID.md )
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
    && git push -q origin main > /dev/null 2>&1 && { say "  pushed after merge"; return 0; }
  say "  PUSH FAILED (next stage retries)"
}

# run_arm ATTN MLP CAP SEED     CAP is "on" or "off"
run_arm() {
  local A=$1 M=$2 CAP=$3 S=$4
  local SUF="" ; local FLAG=""
  if [ "$CAP" = "off" ]; then SUF="_noqknorm"; FLAG="--no-qk-norm"; fi
  local STEM="tff_${A}_${M}_d2_w128_b8192_s${S}${SUF}"
  if [ -f "${STEM}.pt" ]; then say "$STEM exists -- skip"; else
    say "training ${STEM}  (attn=$A mlp=$M cap=$CAP)"
    python tf_factorial.py cell --attn "$A" --mlp "$M" --depth 2 --width 128 \
      --seed "$S" --suffix "$SUF" $FLAG >> "out_${STEM}.txt" 2>&1 \
      || { say "  TRAIN FAILED ${STEM}"; return; }
    say "  trained"
  fi
  [ -f "${STEM}_induction.json" ] && say "  already probed" || {
    python tf_factorial_probe.py --stem "$STEM" >> "out_${STEM}_probe.txt" 2>&1 \
      && say "  probe OK" || say "  PROBE FAILED ${STEM}"; }
}

say "=== three-factor factorial SEEDS 1-2 start (pid $$) ==="
python tf_factorial.py controls >> out_tff_controls.txt 2>&1 \
  && say "gates OK" || { say "GATES FAILED -- stopping"; exit 1; }
python -c "
import json,sys
sys.exit(0 if json.load(open('tf_factorial_controls.json'))['all_pass'] else 1)
" || { say "GATES DID NOT ALL PASS -- stopping"; exit 1; }

# The 8 missing arms, cap-OFF first because P1 and P2 are stated there.
for SPEC in "softmax:bilin:off" "bilinnorm:bilin:off" "bilin:gelu:off" \
            "bilinnorm:gelu:off" "softmax:bilin:on"  "bilinnorm:bilin:on" \
            "bilin:gelu:on"     "bilinnorm:gelu:on"; do
  IFS=: read -r A M CAP <<< "$SPEC"
  for S in 1 2; do run_arm "$A" "$M" "$CAP" "$S"; done
  push "three-factor factorial: ${A} attention + ${M} feed-forward, cap ${CAP}, seeds 1-2"
done
say "=== three-factor factorial seeds 1-2 done ==="
