#!/bin/bash
# Phase V1 training chain: the five non-vanilla variants at depth 2, width 128,
# seed 0, plus the matched-parameter control C0.  Vanilla is already trained and
# is NOT retrained (same data order, same optimizer, same steps).
#
# EVERY ARM USES MUON lr 0.02 WITH NO SWEEP, which is exactly what the vanilla
# checkpoint used (its JSON has no lrsweep block and its log records lr 0.02).
# A per-variant lr sweep would make the arms non-matched, which is the mistake
# the parent program made for a week; the sweep is run separately afterwards as
# a robustness check, not as the primary arm.
cd "$(dirname "$0")" || exit 1
source /venv/main/bin/activate
export PYTHONUNBUFFERED=1
LOG=tf_variant_train.log
: > "$LOG"
run () {   # run <variant> [extra args...]
  echo "=== $(date -u +%H:%M:%S)  variant $*" >> "$LOG"
  python tf_train.py cell --variant "$1" --depth 2 --width 128 --seed 0 \
      --vocab 8192 --tok bpe --no-sweep "${@:2}" >> "$LOG" 2>&1 \
      || echo "!!! FAILED $*" >> "$LOG"
}
run slots
run bandwidth
run predicate
run codebook
run shrink
# matched-parameter control: bandwidth at the VANILLA stream width (slot 32 x 4
# slots = 128), so its embedding is identical to A/B/F
run bandwidth --slot 32 --suffix _slot32
echo "=== $(date -u +%H:%M:%S)  CHAIN DONE" >> "$LOG"
