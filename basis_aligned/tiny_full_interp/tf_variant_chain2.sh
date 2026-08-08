#!/bin/bash
# Phase V1, second training wave, queued after the first result (slots shows
# induction at width 128 where vanilla is null, and opens the
# attention-to-attention path).  Three things this wave buys:
#
# 1. MECHANISM DECOMPOSITION.  `slots` changes FOUR things at once versus
#    vanilla: the write partition, per-slot RMSNorm, the in-loss group lasso,
#    and a NONZERO write init (vanilla zero-inits c_proj and Down).  The last is
#    a real confound -- the reduction gate proves slots(n_slots=1, lasso 0, ZERO
#    writes) is bit-exact vanilla, so a slots cell with n_slots=1 and no lasso
#    differs from vanilla ONLY by the write init.
#      B1 writeinit_only : n_slots 1, lasso 0  -> vanilla + nonzero write init
#      B2 slots_nolasso  : n_slots 4, lasso 0  -> partition + per-slot norm only
#
# 2. DEPTH-1 MATCHED NULLS for the natural-text bag-preserving swap probe.  A
#    depth-1 model cannot compose, so its swap score is the generic
#    prefix-perturbation effect that the depth-2 number must be quoted against.
#
# 3. SEEDS 1 and 2 for every variant that showed a structural difference, so no
#    structure claim rests on one run (the parent program's standing rule).
cd "$(dirname "$0")" || exit 1
source /venv/main/bin/activate
export PYTHONUNBUFFERED=1
LOG=tf_variant_train2.log
: > "$LOG"
run () {
  echo "=== $(date -u +%H:%M:%S)  $*" >> "$LOG"
  python tf_train.py cell --vocab 8192 --tok bpe --no-sweep "$@" >> "$LOG" 2>&1 \
      || echo "!!! FAILED $*" >> "$LOG"
}
# --- 1. mechanism decomposition of `slots` (depth 2, width 128, seed 0) ---
run --variant slots --depth 2 --width 128 --seed 0 --n-slots 1 \
    --group-coeff 0.0 --suffix _writeinit_only
run --variant slots --depth 2 --width 128 --seed 0 --group-coeff 0.0 \
    --suffix _nolasso
# --- 2. depth-1 matched nulls ---
for v in slots bandwidth predicate codebook shrink; do
  run --variant $v --depth 1 --width 128 --seed 0
done
# --- 3. seeds 1 and 2 ---
for s in 1 2; do
  for v in slots bandwidth predicate codebook shrink; do
    run --variant $v --depth 2 --width 128 --seed $s
  done
done
echo "=== $(date -u +%H:%M:%S)  CHAIN2 DONE" >> "$LOG"
