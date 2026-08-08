#!/bin/bash
# ROUND-2 wave B: the depth-1 MATCHED NULLS at a second seed.
# The natural-text swap probe has a large depth-free baseline, so every depth-2
# swap score is quoted as an EXCESS over the same variant's depth-1 cell -- but
# those depth-1 cells exist at seed 0 only, so the excess rests on one seed of
# the null.  Train seed 1 of all six.
cd "$(dirname "$0")" || exit 1
source /venv/main/bin/activate
export PYTHONUNBUFFERED=1
LOG=tf_round2_trainb.log
until grep -q "ROUND2 ALL DONE" tf_round2_train.log 2>/dev/null; do sleep 30; done
: > "$LOG"
for v in vanilla slots bandwidth predicate codebook shrink; do
  echo "=== $(date -u +%H:%M:%S)  d1 $v s1" >> "$LOG"
  python tf_train.py cell --vocab 8192 --tok bpe --no-sweep --depth 1 \
      --width 128 --seed 1 --variant $v >> "$LOG" 2>&1 || echo "!!! FAILED $v" >> "$LOG"
done
for v in vanilla slots bandwidth predicate codebook shrink; do
  python tf_interp3.py --stem tf_${v}_d1_w128_b8192_s1 >> "$LOG" 2>&1 \
      || echo "!!! INTERP FAILED $v" >> "$LOG"
done
python tf_variant_compare.py > tf_variant_compare_stdout.txt 2>&1
python tf_consolidated_table.py > /dev/null 2>&1
echo "=== $(date -u +%H:%M:%S)  ROUND2B DONE" >> "$LOG"
