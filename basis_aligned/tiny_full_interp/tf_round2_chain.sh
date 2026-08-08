#!/bin/bash
# ROUND-2 (independent adversarial review) training arms.
#
# Each arm exists to attack a specific claim in FINDING 11.  Cells are ~5 min
# at d2 w128, so every objection that can be settled by a retrain is settled by
# a retrain rather than argued.
#
# R2-A  "the lasso prunes nothing" may be a COEFFICIENT artifact, not a fact
#       about the architecture.  Sweep the group-lasso coefficient over three
#       decades and read mean_live_slots_per_read off each.
# R2-B  the mechanism decomposition (write-init only / no-lasso) is SEED 0 ONLY.
#       Replicate both arms at seeds 1 and 2.
# R2-C  the matched-embedding (slot32) arms are SEED 0 ONLY.  Replicate at s1.
# R2-D  the learning-rate falsifier is SEED 0 ONLY at 0.01/0.04.  Add seed 1.
# R2-E  DOSE-RESPONSE: if the WRITE PARTITION is the cause, 2 slots should sit
#       between 1 slot (= vanilla + init, null) and 4 slots (full effect).
cd "$(dirname "$0")" || exit 1
source /venv/main/bin/activate
export PYTHONUNBUFFERED=1
LOG=tf_round2_train.log
: > "$LOG"
run () {
  echo "=== $(date -u +%H:%M:%S)  $*" >> "$LOG"
  python tf_train.py cell --vocab 8192 --tok bpe --no-sweep --depth 2 \
      --width 128 "$@" >> "$LOG" 2>&1 || echo "!!! FAILED $*" >> "$LOG"
}

# R2-A lasso sweep (seed 0)
for gc in 3e-4 3e-3 3e-2; do
  run --variant slots --seed 0 --group-coeff $gc --suffix _gc$gc
done
# R2-E partition dose-response
run --variant slots --seed 0 --n-slots 2 --suffix _nslots2
run --variant slots --seed 1 --n-slots 2 --suffix _nslots2
# R2-B mechanism decomposition at seeds 1,2
for s in 1 2; do
  run --variant slots --seed $s --n-slots 1 --group-coeff 0 --suffix _writeinit_only
  run --variant slots --seed $s --group-coeff 0 --suffix _nolasso
done
# R2-C matched embedding at seed 1
run --variant bandwidth --seed 1 --slot 32 --suffix _slot32
run --variant predicate --seed 1 --slot 32 --suffix _slot32
# R2-D learning-rate falsifier at seed 1
run --variant vanilla --seed 1 --lr 0.01 --suffix _lr0.01
run --variant vanilla --seed 1 --lr 0.04 --suffix _lr0.04
echo "=== $(date -u +%H:%M:%S)  ROUND2 TRAIN DONE" >> "$LOG"

# uniform analysis of every new arm through the SAME current tf_interp3
for f in tf_slots_d2_w128_b8192_s0_gc*.pt \
         tf_slots_d2_w128_b8192_s[01]_nslots2.pt \
         tf_slots_d2_w128_b8192_s[12]_writeinit_only.pt \
         tf_slots_d2_w128_b8192_s[12]_nolasso.pt \
         tf_bandwidth_d2_w128_b8192_s1_slot32.pt \
         tf_predicate_d2_w128_b8192_s1_slot32.pt \
         tf_vanilla_d2_w128_b8192_s1_lr0.01.pt \
         tf_vanilla_d2_w128_b8192_s1_lr0.04.pt; do
  [ -e "$f" ] || continue
  echo "=== interp3 $f" >> "$LOG"
  python tf_interp3.py --stem "${f%.pt}" >> "$LOG" 2>&1 || echo "!!! INTERP FAILED $f" >> "$LOG"
done
python tf_variant_compare.py > tf_variant_compare_stdout.txt 2>&1
echo "=== $(date -u +%H:%M:%S)  ROUND2 ALL DONE" >> "$LOG"
