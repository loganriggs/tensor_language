#!/bin/bash
# FINAL analysis pass: wait for every training wave, then re-run tf_interp3 on
# EVERY cell in the slice with the SAME final code -- including the vanilla
# comparators.  Forced, not skip-if-present: the analysis code changed during
# the run (the live-rows content spectrum), and a comparison table whose rows
# were produced by different code versions is exactly the failure mode this
# whole file exists to prevent.
cd "$(dirname "$0")" || exit 1
source /venv/main/bin/activate
export PYTHONUNBUFFERED=1
busy () {
  for s in tf_variant_master.sh tf_variant_master2.sh tf_variant_chain2.sh \
           tf_variant_chain3.sh tf_variant_train_chain.sh tf_interp3_chain.sh; do
    pgrep -f -x "/bin/bash ./$s" > /dev/null && return 0
  done
  return 1
}
while busy; do sleep 30; done
: > tf_interp3_final.log
for f in tf_slots_*.pt tf_bandwidth_*.pt tf_predicate_*.pt tf_codebook_*.pt \
         tf_shrink_*.pt tf_vanilla_d2_w128_b8192_s0.pt \
         tf_vanilla_d2_w128_b8192_s1.pt tf_vanilla_d2_w128_b8192_s2.pt \
         tf_vanilla_d2_w256_b8192_s0.pt tf_vanilla_d1_w128_b8192_s0.pt \
         tf_vanilla_d2_w128_b8192_s0_lr*.pt; do
  [ -e "$f" ] || continue
  stem="${f%.pt}"
  echo "=== $(date -u +%H:%M:%S) $stem" >> tf_interp3_final.log
  python tf_interp3.py --stem "$stem" >> tf_interp3_final.log 2>&1 \
      || echo "!!! FAILED $stem" >> tf_interp3_final.log
done
python tf_variant_compare.py > tf_variant_compare.stdout 2>&1
echo "=== $(date -u +%H:%M:%S) FINAL INTERP3 + COMPARE DONE" >> tf_interp3_final.log
