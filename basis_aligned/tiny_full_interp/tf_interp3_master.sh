#!/bin/bash
# Wait for ALL training waves, then interpret every slice checkpoint that has no
# _interp3.json yet, then build the comparison.  EXACT-NAME pgrep throughout --
# substring pgrep self-matches and killed two runs in the parent program.
cd "$(dirname "$0")" || exit 1
source /venv/main/bin/activate
export PYTHONUNBUFFERED=1
busy () {
  for s in tf_variant_master.sh tf_variant_master2.sh tf_variant_chain2.sh \
           tf_variant_chain3.sh tf_variant_train_chain.sh; do
    pgrep -f -x "/bin/bash ./$s" > /dev/null && return 0
  done
  return 1
}
while busy; do sleep 30; done
for f in tf_slots_*.pt tf_bandwidth_*.pt tf_predicate_*.pt tf_codebook_*.pt \
         tf_shrink_*.pt tf_vanilla_d2_w128_b8192_s0_lr*.pt; do
  [ -e "$f" ] || continue
  stem="${f%.pt}"
  [ -f "${stem}_interp3.json" ] && continue
  echo "=== $(date -u +%H:%M:%S) $stem" >> tf_interp3.log
  python tf_interp3.py --stem "$stem" >> tf_interp3.log 2>&1 \
      || echo "!!! FAILED $stem" >> tf_interp3.log
done
python tf_variant_compare.py > tf_variant_compare.stdout 2>&1
echo "=== $(date -u +%H:%M:%S) ALL INTERP3 + COMPARE DONE" >> tf_interp3.log
