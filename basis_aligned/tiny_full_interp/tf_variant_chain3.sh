#!/bin/bash
# Third wave: the two objections a reviewer will raise about the headline.
#
# A. "It is the learning rate."  Every arm runs Muon 0.02, matched to the
#    vanilla checkpoint, but 0.02 was nobody's swept optimum.  If the plain
#    model produced induction at 0.01 or 0.04, the slot result would be an lr
#    effect wearing an architecture costume.  So both architectures are run at
#    all three grid points and the induction score is read off each.
# B. "It is the extra embedding."  bandwidth/predicate/codebook reinvest the
#    small decoder's savings into a 160-wide stream, which is 262,144 more
#    embedding parameters than vanilla/slots/shrink.  `--slot 32` pins the
#    stream back to 128 so the embedding is IDENTICAL, isolating the mechanism.
cd "$(dirname "$0")" || exit 1
source /venv/main/bin/activate
export PYTHONUNBUFFERED=1
LOG=tf_variant_train3.log
: > "$LOG"
run () {
  echo "=== $(date -u +%H:%M:%S)  $*" >> "$LOG"
  python tf_train.py cell --vocab 8192 --tok bpe --no-sweep --depth 2 \
      --width 128 --seed 0 "$@" >> "$LOG" 2>&1 || echo "!!! FAILED $*" >> "$LOG"
}
for lr in 0.01 0.04; do
  run --variant vanilla --lr $lr --suffix _lr$lr
  run --variant slots   --lr $lr --suffix _lr$lr
done
run --variant predicate --slot 32 --suffix _slot32
echo "=== $(date -u +%H:%M:%S)  CHAIN3 DONE" >> "$LOG"
