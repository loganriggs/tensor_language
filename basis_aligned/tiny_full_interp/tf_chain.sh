#!/bin/bash
# Gated chain for the tiny-full-interpretation program, first pass.
#
#   stage 1  corpus build            CPU, idempotent, runs immediately
#   stage 2  bigram/unigram baselines CPU, idempotent, runs immediately
#   stage 3  GPU GATE                 wait for the parent program's qk_e34 chain
#   stage 4  positive controls        planted table test, variant reductions,
#                                     fold identity gate for all six variants
#   stage 5  the three depth-1 cells local owns (widths 32, 64, 128) at seed 0,
#            FOLDING EACH IMMEDIATELY after training so we learn early whether
#            rungs 1-2 pass at every width.  One seed only: this pass validates
#            the pipeline end to end; seeds 1-2 come after.
#
# Gate rules (AGENT_BRIEF): EXACT-NAME pgrep -- the [a] character-class trick
# stops the pattern from matching this script's own command line, which bit the
# parent program twice.  Then >= 10000 MiB free, measured 3 times 60s apart.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=tf_chain.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== tf_chain start (pid $$) ==="

# ---------------- stage 1: corpus (CPU) ----------------
if [ -f tf_corpus_v4096/MANIFEST.json ]; then
  say "stage 1 corpus: already built -- skip"
else
  say "stage 1 corpus: building"
  python tf_corpus.py >> tf_corpus_build.out 2>&1 || { say "CORPUS FAILED"; exit 1; }
fi

# ---------------- stage 2: baselines (CPU) ----------------
if [ -f tf_baselines_v4096.json ]; then
  say "stage 2 baselines: already computed -- skip"
else
  say "stage 2 baselines: unigram floor + closed-form bigram"
  python tf_train.py baselines --vocab 4096 >> tf_baselines.out 2>&1 \
    || { say "BASELINES FAILED (bigram must beat unigram)"; exit 1; }
fi
say "stage 2 result: $(python -c "import json;d=json.load(open('tf_baselines_v4096.json'));print('unigram',d['unigram_floor_held_ce'],'bigram',d['bigram_held_ce'],'gap',d['bigram_beats_unigram_nats'])")"

# ---------------- stage 3: GPU gate ----------------
say "stage 3 gate: waiting for the parent qk_e34 chain to finish"
while true; do
  if pgrep -f -- 'qk_e34_[a]blate_run\.py' > /dev/null \
     || pgrep -f -- 'qk_e34_[c]hain\.sh' > /dev/null; then
    sleep 120
    continue
  fi
  ok=0
  for i in 1 2 3; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    say "gate probe $i/3: ${free} MiB free"
    if [ "$free" -ge 10000 ]; then ok=$((ok+1)); else ok=0; break; fi
    [ "$i" -lt 3 ] && sleep 60
  done
  [ "$ok" -ge 3 ] && break
  sleep 120
done
say "stage 3 gate: GPU free -- proceeding"

# ---------------- stage 4: positive controls ----------------
say "stage 4 controls: planted table test + variant reductions + fold gates"
python tf_model.py > tf_controls.out 2>&1 || { say "CONTROLS FAILED"; exit 1; }
grep -E "planted|reduction|pass=" tf_controls.out | tee -a "$LOG"
if grep -q "pass=False" tf_controls.out || grep -q "'pass': False" tf_controls.out; then
  say "CONTROLS FAILED -- refusing to train"; exit 1
fi

# ---------------- stage 5: cells + immediate folds ----------------
for W in 32 64 128; do
  STEM="tf_vanilla_d1_w${W}_v4096_s0"
  say "stage 5: training ${STEM}"
  python tf_train.py cell --variant vanilla --depth 1 --width "$W" --seed 0 \
    >> "tf_${STEM}.out" 2>&1 || { say "TRAIN FAILED ${STEM}"; continue; }
  say "stage 5: folding ${STEM} immediately"
  python tf_fold.py --stem "$STEM" --deltas 0,1,2 --direct-svd \
    >> "tf_${STEM}_fold.out" 2>&1 || { say "FOLD GATE FAILED ${STEM}"; continue; }
  say "stage 5: ${STEM} done -- $(python -c "import json;d=json.load(open('${STEM}.json'));print('held CE',d['run']['final_held_ce'],'| vs bigram',d.get('vs_baselines',{}).get('model_minus_bigram'),'| fold gate',d['fold']['identity_gate']['pass'])")"
done

say "=== tf_chain done ==="
