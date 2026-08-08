#!/bin/bash
# Gated chain, pass 2 -- TRAINED-BPE PRIMARY + TRUNCATED COMPARISON ARM.
#
# Replaces tf_chain.sh (which was still idle at the GPU gate when the tokenizer
# changed, so nothing was discarded: no cell had started).  The truncated-vocab
# cells are KEPT as a deliberate arm, re-pointed from V=4096 to V=8192 so the
# only difference from the primary is the TOKENIZER, not the vocabulary size.
#
#   stage 1  tokenizer + corpora        CPU, idempotent
#   stage 2  baselines (both arms) +
#            tokenizer controls         CPU, idempotent
#   stage 3  GPU GATE                   wait for the parent program's chains
#   stage 4  positive controls          planted table, variant reductions, folds
#   stage 5  PRIMARY  bpe-8192   d1 x w{32,64,128} s0, folded immediately
#   stage 6  COMPARE  trunc-8192 d1 x w{32,64,128} s0, folded immediately
#
# Gate rules (AGENT_BRIEF): EXACT-NAME pgrep via the [a] character class so the
# pattern cannot match this script's own command line.
set -u
cd /workspace/tensor_language/basis_aligned/tiny_full_interp || exit 1
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=tf_chain2.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== tf_chain2 start (pid $$) ==="

# ---------------- stage 1: tokenizer + corpora (CPU) ----------------
if [ -f tf_bpe_8192.json ]; then
  say "stage 1a tokenizers: already trained -- skip"
else
  say "stage 1a: training byte-level BPE at 2048/4096/8192"
  python tf_tokenizer.py train >> tf_tokenizer_train.out 2>&1 \
    || { say "BPE TRAIN FAILED"; exit 1; }
fi
for V in 8192 4096; do
  if [ -f "tf_corpus_b${V}/MANIFEST.json" ]; then
    say "stage 1b corpus b${V}: already built -- skip"
  else
    say "stage 1b: building tf_corpus_b${V}"
    python tf_tokenizer.py corpus "$V" >> tf_corpus_bpe_build.out 2>&1 \
      || { say "CORPUS b${V} FAILED"; exit 1; }
  fi
done
if [ ! -f tf_corpus_v8192/MANIFEST.json ]; then
  say "stage 1c: building the truncated comparison corpus"
  python tf_corpus.py 8192 >> tf_corpus_build.out 2>&1 \
    || { say "TRUNC CORPUS FAILED"; exit 1; }
fi

# ---------------- stage 2: baselines + tokenizer controls (CPU) -----------
for SPEC in "8192 bpe b8192" "4096 bpe b4096" "8192 trunc v8192"; do
  set -- $SPEC
  if [ -f "tf_baselines_$3.json" ]; then
    say "stage 2 baselines $3: already computed -- skip"
  else
    say "stage 2 baselines $3"
    python tf_train.py baselines --vocab "$1" --tok "$2" >> tf_baselines.out 2>&1 \
      || { say "BASELINES FAILED $3 (bigram must beat unigram)"; exit 1; }
  fi
  say "stage 2 $3: $(python -c "import json;d=json.load(open('tf_baselines_$3.json'));print('uni',d['unigram_floor_bits_per_byte'],'bi',d['bigram_bits_per_byte'],'bits/byte; honest bi',d.get('bigram_bits_per_byte_honest','n/a'))")"
done
if [ ! -f tf_tokenizer_compare.json ]; then
  say "stage 2: tokenizer comparison table"
  python tf_tokenizer.py compare >> tf_tokenizer_compare.out 2>&1 \
    || { say "COMPARE FAILED"; exit 1; }
fi
say "stage 2: tokenizer controls"
python tf_tokenizer.py controls >> tf_tokenizer_controls.out 2>&1 \
  || { say "TOKENIZER CONTROLS FAILED"; exit 1; }
if ! python -c "import json,sys; sys.exit(0 if json.load(open('tf_tokenizer_controls.json'))['ALL_PASS'] else 1)"; then
  say "TOKENIZER CONTROLS NOT ALL_PASS -- refusing to train"; exit 1
fi

# ---------------- stage 3: GPU gate ----------------
say "stage 3 gate: waiting for the parent program's chains to finish"
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

# ---------------- stages 5/6: cells + immediate folds ----------------
run_cells() {   # $1 = tok, $2 = vocab, $3 = stem letter
  for W in 32 64 128; do
    STEM="tf_vanilla_d1_w${W}_${3}${2}_s0"
    say "training ${STEM} (tok=$1)"
    python tf_train.py cell --variant vanilla --depth 1 --width "$W" --seed 0 \
      --vocab "$2" --tok "$1" >> "tf_${STEM}.out" 2>&1 \
      || { say "TRAIN FAILED ${STEM}"; continue; }
    say "folding ${STEM} immediately"
    python tf_fold.py --stem "$STEM" --deltas 0,1,2 --direct-svd \
      >> "tf_${STEM}_fold.out" 2>&1 \
      || { say "FOLD GATE FAILED ${STEM}"; continue; }
    say "${STEM} done -- $(python -c "import json;d=json.load(open('${STEM}.json'));v=d.get('vs_baselines',{});print('held CE',d['run']['final_held_ce'],'|',v.get('model_bits_per_byte'),'bits/byte | vs bigram',v.get('model_minus_bigram'),'| fold gate',d['fold']['identity_gate']['pass'])")"
  done
}

say "=== stage 5: PRIMARY arm, trained byte-level BPE V=8192 (zero UNK) ==="
run_cells bpe 8192 b
say "=== stage 6: COMPARISON arm, truncated GPT-2 V=8192 (13.2% UNK) ==="
run_cells trunc 8192 v

say "=== tf_chain2 done ==="
