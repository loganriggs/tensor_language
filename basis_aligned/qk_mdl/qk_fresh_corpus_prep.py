"""Fresh single-epoch corpus for the interpretable-architecture experiments (Logan's
memorization concern, 2026-08-03): the width-264/384 families train 6 epochs over 5500
sequences (8250 steps x batch 4 = 33,000 visits), so train-CE gaps partly reflect
memorization capacity. This builds a corpus large enough that every training step sees a
never-before-seen sequence.

Stream FineWeb sample-10BT with the GPT-2 tokenizer (same recipe as qk_cooc_prep.py),
SKIP the first 20,000 docs — the existing corpora (audit set: first 404 docs; cooc corpus:
docs 1001 through ~9,000; fw eval: first docs) all live far below that, so disjointness
holds by construction — then collect 34,500 sequences of 513 tokens:
  train = [0:33000]  (one epoch at 8250 steps x batch 4, zero repeats)
  held  = [33000:34500]  (fresh held set; the old cooc held [5500:6000] stays usable for
                          cross-family comparability since no new model trains on it)
Save uint16 -> /workspace/tensor_language/data_fineweb_fresh34k_tokens.npy (~35 MB).
"""
import numpy as np
import os

SEQ, NSEQ, SKIP_DOCS = 513, 34500, 20000
CORPUS = '/workspace/tensor_language/data_fineweb_fresh34k_tokens.npy'

if os.path.exists(CORPUS):
    print(f'{CORPUS} already exists: {np.load(CORPUS, mmap_mode="r").shape}')
    raise SystemExit
from datasets import load_dataset
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained('gpt2')
ds = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT', split='train', streaming=True)
buf, seqs, n = [], [], 0
for r in ds:
    n += 1
    if n <= SKIP_DOCS:
        continue
    ids = tk(r['text'])['input_ids']
    buf.extend(ids)
    buf.append(tk.eos_token_id)
    while len(buf) >= SEQ and len(seqs) < NSEQ:
        seqs.append(buf[:SEQ])
        buf = buf[SEQ:]
    if len(seqs) >= NSEQ:
        break
    if n % 2000 == 0:
        print(f'{n} docs, {len(seqs)} seqs', flush=True)
arr = np.array(seqs, dtype=np.uint16)
assert arr.max() <= 50256
np.save(CORPUS, arr)
print(f'saved {arr.shape} to {CORPUS} (docs {SKIP_DOCS + 1}..{n})', flush=True)
