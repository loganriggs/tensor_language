"""Parameterized fresh-corpus builder for scale runs (single-epoch, no-memorization
protocol). Streams FineWeb sample-10BT with the GPT-2 tokenizer and writes uint16
shards of (SHARD_SEQS, 513) into corpus_fresh/ (a name that dodges the repo's
data_fineweb*.npy gitignore so shards CAN be committed and pulled by other sessions).

Doc-range bookkeeping (disjointness by construction — never overlap these):
  docs      1..404     307k-prediction audit set (historical)
  docs   1001..~9000   data_fineweb_cooc_tokens.npy (6000 seqs; train/held for all
                       multi-epoch mini-model results, incl. old held [5500:6000])
  docs  20001..45366   data_fineweb_fresh34k_tokens.npy (34,500 seqs; single-epoch
                       width-264 E-runs; fresh held = rows [33000:34500])
  docs  45367..        THIS BUILDER (start doc set below)

Usage: python qk_corpus_build.py [n_seqs] [start_doc]   (defaults 300000, 45367)
Rebuild is deterministic for a fixed dataset revision; if HuggingFace re-shards
sample-10BT the doc indexing shifts, so prefer pulling committed shards over
rebuilding when exact-match data matters.
"""
import numpy as np
import os
import sys

SEQ = 513
SHARD_SEQS = 45000                     # ~46 MB/shard, under GitHub's 100 MB limit
NSEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 300000
START_DOC = int(sys.argv[2]) if len(sys.argv) > 2 else 45367
OUT = '/workspace/tensor_language/basis_aligned/qk_mdl/corpus_fresh'
os.makedirs(OUT, exist_ok=True)

from datasets import load_dataset
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained('gpt2')
ds = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT', split='train', streaming=True)

buf, seqs, n, shard_i, total = [], [], 0, 0, 0
texts = []

def flush_texts():
    global buf
    if not texts:
        return
    for ids in tk(texts)['input_ids']:          # batched fast tokenizer
        buf.extend(ids)
        buf.append(tk.eos_token_id)
    texts.clear()

def flush_shard(final=False):
    global seqs, shard_i, total
    while len(seqs) >= SHARD_SEQS or (final and seqs):
        take = seqs[:SHARD_SEQS]
        seqs = seqs[SHARD_SEQS:]
        arr = np.array(take, dtype=np.uint16)
        assert arr.max() <= 50256
        p = f'{OUT}/shard{shard_i:02d}.npy'
        np.save(p, arr)
        total += len(arr)
        print(f'saved {p} {arr.shape} (total {total})', flush=True)
        shard_i += 1

for r in ds:
    n += 1
    if n < START_DOC:
        continue
    texts.append(r['text'])
    if len(texts) >= 256:
        flush_texts()
        while len(buf) >= SEQ and total + len(seqs) < NSEQ:
            seqs.append(buf[:SEQ])
            buf = buf[SEQ:]
        buf = buf[:SEQ * 2]  # cap carry-over if target reached
        flush_shard()
        if total + len(seqs) >= NSEQ:
            break
    if n % 5000 == 0:
        print(f'{n} docs, {total + len(seqs)} seqs', flush=True)
flush_texts()
while len(buf) >= SEQ and total + len(seqs) < NSEQ:
    seqs.append(buf[:SEQ])
    buf = buf[SEQ:]
flush_shard(final=True)
with open(f'{OUT}/MANIFEST.txt', 'w') as f:
    f.write(f'seq_len {SEQ}\nn_seqs {total}\nshard_seqs {SHARD_SEQS}\n'
            f'docs {START_DOC}..{n}\ntokenizer gpt2\ndtype uint16\n')
print(f'done: {total} seqs through doc {n}', flush=True)
