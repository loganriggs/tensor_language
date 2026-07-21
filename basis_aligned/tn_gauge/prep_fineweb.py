"""Stream FineWeb (the model's training distribution), tokenize with GPT-2, cache token windows
to disk for the rank-vs-data + used-subspace experiments. Target ~600 sequences of 513 tokens."""
import sys, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained('gpt2')
SEQ = 513
NSEQ = 600
OUT = '/workspace/tensor_language/data_fineweb_tokens.npy'
buf = []
seqs = []
try:
    ds = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT', split='train', streaming=True)
except Exception as e:
    print('sample-10BT failed, trying default:', repr(e)[:150], flush=True)
    ds = load_dataset('HuggingFaceFW/fineweb', split='train', streaming=True)
n = 0
for r in ds:
    ids = tk(r['text'])['input_ids']
    buf.extend(ids); buf.append(tk.eos_token_id)
    while len(buf) >= SEQ:
        seqs.append(buf[:SEQ]); buf = buf[SEQ:]
        if len(seqs) >= NSEQ:
            break
    n += 1
    if len(seqs) >= NSEQ:
        break
    if n % 200 == 0:
        print(f'{n} docs, {len(seqs)} seqs', flush=True)
arr = np.array(seqs, dtype=np.uint16)
np.save(OUT, arr)
print(f'saved {arr.shape} to {OUT} (from {n} docs)', flush=True)
