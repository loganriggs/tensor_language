"""Co-occurrence prep for the context-objective refinements (tick 162).

1. Stream FineWeb sample-10BT, SKIP the first 1000 docs (the 307k-prediction audit set was
   built from the first 404 docs of the same stream — disjointness by construction), collect
   6000 sequences of 513 tokens -> /workspace/tensor_language/data_fineweb_cooc_tokens.npy.
2. Cluster the unit-RMS embedding rows (V=50304, D=1152) into C=256 clusters (k-means).
3. Count causal within-window cluster pairs (query cluster a at position i, key cluster b at
   position j<i) over the corpus; lift L(a,b) = P(b|a)/P(b) with +5 smoothing.
4. Save qk_cooc_lift.pt: {assign (V,) int16, lift (256,256) float32, counts diagnostics}.
"""
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')

SEQ, NSEQ, SKIP_DOCS, C = 513, 6000, 1000, 256
CORPUS = '/workspace/tensor_language/data_fineweb_cooc_tokens.npy'
OUT = '/workspace/tensor_language/basis_aligned/qk_mdl/qk_cooc_lift.pt'
DEV = 'cuda'

if not os.path.exists(CORPUS):
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
        if n % 500 == 0:
            print(f'{n} docs, {len(seqs)} seqs', flush=True)
    arr = np.array(seqs, dtype=np.uint16)
    np.save(CORPUS, arr)
    print(f'saved {arr.shape} to {CORPUS} (through doc {n}, skipped first {SKIP_DOCS})', flush=True)

from tier2_model import load_elriggs
from qk_sae_lib import kmeans

m, cfg = load_elriggs('bilin18')
D, V = cfg['n_embd'], cfg['vocab_size']
E = F.rms_norm(m.transformer.wte.weight.detach().float(), (D,)).to(DEV)
assign, _ = kmeans(E, C, iters=25, seed=0)
sizes = torch.bincount(assign, minlength=C)
print(f'kmeans done: cluster sizes min {sizes.min().item()} med {sizes.median().item()} '
      f'max {sizes.max().item()}', flush=True)

toks = torch.from_numpy(np.load(CORPUS).astype(np.int64))
N = torch.zeros(C * C, dtype=torch.float64, device=DEV)
ii, jj = torch.tril_indices(SEQ, SEQ, offset=-1)          # all causal pairs j < i
ii, jj = ii.to(DEV), jj.to(DEV)
for s0 in range(0, len(toks), 50):
    batch = toks[s0:s0 + 50].to(DEV)
    cl = assign[batch]                                     # (B, SEQ)
    pair = cl[:, ii] * C + cl[:, jj]                       # (B, npairs)
    N += torch.bincount(pair.flatten(), minlength=C * C).double()
    if s0 % 1000 == 0:
        print(f'  counted {s0 + len(batch)}/{len(toks)} seqs', flush=True)
N = N.view(C, C)
alpha = 5.0
Pba = (N + alpha) / (N + alpha).sum(1, keepdim=True)       # P(key cluster b | query cluster a)
Pb = (N + alpha).sum(0) / (N + alpha).sum()                # P(key cluster b)
lift = (Pba / Pb[None, :]).float()
print(f'pairs counted {N.sum().item():.3g}; lift range [{lift.min().item():.3f}, '
      f'{lift.max().item():.3f}], median {lift.median().item():.3f}', flush=True)
torch.save({'assign': assign.to(torch.int16).cpu(), 'lift': lift.cpu(),
            'cluster_sizes': sizes.cpu(), 'n_pairs': float(N.sum().item())}, OUT)
print(f'saved {OUT}', flush=True)
