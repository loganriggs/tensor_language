"""TICK 236: KEY-ABLATION probe (Logan's question: which window tokens form the
memory key?). For each located failure example: substitute each of the last 16 window
tokens, one at a time, with a neutral token (' one'), and measure the drop in the
model's log-probability of the true target at the prediction position. The tokens
whose substitution collapses the prediction ARE the key. Full model, no patching —
a pure causal probe of the memory's key structure."""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward
from transformers import AutoTokenizer

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
tok = AutoTokenizer.from_pretrained('gpt2')
m, cfg = load_elriggs('bilin18')
V = cfg['vocab_size']
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQS = FINEWEB[:128]
NEUTRAL = tok.encode(' one')[0]

ex = json.load(open(f'{QK}/qk_five_examples.json'))
docs = [SEQS[i].tolist() for i in range(128)]
dec_cache = [tok.decode(d) for d in docs]
out = []
for e in ex:
    ctx_tail = e['context'][-60:]
    loc = None
    for di, dtext in enumerate(dec_cache):
        if ctx_tail in dtext:
            seq = docs[di]
            acc = ''
            for p in range(len(seq) - 1):
                acc += tok.decode([seq[p]])
                if acc.endswith(ctx_tail):
                    loc = (di, p)
                    break
            break
    if loc is None:
        continue
    di, p = loc
    tgt = SEQS[di, p + 1]
    idx = SEQS[di:di + 1, :-1].clone()
    with torch.no_grad():
        lg = reference_forward(m, idx.to(DEV), 'bf16')[0, p].float()
        base_lp = float(F.log_softmax(lg, 0)[tgt])
    drops = []
    for off in range(0, 16):
        j = p - off
        if j < 0:
            break
        orig = int(idx[0, j])
        idx2 = idx.clone()
        idx2[0, j] = NEUTRAL
        with torch.no_grad():
            lg2 = reference_forward(m, idx2.to(DEV), 'bf16')[0, p].float()
            lp2 = float(F.log_softmax(lg2, 0)[tgt])
        drops.append({'offset': off, 'token': tok.decode([orig]),
                      'dlogp': round(lp2 - base_lp, 2)})
    drops.sort(key=lambda d: d['dlogp'])
    out.append({'cluster': e['cluster'], 'target': e['target'],
                'base_logp': round(base_lp, 2), 'key_tokens': drops[:6]})
    print(f"cluster {e['cluster']} target {e['target']!r} (base logp {base_lp:.2f}):",
          flush=True)
    for d in drops[:6]:
        print(f"   swap {d['token']!r} (offset {d['offset']}): dlogp {d['dlogp']:+.2f}",
              flush=True)
json.dump(out, open(f'{QK}/qk_key_ablation.json', 'w'), indent=1)
print('KEY ABLATION DONE', flush=True)
