"""Extract CONCRETE dataset examples for the §76 distributed class priors (Logan's request):
for mlp.L17.d0 (capital-prior), h.L14.h4 (word-prior), h.L11.h3 (subword-prior) — find their top firing
positions on held-back FW[448:600], decode the surrounding sentence context + actual next token, and list
the top individual tokens their mean delta-logit boosts inside the pushed class. Forward/gram/activation
conventions copied VERBATIM from qk_extend_coverage.py. Small footprint (batch 2, coexists with training)."""
import json, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
# GPU guard (coexist with SAE training)
import subprocess
free = int(subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
                          capture_output=True,text=True).stdout.strip().split('\n')[0])
assert free > 4000, f"only {free} MiB free"
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
from transformers import GPT2TokenizerFast
tok = GPT2TokenizerFast.from_pretrained('gpt2')
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
TRAIN = FW[0:256, :128].to(DEV); HELD = FW[448:600, :128].to(DEV)
BATCH = 2; N_SVD = 4
TARGETS = [('mlp', 17, 0, 'capital'), ('head', 14, 4, 'word'), ('head', 11, 3, 'subword')]

# --- MLP dir for L17 from TRAIN gram (VERBATIM construction) ---
gram17 = torch.zeros(D, D, device=DEV)
@torch.no_grad()
def fwd_pass(idx, want_gram=False, collect=False):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None,:,None,:], sin[None,:,None,:]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    acts = {}
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        if collect:
            for (kind, tli, ix, _c) in TARGETS:
                if kind == 'head' and tli == li:
                    # head-contribution norm through c_proj (VERBATIM census style)
                    Wr = a.c_proj.weight.T.view(NH, HD, D)[ix]          # (HD,D)
                    acts[('head', li, ix)] = torch.einsum('bthc,cd->btd', yh4[:, :, ix:ix+1].squeeze(2).unsqueeze(2), Wr.unsqueeze(0).unsqueeze(0)).squeeze(2).norm(dim=-1) if False else (yh4[:, :, ix] @ Wr).norm(dim=-1)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); mo = blk.mlp(F.rms_norm(x, (D,)))
        if want_gram and li == 17: gram17.add_(torch.einsum('btd,bte->de', mo, mo))
        if collect and li == 17:
            acts[('mlp17',)] = mo
        x = x + mo
    return acts
print("gram pass ...", flush=True)
for i in range(0, TRAIN.shape[0], BATCH): fwd_pass(TRAIN[i:i+BATCH], want_gram=True)
_ev, _evec = torch.linalg.eigh(gram17)
d17 = _evec[:, -N_SVD:].T.flip(0)[0]                                    # dir 0
print("collect pass on HELD ...", flush=True)
A = {t: [] for t in [('mlp',17,0), ('head',14,4), ('head',11,3)]}
for i in range(0, HELD.shape[0], BATCH):
    acts = fwd_pass(HELD[i:i+BATCH], collect=True)
    A[('mlp',17,0)].append((acts[('mlp17',)] @ d17).abs().cpu())
    A[('head',14,4)].append(acts[('head',14,4)].cpu())
    A[('head',11,3)].append(acts[('head',11,3)].cpu())
res = {}
for key, name, pushed in [(('mlp',17,0),'mlp.L17.d0','capital'), (('head',14,4),'h.L14.h4','word'), (('head',11,3),'h.L11.h3','subword')]:
    act = torch.cat(A[key], 0)                                          # (S,T)
    S, T = act.shape
    flat = act.flatten(); top = flat.topk(10).indices
    examples = []
    for p in top.tolist():
        s, t = p // T, p % T
        ids = HELD[s].tolist()
        ctx = tok.decode(ids[max(0, t-24):t+1])
        cur = tok.decode([ids[t]]); nxt = tok.decode([ids[t+1]]) if t+1 < T else '?'
        examples.append({'seq': s, 'pos': t, 'context': ctx, 'fires_on': cur, 'actual_next': nxt,
                         'act': round(float(act[s, t]), 1)})
    res[name] = {'pushed_class': pushed, 'examples': examples}
    print(f"\n===== {name} (pushes the {pushed.upper()} class) — top firing positions =====")
    for e in examples[:6]:
        print(f"  [act {e['act']}] ...{e['context']!r}  ||fires on|| {e['fires_on']!r} -> next {e['actual_next']!r}")
json.dump(res, open(f'{QK}/qk_prior_examples.json', 'w'), indent=1)
print("\nDONE", flush=True)
