"""Cross-associations on the real model (spec codebook 2 proper): SEPARATE
q-side and k-side token partitions per (head, branch) — from-role and to-role
classed independently — vs the shared-partition vq used so far. Joint
all-heads dCE at matched-ish DL."""
import json, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward, build_eval_tokens
from tier2_folding import branch_factors, scores_from_factors

torch.manual_seed(0)
DEV = 'cuda'
m, cfg = load_elriggs('bilin18')
NH, HD = cfg['n_head'], cfg['n_embd'] // cfg['n_head']
V = cfg['vocab_size']
AUDIT = build_eval_tokens(n_chunks=20, seq_len=513)[4:]
FACT = {br: branch_factors(m, br, dtype=torch.float32) for br in (1, 2)}

def kmeans(X, k, iters=12, seed=0):
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:k]].clone()
    for _ in range(iters):
        assign = torch.empty(len(X), dtype=torch.long, device=X.device)
        for i in range(0, len(X), 4096):
            xx = X[i:i + 4096]
            assign[i:i + 4096] = ((xx**2).sum(1, keepdim=True) - 2*xx@C.T + (C**2).sum(1)[None]).argmin(1)
        Cn = torch.zeros_like(C); cnt = torch.zeros(k, device=X.device)
        Cn.index_add_(0, assign, X); cnt.index_add_(0, assign, torch.ones(len(X), device=X.device))
        C = torch.where((cnt == 0)[:, None], C, Cn / cnt.clamp(min=1)[:, None])
    return C, assign

@torch.no_grad()
def ce(qk_fact=None, batch=4):
    tot, n = 0.0, 0
    st = {'tokens': None}
    def wrap(li, s1, s2):
        if li != 0: return s1, s2
        s1n = scores_from_factors(*qk_fact[1], st['tokens'], HD).to(s1.dtype)
        s2n = scores_from_factors(*qk_fact[2], st['tokens'], HD).to(s2.dtype)
        return s1n, s2n
    for i in range(0, len(AUDIT), batch):
        b = AUDIT[i:i + batch].to(DEV)
        st['tokens'] = b[:, :-1]
        logits = reference_forward(m, b[:, :-1], score_patch=wrap if qk_fact else None).float()
        tot += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1)).item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n

CE0 = ce()
print(f'baseline {CE0:.4f}')
results = {'baseline_ce': CE0, 'arms': {}}

def build(mode, k):
    out = {}
    for br in (1, 2):
        qh, kh = FACT[br]
        qc = torch.empty_like(qh); kc = torch.empty_like(kh)
        for hh in range(NH):
            if mode == 'shared':
                C, a_ = kmeans(torch.cat([qh[:, hh], kh[:, hh]], 1), k)
                qc[:, hh], kc[:, hh] = C[a_][:, :HD], C[a_][:, HD:]
            else:  # separate from-role / to-role partitions
                Cq, aq = kmeans(qh[:, hh].contiguous(), k)
                Ck, ak = kmeans(kh[:, hh].contiguous(), k, seed=1)
                qc[:, hh], kc[:, hh] = Cq[aq], Ck[ak]
        out[br] = (qc, kc)
    return out

for mode, k in [('shared', 256), ('separate', 256), ('separate', 512),
                ('shared', 1024), ('separate', 1024)]:
    d = ce(build(mode, k)) - CE0
    # DL: shared k: k*2HD floats + V*log2k bits; separate: 2*(k*HD floats) + 2*V*log2k bits
    fl = k * 2 * HD; bits = V * (torch.log2(torch.tensor(float(k))).item()) * (1 if mode == 'shared' else 2)
    results['arms'][f'{mode} k={k}'] = {'dce': d, 'floats_per_headbranch': fl, 'index_bits_per_headbranch': bits}
    print(f'{mode:9s} k={k:5d}: dCE {d:+.4f}', flush=True)
    json.dump(results, open('cross_assoc_real.json', 'w'), indent=2)
print('cross assoc done')
