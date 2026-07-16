"""Does keeping L5's two contextual heads (H5, H7) live crack the wall?
Arm A (marginal view, rest of model live): L5 tabled except H5+H7 live.
Arm B (the wall): menu-static with TRAINED codebooks, but L5 H5+H7 reverted
to live scores. Caveat on B: the trained tables were optimized under a fully
tabled L5, so this is a lower bound on what a retrained variant would get."""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens, reference_forward
from tier2_folding import scores_from_factors

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
LIVE_HEADS = [5, 7]
ZERO_L = {8, 14, 15, 17}
TAB_L = [L for L in range(1, 18) if L not in ZERO_L]
OUT = f'{QK}/l5_pair.json'
m, cfg = load_elriggs('bilin18')
NH, HD = cfg['n_head'], cfg['n_embd'] // cfg['n_head']
V = cfg['vocab_size']
AUDIT = build_eval_tokens(n_chunks=20, seq_len=513)[4:20]

raw = torch.load(f'{QK}/all17_tables.pt')
l5tabs = {n: raw[f'5_{n}'].float().to(DEV) for n in ('q1', 'k1', 'q2', 'k2')}

# rebuild the trained menu tables: assignments recomputed identically (same seeds
# as menu_trained.py), trained codebook values from menu_cbs_trained.pt
cbs = torch.load(f'{QK}/menu_cbs_trained.pt')
menu_tabs = {}
K = 256
for L in TAB_L:
    tabs = {n: raw[f'{L}_{n}'].float() for n in ('q1', 'k1', 'q2', 'k2')}
    for br, (qn, kn) in enumerate((('q1', 'k1'), ('q2', 'k2'))):
        for nm in (qn, kn):
            menu_tabs.setdefault((L, nm), torch.empty(V, NH, HD))
        for h in range(NH):
            X = torch.cat([tabs[qn][:, h], tabs[kn][:, h]], 1).to(DEV)
            g = torch.Generator(); g.manual_seed(L * 100 + h * 2 + br)
            C0 = X[torch.randperm(V, generator=g)[:K].to(DEV)].clone()
            C = C0
            for _ in range(12):
                a_ = torch.empty(V, dtype=torch.long, device=DEV)
                for i in range(0, V, 4096):
                    xx = X[i:i + 4096]
                    a_[i:i + 4096] = ((xx * xx).sum(1, True) - 2 * xx @ C.T
                                      + (C * C).sum(1)[None]).argmin(1)
                Cn = torch.zeros_like(C)
                c2 = torch.zeros(K, device=DEV)
                Cn.index_add_(0, a_, X)
                c2.index_add_(0, a_, torch.ones(V, device=DEV))
                nz = c2 > 0
                C[nz] = Cn[nz] / c2[nz][:, None]
            rows = cbs[f'{L}_{h}_{br}'].to(DEV)[a_]        # trained values
            menu_tabs[(L, qn)][:, h] = rows[:, :HD].cpu()
            menu_tabs[(L, kn)][:, h] = rows[:, HD:].cpu()
del raw
print('menu tables rebuilt from trained codebooks', flush=True)


@torch.no_grad()
def audit_ce(mode):
    # mode: 'A' = only L5 patched (tabled except LIVE_HEADS);
    #        'B' = full trained menu, L5 heads in LIVE_HEADS live;
    #        'B0' = full trained menu as trained (sanity: should be ~+0.757)
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        idx = b[:, :-1]
        idx_cpu = idx.cpu()

        def patch(li, s1, s2):
            if mode == 'A':
                if li != 5:
                    return s1, s2
                n1 = scores_from_factors(l5tabs['q1'], l5tabs['k1'], idx, HD).to(s1.dtype)
                n2 = scores_from_factors(l5tabs['q2'], l5tabs['k2'], idx, HD).to(s2.dtype)
                keep = torch.tensor([h in LIVE_HEADS for h in range(NH)],
                                    device=DEV)[None, :, None, None]
                return torch.where(keep, s1, n1), torch.where(keep, s2, n2)
            if li in ZERO_L:
                return torch.zeros_like(s1), torch.zeros_like(s2)
            if li not in TAB_L:
                return s1, s2
            Fq1 = menu_tabs[(li, 'q1')][idx_cpu].to(DEV)
            Fk1 = menu_tabs[(li, 'k1')][idx_cpu].to(DEV)
            Fq2 = menu_tabs[(li, 'q2')][idx_cpu].to(DEV)
            Fk2 = menu_tabs[(li, 'k2')][idx_cpu].to(DEV)
            n1 = cs(Fq1, Fk1, s1.dtype)
            n2 = cs(Fq2, Fk2, s2.dtype)
            if mode == 'B' and li == 5:
                keep = torch.tensor([h in LIVE_HEADS for h in range(NH)],
                                    device=DEV)[None, :, None, None]
                return torch.where(keep, s1, n1), torch.where(keep, s2, n2)
            return n1, n2

        def cs(Fq, Fk, dtype):
            d = HD // 2
            T = Fq.shape[1]
            cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
            cosD = torch.einsum('if,jf->ijf', cos, cos) + torch.einsum('if,jf->ijf', sin, sin)
            sinD = torch.einsum('if,jf->ijf', sin, cos) - torch.einsum('if,jf->ijf', cos, sin)
            qa, qb = Fq[..., :d], Fq[..., d:]
            ka, kb = Fk[..., :d], Fk[..., d:]
            s = (torch.einsum('bihf,bjhf,ijf->bhij', qa, ka, cosD)
                 + torch.einsum('bihf,bjhf,ijf->bhij', qb, kb, cosD)
                 + torch.einsum('bihf,bjhf,ijf->bhij', qb, ka, sinD)
                 - torch.einsum('bihf,bjhf,ijf->bhij', qa, kb, sinD))
            return (s / HD).to(dtype)

        logits = reference_forward(m, idx, 'bf16', score_patch=patch).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


tot, n = 0.0, 0
with torch.no_grad():
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        logits = reference_forward(m, b[:, :-1], 'bf16').float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel(); n += b[:, 1:].numel()
base = tot / n
res = {'baseline_ce': base}
for mode, name in (('B0', 'menu-trained sanity (expect ~+0.757)'),
                   ('A', 'L5 tabled except H5,H7 live (rest of model live)'),
                   ('B', 'menu-trained + L5 H5,H7 live')):
    d = audit_ce(mode) - base
    res[name] = d
    print(f'{name}: dCE {d:+.4f}', flush=True)
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=2)
print('l5 pair done', flush=True)
