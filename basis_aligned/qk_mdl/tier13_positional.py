"""Tier 1.3: positional-head sweep + tiny-model joint-svd frontier.

Positional codebook = replace a (layer, head, branch)'s scores by their per-Δ
mean (token structure destroyed, Δ-profile kept). A head-branch is
'behaviorally positional' if this costs |ΔCE| ≤ 0.01 on val text (and, for the
rp model, |ΔP(copy)| ≤ 0.02 on tiled sequences). DL collapse for a positional
branch: 2·V·2F floats → (2T−1) Δ-values (or fewer Fourier modes; accounted at
full Δ-table here — conservative).

Models: attn2-mix10-seed0 (completes its tier-1.1 table with the missing
positional codebook) and attn2-s30k-mix50-rp-dense-seed0 (both layers — the
copy circuit's positional substrate). Joint audits: all positional-classified
branches replaced simultaneously. Plus the tick-4 leftover: mix10 joint svd-r
frontier.
"""

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from folding import load_tiny, branch_factors

torch.manual_seed(0)
DEV = 'cuda'
OUT = '/workspace/tensor_language/basis_aligned/qk_mdl/tier13_positional.json'
val = np.memmap('/workspace/tensor_language/data_owt/val.bin', dtype=np.uint16, mode='r')


def val_chunks(T, n=64, seed=0):
    g = torch.Generator(); g.manual_seed(seed)
    starts = torch.randint(0, len(val) - (T + 1), (n,), generator=g)
    return torch.stack([torch.tensor(val[int(s):int(s) + T + 1].astype(np.int64))
                        for s in starts]).to(DEV)


class Probe:
    def __init__(self, run):
        self.model, self.cfg = load_tiny(run, dtype=torch.float32, device=DEV)
        self.NH = self.cfg['n_head']
        self.DH = self.cfg['d_model'] // self.NH
        self.V = self.cfg['vocab']
        self.T = self.cfg['n_ctx']

    @torch.no_grad()
    def forward(self, tokens, posavg=(), factor_patch=None):
        """posavg: iterable of (layer, head, branch) to per-Δ-average.
        factor_patch: dict (layer, head, branch) -> (q_stack, k_stack) (V, 2F)."""
        m, NH, DH = self.model, self.NH, self.DH
        x = m.embed(tokens)
        Tn = tokens.shape[-1]
        for li, layer in enumerate(m.layers):
            h = layer.norm(x)
            hs = lambda t: t.reshape(*t.shape[:-1], NH, DH)
            s = {}
            for br, (wq, wk) in {1: (layer.q1, layer.k1), 2: (layer.q2, layer.k2)}.items():
                q = layer.rotary(hs(wq(h)))
                k = layer.rotary(hs(wk(h)))
                s[br] = torch.einsum('bihd,bjhd->bhij', q, k)
            for (pl, ph, pb) in posavg:
                if pl != li:
                    continue
                sc = s[pb][:, ph]
                d_idx = (torch.arange(Tn, device=DEV)[:, None]
                         - torch.arange(Tn, device=DEV)[None, :]).clamp(min=0)
                sums = torch.zeros(Tn, device=DEV).index_add_(
                    0, d_idx.flatten(), sc.mean(0).flatten())
                cnts = torch.zeros(Tn, device=DEV).index_add_(
                    0, d_idx.flatten(), torch.ones(Tn * Tn, device=DEV))
                s[pb] = s[pb].clone()
                s[pb][:, ph] = (sums / cnts.clamp(min=1))[d_idx][None]
            if factor_patch:
                for (pl, ph, pb), (qs, ks) in factor_patch.items():
                    if pl != li:
                        continue
                    FB = DH // 2
                    cs = layer.rotary.cos_cached[0, :Tn, 0, :FB]
                    sn = layer.rotary.sin_cached[0, :Tn, 0, :FB]
                    cosD = torch.einsum('if,jf->ijf', cs, cs) + torch.einsum('if,jf->ijf', sn, sn)
                    sinD = torch.einsum('if,jf->ijf', sn, cs) - torch.einsum('if,jf->ijf', cs, sn)
                    qa, qb = qs[tokens][..., :FB], qs[tokens][..., FB:]
                    ka, kb = ks[tokens][..., :FB], ks[tokens][..., FB:]
                    sc = (torch.einsum('bif,bjf,ijf->bij', qa, ka, cosD)
                          + torch.einsum('bif,bjf,ijf->bij', qb, kb, cosD)
                          + torch.einsum('bif,bjf,ijf->bij', qa, kb, sinD)
                          - torch.einsum('bif,bjf,ijf->bij', qb, ka, sinD))
                    s[pb] = s[pb].clone()
                    s[pb][:, ph] = sc
            mask = torch.tril(torch.ones(Tn, Tn, device=DEV, dtype=x.dtype))
            pat = s[1] * s[2] / DH ** 2 * mask
            v = hs(layer.v(layer.norm(x)))
            z = torch.einsum('bhij,bjhd->bihd', pat, v).reshape(*tokens.shape, -1)
            x = torch.lerp(x, layer.o(z), layer.scale)
        return m.head(x)

    @torch.no_grad()
    def ce(self, tokens, batch=16, **kw):
        tot, n = 0.0, 0
        for i in range(0, len(tokens), batch):
            bt = tokens[i:i + batch]
            logits = self.forward(bt[:, :-1], **kw).float()
            tot += F.cross_entropy(logits.reshape(-1, self.V),
                                   bt[:, 1:].reshape(-1)).item() * bt[:, 1:].numel()
            n += bt[:, 1:].numel()
        return tot / n


results = {'models': {}}

# =============== positional sweep ===============
for run, check_copy in [('attn2-mix10-seed0', False),
                        ('attn2-s30k-mix50-rp-dense-seed0', True)]:
    pr = Probe(run)
    EV = val_chunks(pr.T)
    ce0 = pr.ce(EV)
    entry = {'baseline_ce': ce0, 'branches': {}, 'joint': {}}
    if check_copy:
        P = 96
        g = torch.Generator(); g.manual_seed(0)
        w = torch.randint(pr.V, (64, P), generator=g)
        tiled = w.repeat(1, (pr.T + 1 + P - 1) // P)[:, :pr.T + 1].to(DEV)
        XT, YT = tiled[:, :-1], tiled[:, 1:]
        qpos = torch.arange(P + 2, pr.T - 2, device=DEV)

        def pcopy(**kw):
            p = torch.softmax(pr.forward(XT, **kw).float(), -1)
            return float(p[:, qpos, :].gather(2, YT[:, qpos].unsqueeze(-1)).mean())
        pc0 = pcopy()
        entry['baseline_pcopy'] = pc0

    positional = []
    for li in range(len(pr.model.layers)):
        for hh in range(pr.NH):
            for br in (1, 2):
                key = f'L{li}H{hh}b{br}'
                dce = pr.ce(EV, posavg=[(li, hh, br)]) - ce0
                rec = {'dce_posavg': dce}
                if check_copy:
                    rec['dpcopy_posavg'] = pcopy(posavg=[(li, hh, br)]) - pc0
                ok = abs(dce) <= 0.01 and (not check_copy or abs(rec['dpcopy_posavg']) <= 0.02)
                rec['positional'] = ok
                if ok:
                    positional.append((li, hh, br))
                entry['branches'][key] = rec
                print(f"{run} {key}: dCE {dce:+.4f}"
                      + (f"  dPcopy {rec['dpcopy_posavg']:+.4f}" if check_copy else '')
                      + ('  -> POSITIONAL' if ok else ''), flush=True)
    if positional:
        djoint = pr.ce(EV, posavg=positional) - ce0
        entry['joint']['all_positional'] = {
            'n_branches': len(positional), 'dce': djoint}
        if check_copy:
            entry['joint']['all_positional']['dpcopy'] = pcopy(posavg=positional) - pc0
        full = 32 * 2 * pr.V * pr.DH
        pos_dl = 32 * (2 * pr.T - 1)
        entry['joint']['all_positional']['dl_collapse_per_branch'] = pos_dl / full
        print(f"{run} JOINT ({len(positional)} positional branches): dCE {djoint:+.4f}"
              + (f"  dPcopy {entry['joint']['all_positional'].get('dpcopy', 0):+.4f}"
                 if check_copy else ''), flush=True)
    results['models'][run] = entry

# =============== mix10 joint svd-r frontier (tick-4 leftover) ===============
pr = Probe('attn2-mix10-seed0')
EV = val_chunks(pr.T)
ce0 = pr.ce(EV)
FACT = {}
for br in (1, 2):
    qa, qb, ka, kb = branch_factors(pr.model, 0, br)
    for hh in range(pr.NH):
        FACT[(hh, br)] = (torch.cat([qa[:, hh], qb[:, hh]], 1).float().contiguous(),
                          torch.cat([ka[:, hh], kb[:, hh]], 1).float().contiguous())
frontier = {}
for r in [1, 2, 4, 8, 16]:
    patch = {}
    for (hh, br), (q, k) in FACT.items():
        def tr(Xm, r=r):
            U, S, Vt = torch.linalg.svd(Xm, full_matrices=False)
            return (U[:, :r] @ torch.diag(S[:r]) @ Vt[:r]).contiguous()
        patch[(0, hh, br)] = (tr(q), tr(k))
    d = pr.ce(EV, factor_patch=patch) - ce0
    dl_ratio = r * (pr.V + 2 * (pr.DH // 2) + 1) / (pr.V * pr.DH)
    frontier[f'joint svd{r}'] = {'dce': d, 'dl_ratio': dl_ratio}
    print(f'mix10 joint svd{r} (layer 0, all 8 branches): dCE {d:+.4f}  '
          f'DL ratio {dl_ratio:.3f}', flush=True)
results['mix10_joint_svd_frontier'] = frontier

with open(OUT, 'w') as fh:
    json.dump(results, fh, indent=2)
print('tier13 done')
