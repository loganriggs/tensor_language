"""PARETO SWEEP (tick 160, Logan: in-depth overnight run): dictionary frontier over (n, k)
budgets x {plain-MSE, OV-context-trained} objectives x seeds, all audited on FineWeb (307k
held-out predictions, the standard set).

Jobs (priority-ordered so partial completion is still useful; results written incrementally):
  Phase A (seed 0): (n,k) in (256,4),(512,4),(1024,4),(1024,8),(2048,8),(4096,8),(8192,8),(4096,16)
           per config: MSE fit (18 head-branches) -> audit linear + OMP encoders;
           OV-context finetune (9 heads, both branches jointly, eq. dagger objective) -> audit.
  Phase B (seeds 1,2): (512,4),(1024,8),(4096,8) — same minus the OMP audit.

Bits: dl_sparse_dict per head-branch, identical for both objectives at a config (the OV finetune
changes no structure). Each job is wrapped in try/except; a failure logs and moves on.
Writes qk_pareto_sweep.json (resumable at job granularity).
"""
import json
import os
import sys
import traceback
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward
from tier2_folding import branch_factors, scores_from_factors
from mdl_accounting import dl_sparse_dict
from qk_sae_lib import train_dict, encode_token, encode_omp, fvu

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_pareto_sweep.json'
NAMES = ('q1', 'k1', 'q2', 'k2')
BRANCHES = (('q1', 'k1'), ('q2', 'k2'))
PHASE_A = [(256, 4), (512, 4), (1024, 4), (1024, 8), (2048, 8), (4096, 8), (8192, 8), (4096, 16)]
PHASE_B = [(512, 4), (1024, 8), (4096, 8)]
JOBS = [(n, k, 0) for (n, k) in PHASE_A] + [(n, k, s) for s in (1, 2) for (n, k) in PHASE_B]
CTX_STEPS, CTX_M, CTX_LR, T_CTX = 1500, 1024, 3e-4, 512.0

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
NHB, ROW = NH * 2, 2 * HD
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))

TAB = {}
for br, (qn, kn) in enumerate(BRANCHES, start=1):
    qh, kh = branch_factors(m, br)
    TAB[qn], TAB[kn] = qh.float().to(DEV), kh.float().to(DEV)
HB = [(h, qn, kn) for h in range(NH) for (qn, kn) in BRANCHES]

with torch.no_grad():
    a = m.transformer.h[0].attn
    E = F.rms_norm(m.transformer.wte.weight.detach().float(), (D,))
    Vv = a.c_v(E).view(V, NH, HD)
    Wo = a.c_proj.weight.detach().float().view(D, NH, HD)
QFULL = (torch.bincount(FINEWEB.flatten(), minlength=V).float() + 0.5).to(DEV)


def rows(h, qn, kn):
    return torch.cat([TAB[qn][:, h], TAB[kn][:, h]], 1)


def unit_rms(t):
    return t * (t.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt())


def tables_from(recs):
    out = {n: TAB[n].clone() for n in NAMES}
    for (h, qn, kn), rec in zip(HB, recs):
        out[qn][:, h] = rec[:, :HD]
        out[kn][:, h] = rec[:, HD:]
    return {n: unit_rms(out[n]) for n in NAMES}


@torch.no_grad()
def audit_fw(tabs, batch=4):
    tot, n = 0.0, 0
    for i in range(0, len(FINEWEB), batch):
        b = FINEWEB[i:i + batch].to(DEV)
        idx = b[:, :-1]

        def patch(li, s1, s2):
            if li != 0:
                return s1, s2
            n1 = scores_from_factors(tabs['q1'], tabs['k1'], idx, HD)
            n2 = scores_from_factors(tabs['q2'], tabs['k2'], idx, HD)
            return n1.to(s1.dtype), n2.to(s2.dtype)

        logits = reference_forward(m, idx, 'bf16', score_patch=None if tabs is None else patch).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


def ctx_finetune(fits, n, k, seed):
    """OV-context finetune (qk_ctx_train recipe): per head, both branch dicts jointly."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    out_fits = list(fits)
    losses = []
    for h in range(NH):
        parts, params = {}, []
        for br in (0, 1):
            Dn0, b0, We0 = fits[h * 2 + br]
            Dm = Dn0.clone().requires_grad_(True)
            We = We0.clone().requires_grad_(True)
            b = b0.clone().requires_grad_(True)
            parts[br] = (Dm, We, b)
            params += [Dm, We, b]
        opt = torch.optim.Adam(params, lr=CTX_LR)
        Uh = Vv[:, h] @ Wo[:, h].T
        first = last = None
        for step in range(CTX_STEPS):
            sample = torch.randperm(V, generator=g)[:CTX_M].to(DEV)
            qs = QFULL[sample]
            qs = qs / qs.sum()
            Us = Uh[sample]
            w2 = Us.pow(2).sum(1)
            Shat, S = [], []
            for br, (qn, kn) in enumerate(BRANCHES):
                X = torch.cat([TAB[qn][sample, h], TAB[kn][sample, h]], 1)
                Dm, We, b = parts[br]
                Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
                z = (X - b) @ We.T
                vals, idx = z.abs().topk(k, dim=1)
                coeff = torch.gather(z, 1, idx)
                rec = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
                Shat.append(unit_rms(rec[:, :HD]) @ unit_rms(rec[:, HD:]).T / HD)
                S.append(X[:, :HD] @ X[:, HD:].T / HD)
            P = S[0] * S[1]
            dP = Shat[0] * Shat[1] - P

            def ctx(mat):
                mu = (mat * qs[None, :]) @ Us
                mu2 = mu.pow(2).sum(1)
                s_ = (mat.pow(2) * (qs * w2)[None, :]).sum(1)
                return (qs * (T_CTX * (s_ - mu2).clamp_min(0) + T_CTX * T_CTX * mu2)).sum()

            with torch.no_grad():
                den = ctx(P).clamp_min(1e-12)
            loss = ctx(dP) / den
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if step == 0:
                first = loss.item()
            last = loss.item()
        losses.append((round(first, 4), round(last, 4)))
        for br in (0, 1):
            Dm, We, b = parts[br]
            out_fits[h * 2 + br] = ((Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach(),
                                    b.detach(), We.detach())
    return out_fits, losses


res = json.load(open(OUT)) if os.path.exists(OUT) else {'jobs': {}}
if 'baseline_ce_fw' not in res:
    res['baseline_ce_fw'] = round(audit_fw(None), 4)
CE0 = res['baseline_ce_fw']
print(f'baseline CE fineweb {CE0}', flush=True)


def save():
    json.dump(res, open(OUT, 'w'), indent=2)


for (n, k, seed) in JOBS:
    key = f'n{n}_k{k}_s{seed}'
    if key in res['jobs'] and 'dce_ctx' in res['jobs'][key]:
        print(f'{key}: cached, skipping', flush=True)
        continue
    try:
        bits = NHB * dl_sparse_dict(n, ROW, V * k)
        row = {'n': n, 'k': k, 'seed': seed, 'Mbits': round(bits / 1e6, 1),
               'pct_raw': round(100 * bits / (32 * NHB * V * ROW), 2)}
        print(f'=== {key} ({row["Mbits"]} Mbit, {row["pct_raw"]}% raw)', flush=True)

        blob_path = f'{QK}/qk_dict_l0_seed0.pt'
        if (n, k, seed) == (1024, 8, 0) and os.path.exists(blob_path):
            blob = torch.load(blob_path, map_location=DEV)
            fits = [(blob[f'Dn{i}'], blob[f'b{i}'], blob[f'We{i}']) for i in range(NHB)]
            print('  loaded cached seed-0 n=1024 k=8 fits', flush=True)
        else:
            fits = []
            for bi, hb in enumerate(HB):
                Dn, b, We = train_dict(rows(*hb), n, k, seed=seed)
                fits.append((Dn, b, We))
            print(f'  fitted {NHB} head-branches (MSE)', flush=True)

        recs = [encode_token(rows(*hb), f[0], f[1], f[2], k) for f, hb in zip(fits, HB)]
        row['fvu_lin'] = round(sum(fvu(r, rows(*hb)) for r, hb in zip(recs, HB)) / NHB, 5)
        row['dce_lin'] = round(audit_fw(tables_from(recs)) - CE0, 4)
        print(f'  MSE linear:  dCE {row["dce_lin"]:+.4f}  (fvu {row["fvu_lin"]:.3f})', flush=True)
        res['jobs'][key] = row; save()

        if seed == 0:
            recs = [encode_omp(rows(*hb), f[0], f[1], k) for f, hb in zip(fits, HB)]
            row['dce_omp'] = round(audit_fw(tables_from(recs)) - CE0, 4)
            print(f'  MSE OMP/LS:  dCE {row["dce_omp"]:+.4f}', flush=True)
            save()

        ctx_fits, losses = ctx_finetune(fits, n, k, seed)
        row['ctx_losses'] = losses
        recs = [encode_token(rows(*hb), f[0], f[1], f[2], k) for f, hb in zip(ctx_fits, HB)]
        row['dce_ctx'] = round(audit_fw(tables_from(recs)) - CE0, 4)
        print(f'  OV-context:  dCE {row["dce_ctx"]:+.4f}', flush=True)
        del fits, ctx_fits, recs
        torch.cuda.empty_cache()
        save()
    except Exception:
        res['jobs'][key] = {**res['jobs'].get(key, {}), 'error': traceback.format_exc()[-500:]}
        save()
        print(f'  {key} FAILED — continuing', flush=True)
        torch.cuda.empty_cache()

print('\nsweep complete', flush=True)
save()
