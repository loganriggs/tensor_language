"""CONTEXT-OBJECTIVE REFINEMENTS (tick 162, Logan: "Go ahead with all 3"): factorial test of
the three refinements to the OV-context training objective, at the flagship budget (1024,8)
and the plateau budget (4096,16), seed 0, standard FineWeb 307k-prediction audit.

  R — rotary inside the objective: score at positional offset D is apply_rot(q, cos_D, sin_D)
      . k / 128 (exactly the model's convention; k-side rotation folds into the q-side).
      Query fixed at the last position of a T=512 window -> offsets uniform on {0..511};
      8 offsets sampled per step. Exact target:
        E||e||^2 = sum_D (s_D - ||mu_D||^2)  +  ||sum_D mu_D||^2
      estimated as T * mean_D(scatter_D) + T^2 * <mu_bar_A, mu_bar_B> with an A/B split of the
      offsets so the squared-mean term is unbiased.
  C — co-occurrence-corrected context weights: q_{t|i} proportional to q_t * L(cl_i, cl_t),
      cluster lift from qk_cooc_lift.pt (256 embedding k-means clusters, 788M causal pairs
      from a disjoint FineWeb slice, +5 smoothing). Row-normalized on each sample.
  B — blended loss: 0.5 * relative-MSE(rows) + 0.5 * context ratio.

None of the three adds description-length bits (objective-side apparatus only, same status as
the unigram frequencies). 2 budgets x 8 combos = 16 arms; combo 000 re-runs the tick-160 ctx
recipe as a sanity anchor (expect ~+0.0054 / ~+0.0052). Resumable json -> qk_ctx_refine.json.
"""
import json
import os
import sys
import traceback
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward, rope_tables, apply_rot
from tier2_folding import branch_factors, scores_from_factors
from mdl_accounting import dl_sparse_dict
from qk_sae_lib import train_dict, encode_token

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
SMOKE = os.environ.get('QK_SMOKE') == '1'
OUT = f'{QK}/qk_ctx_refine_smoke.json' if SMOKE else f'{QK}/qk_ctx_refine.json'
NAMES = ('q1', 'k1', 'q2', 'k2')
BRANCHES = (('q1', 'k1'), ('q2', 'k2'))
CTX_STEPS, CTX_M, CTX_LR, T_CTX, N_OFF = (20 if SMOKE else 1500), 1024, 3e-4, 512, 8
BUDGETS = [(1024, 8), (4096, 16)]
COMBOS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1), (1, 0, 1), (0, 1, 1), (0, 0, 0)]
JOBS = [(n, k, r, c, b) for (n, k) in BUDGETS for (r, c, b) in COMBOS]
if SMOKE:
    JOBS = [(1024, 8, 1, 1, 1)]

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
NHB, ROW = NH * 2, 2 * HD
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
if SMOKE:
    FINEWEB = FINEWEB[:8]

TAB = {}
for br, (qn, kn) in enumerate(BRANCHES, start=1):
    qh, kh = branch_factors(m, br)
    TAB[qn], TAB[kn] = qh.float().to(DEV), kh.float().to(DEV)
HB = [(h, qn, kn) for h in range(NH) for (qn, kn) in BRANCHES]

with torch.no_grad():
    a0 = m.transformer.h[0].attn
    E = F.rms_norm(m.transformer.wte.weight.detach().float(), (D,))
    Vv = a0.c_v(E).view(V, NH, HD)
    Wo = a0.c_proj.weight.detach().float().view(D, NH, HD)
QFULL = (torch.bincount(FINEWEB.flatten(), minlength=V).float() + 0.5).to(DEV)
COS, SIN = rope_tables(T_CTX, HD, DEV, torch.float32, table_dtype='fp32')   # (512, 64)

cooc = torch.load(f'{QK}/qk_cooc_lift.pt', map_location=DEV)
ASSIGN = cooc['assign'].to(DEV).long()
LIFT = cooc['lift'].to(DEV)


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


def train_head_refined(h, fits, k, use_rot, use_cooc, use_blend, seed):
    g = torch.Generator(device='cpu').manual_seed(seed + 7 + 100 * h)
    parts, params = {}, []
    Xb = {}
    for br, (qn, kn) in enumerate(BRANCHES):
        Dn0, b0, We0 = fits[h * 2 + br]
        Dm = Dn0.clone().requires_grad_(True)
        We = We0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        parts[br] = (Dm, We, b)
        params += [Dm, We, b]
    opt = torch.optim.Adam(params, lr=CTX_LR)
    Uh = Vv[:, h] @ Wo[:, h].T
    W2 = Uh.pow(2).sum(1)
    first = last = None
    for step in range(CTX_STEPS):
        sample = torch.randperm(V, generator=g)[:CTX_M].to(DEV)
        qs = QFULL[sample]
        qs = qs / qs.sum()
        if use_cooc:
            cl = ASSIGN[sample]
            Wm = qs[None, :] * LIFT[cl][:, cl]
            Wm = Wm / Wm.sum(1, keepdim=True).clamp_min(1e-12)
        else:
            Wm = qs[None, :].expand(CTX_M, CTX_M)
        Us = Uh[sample]
        w2 = W2[sample]
        qw = Wm * w2[None, :]

        # reconstructed + original factor halves (unit-RMS, pre-rotation — model order)
        halves = []                                # per branch: (qr, kr, qo, ko, mse_br)
        for br, (qn, kn) in enumerate(BRANCHES):
            X = torch.cat([TAB[qn][sample, h], TAB[kn][sample, h]], 1)
            Dm, We, b = parts[br]
            Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
            z = (X - b) @ We.T
            vals, idx = z.abs().topk(k, dim=1)
            coeff = torch.gather(z, 1, idx)
            rec = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
            mse_br = (rec - X).pow(2).sum() / X.pow(2).sum().clamp_min(1e-12)
            halves.append((unit_rms(rec[:, :HD]), unit_rms(rec[:, HD:]),
                           X[:, :HD], X[:, HD:], mse_br))

        offs = (torch.randperm(T_CTX, generator=g)[:N_OFF] if use_rot
                else torch.zeros(1, dtype=torch.long))
        mus, scat, mu0s, scat0 = [], [], [], []
        for Dlt in offs.tolist():
            c, s = COS[Dlt], SIN[Dlt]
            Ph, P = None, None
            for (qr, kr, qo, ko, _) in halves:
                Sh = apply_rot(qr, c, s) @ kr.T / HD
                with torch.no_grad():
                    So = apply_rot(qo, c, s) @ ko.T / HD
                Ph = Sh if Ph is None else Ph * Sh
                P = So if P is None else P * So
            dP = Ph - P
            mu = (dP * Wm) @ Us
            s_ = (dP.pow(2) * qw).sum(1)
            scat.append((qs * (s_ - mu.pow(2).sum(1)).clamp_min(0)).sum())
            mus.append(mu)
            with torch.no_grad():
                mu0 = (P * Wm) @ Us
                s0 = (P.pow(2) * qw).sum(1)
                scat0.append((qs * (s0 - mu0.pow(2).sum(1)).clamp_min(0)).sum())
                mu0s.append(mu0)
        n_off = len(mus)
        scatter = T_CTX * torch.stack(scat).mean()
        if use_rot:
            muA = torch.stack(mus[:n_off // 2]).mean(0)
            muB = torch.stack(mus[n_off // 2:]).mean(0)
            static = T_CTX ** 2 * (qs * (muA * muB).sum(1)).sum()
        else:
            static = T_CTX ** 2 * (qs * mus[0].pow(2).sum(1)).sum()
        with torch.no_grad():
            mu0m = torch.stack(mu0s).mean(0)
            den = (T_CTX * torch.stack(scat0).mean()
                   + T_CTX ** 2 * (qs * mu0m.pow(2).sum(1)).sum()).clamp_min(1e-12)
        loss = (scatter + static) / den
        if use_blend:
            loss = 0.5 * loss + 0.5 * sum(hv[4] for hv in halves)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    del Uh, W2
    out = []
    for br in (0, 1):
        Dm, We, b = parts[br]
        out.append(((Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach(),
                    b.detach(), We.detach()))
    return out, (round(first, 4), round(last, 4))


res = json.load(open(OUT)) if os.path.exists(OUT) else {'jobs': {}}
if 'baseline_ce_fw' not in res:
    sweep_path = f'{QK}/qk_pareto_sweep.json'
    if not SMOKE and os.path.exists(sweep_path):
        res['baseline_ce_fw'] = json.load(open(sweep_path))['baseline_ce_fw']
    else:
        res['baseline_ce_fw'] = round(audit_fw(None), 4)
CE0 = res['baseline_ce_fw']
print(f'baseline CE fineweb {CE0}', flush=True)


def save():
    json.dump(res, open(OUT, 'w'), indent=2)


FITS_CACHE = {}


def get_fits(n, k):
    if (n, k) in FITS_CACHE:
        return FITS_CACHE[(n, k)]
    blob_path = f'{QK}/qk_dict_l0_seed0.pt'
    if (n, k) == (1024, 8) and os.path.exists(blob_path):
        blob = torch.load(blob_path, map_location=DEV)
        fits = [(blob[f'Dn{i}'], blob[f'b{i}'], blob[f'We{i}']) for i in range(NHB)]
        print('  loaded cached seed-0 n=1024 k=8 fits', flush=True)
    else:
        fits = [train_dict(rows(*hb), n, k, seed=0) for hb in HB]
        print(f'  fitted {NHB} head-branches (MSE)', flush=True)
    FITS_CACHE[(n, k)] = fits
    return fits


for (n, k, r_, c_, b_) in JOBS:
    key = f'n{n}_k{k}_R{r_}C{c_}B{b_}'
    if key in res['jobs'] and 'dce' in res['jobs'][key]:
        print(f'{key}: cached, skipping', flush=True)
        continue
    try:
        bits = NHB * dl_sparse_dict(n, ROW, V * k)
        row = {'n': n, 'k': k, 'rot': r_, 'cooc': c_, 'blend': b_,
               'Mbits': round(bits / 1e6, 1)}
        print(f'=== {key} ({row["Mbits"]} Mbit)', flush=True)
        fits = get_fits(n, k)
        trained, losses = [], []
        for h in range(NH):
            fh, fl = train_head_refined(h, fits, k, r_, c_, b_, seed=0)
            trained += fh
            losses.append(fl)
        row['losses'] = losses
        recs = [encode_token(rows(*hb), f[0], f[1], f[2], k) for f, hb in zip(trained, HB)]
        tabs = tables_from(recs)
        del recs, trained
        torch.cuda.empty_cache()
        row['dce'] = round(audit_fw(tabs) - CE0, 4)
        print(f'  dCE {row["dce"]:+.4f}', flush=True)
        del tabs
        torch.cuda.empty_cache()
        res['jobs'][key] = row
        save()
    except Exception:
        res['jobs'][key] = {**res['jobs'].get(key, {}), 'error': traceback.format_exc()[-600:]}
        save()
        print(f'  {key} FAILED — continuing', flush=True)
        torch.cuda.empty_cache()

print('\nctx-refine factorial complete', flush=True)
save()
