"""OV-LoRA joint arc (tick 161, Logan 2026-07-23): jointly optimize the sparse QK dictionary
AND a low-rank (LoRA) edit of the layer-0 OV reader, per head — Logan's question: does letting
the reader co-adapt buy a more extreme MDL, and does computation migrate?

Faithfulness-preserving design:
  - The objective is NOT downstream cross-entropy. Per head we match the ORIGINAL head's
    context-expected delivery to the residual stream:
        error_ij = Phat_ij * uhat_j  -  P_ij * u_j
    charged by eq. dagger of ov_metric_explainer.md (scatter at T, systematic at T^2,
    unigram-weighted). The compressed head must reproduce what the original head wrote into
    the stream; it may re-divide labor between pattern and reader internally, but it cannot
    invent new function. This generalizes the tick-159/160 context objective (which held the
    reader fixed).
  - The OV edit is a per-head LoRA on W_v^h and W_o^h (rank r, B zero-init so training starts
    at the exact original reader). Bits charged: 2 * r * (D + HD) * 32 per head on top of the
    dictionary bits.
  - Migration diagnostics per arm:
      * control audit: EXACT layer-0 scores + LoRA'd OV — how far the reader moved as a
        stand-alone model edit;
      * static share: fraction of context-expected output energy in the T^2 (context-mean,
        i.e. static) term, reconstruction vs original — if the LoRA inflates it, the head is
        being turned into a static bias (computation migrating out of attention);
      * relative Frobenius size of the reader change, and unigram-weighted content rank
        (eigenvalues to 90% energy) of the reader before/after.

Arms (all seed 0, FineWeb 307k-held-out audit, resumable json):
  joint (dict + LoRA r=16)  at (512,4) (1024,8) (4096,8) (4096,16)
  lora_only (dict frozen at MSE fit, LoRA r=16) at (1024,8) (4096,16)
  joint r=64 at (1024,8) (4096,16)   # does more reader capacity break the ctx plateau?
Comparators from qk_pareto_sweep.json: dce_lin / dce_omp / dce_ctx at the same budgets.
"""
import json
import os
import sys
import traceback
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward
from tier2_folding import branch_factors, scores_from_factors
from mdl_accounting import dl_sparse_dict
from qk_sae_lib import train_dict, encode_token

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
SMOKE = os.environ.get('QK_SMOKE') == '1'
OUT = f'{QK}/qk_ov_lora_smoke.json' if SMOKE else f'{QK}/qk_ov_lora.json'
NAMES = ('q1', 'k1', 'q2', 'k2')
BRANCHES = (('q1', 'k1'), ('q2', 'k2'))
JOBS = [
    (1024, 8, 'joint', 16),
    (4096, 16, 'joint', 16),
    (512, 4, 'joint', 16),
    (4096, 8, 'joint', 16),
    (1024, 8, 'lora_only', 16),
    (4096, 16, 'lora_only', 16),
    (1024, 8, 'joint', 64),
    (4096, 16, 'joint', 64),
]
CTX_STEPS, CTX_M, CTX_LR, T_CTX = (20 if SMOKE else 1500), 1024, 3e-4, 512.0
if SMOKE:
    JOBS = JOBS[:1]

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
    WV = a0.c_v.weight.detach().float().view(NH, HD, D)
    BVb = (a0.c_v.bias.detach().float().view(NH, HD)
           if getattr(a0.c_v, 'bias', None) is not None else None)
QFULL = (torch.bincount(FINEWEB.flatten(), minlength=V).float() + 0.5).to(DEV)
QP = QFULL / QFULL.sum()


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


@contextmanager
def lora_applied(loras):
    """Temporarily add the per-head LoRA deltas into the model's layer-0 c_v / c_proj weights."""
    cv, cp = a0.c_v.weight, a0.c_proj.weight
    cv0, cp0 = cv.data.clone(), cp.data.clone()
    with torch.no_grad():
        dWv = torch.zeros(NH * HD, D, device=DEV)
        dWo = torch.zeros(D, NH * HD, device=DEV)
        for h, (Av, Bv, Ao, Bo) in enumerate(loras):
            dWv[h * HD:(h + 1) * HD] = Av @ Bv
            dWo[:, h * HD:(h + 1) * HD] = Ao @ Bo
        cv.data = (cv.data.float() + dWv).to(cv0.dtype)
        cp.data = (cp.data.float() + dWo).to(cp0.dtype)
    try:
        yield
    finally:
        cv.data, cp.data = cv0, cp0


def branch_patterns(h, sample, parts, k):
    """Differentiable reconstructed + original pre-rotary pattern products on a token sample."""
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
    return Shat[0] * Shat[1], S[0] * S[1]


def dagger(P_hat, U_hat, P, U, qs):
    """Generalized eq.-dagger energy of the delivery error Phat*uhat - P*u (both None-able)."""
    if P_hat is None:                       # signal energy of (P, U) alone
        uu = U.pow(2).sum(1)
        s_ = (P.pow(2) * (qs * uu)[None, :]).sum(1)
        mu = (P * qs[None, :]) @ U
    else:
        nu = U_hat.pow(2).sum(1)
        uu = U.pow(2).sum(1)
        cu = (U_hat * U).sum(1)
        s_ = ((P_hat.pow(2) * (qs * nu)[None, :]).sum(1)
              - 2 * (P_hat * P * (qs * cu)[None, :]).sum(1)
              + (P.pow(2) * (qs * uu)[None, :]).sum(1))
        mu = (P_hat * qs[None, :]) @ U_hat - (P * qs[None, :]) @ U
    mu2 = mu.pow(2).sum(1)
    scatter = (qs * T_CTX * (s_ - mu2).clamp_min(0)).sum()
    static = (qs * T_CTX * T_CTX * mu2).sum()
    return scatter, static


def train_head(h, fits, arm, r, k, seed):
    g = torch.Generator(device='cpu').manual_seed(seed + 7 + 100 * h)
    parts, params = {}, []
    for br in (0, 1):
        Dn0, b0, We0 = fits[h * 2 + br]
        Dm, We, b = Dn0.clone(), We0.clone(), b0.clone()
        for t in (Dm, We, b):
            t.requires_grad_(arm != 'lora_only')
        parts[br] = (Dm, We, b)
        if arm != 'lora_only':
            params += [Dm, We, b]
    Av = (0.01 * torch.randn(HD, r, generator=g)).to(DEV).requires_grad_(True)
    Bv = torch.zeros(r, D, device=DEV, requires_grad=True)
    Ao = (0.01 * torch.randn(D, r, generator=g)).to(DEV).requires_grad_(True)
    Bo = torch.zeros(r, HD, device=DEV, requires_grad=True)
    params += [Av, Bv, Ao, Bo]
    opt = torch.optim.Adam(params, lr=CTX_LR)
    Uh = Vv[:, h] @ Wo[:, h].T
    first = last = None
    for step in range(CTX_STEPS):
        sample = torch.randperm(V, generator=g)[:CTX_M].to(DEV)
        qs = QFULL[sample]
        qs = qs / qs.sum()
        Us = Uh[sample]
        Wvh_new = WV[h] + Av @ Bv
        Woh_new = Wo[:, h] + Ao @ Bo
        Vh_s = E[sample] @ Wvh_new.T
        if BVb is not None:
            Vh_s = Vh_s + BVb[h]
        Uhat_s = Vh_s @ Woh_new.T
        P_hat, P = branch_patterns(h, sample, parts, k)
        with torch.no_grad():
            d_sc, d_st = dagger(None, None, P, Us, qs)
            den = (d_sc + d_st).clamp_min(1e-12)
        n_sc, n_st = dagger(P_hat, Uhat_s, P, Us, qs)
        loss = (n_sc + n_st) / den
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    del Uh
    trained = {br: tuple(t.detach() for t in parts[br]) for br in (0, 1)}
    # renormalize atoms into the stored convention (decoder rows unit-norm)
    for br in (0, 1):
        Dm, We, b = trained[br]
        trained[br] = (Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8), We, b)
    return trained, (Av.detach(), Bv.detach(), Ao.detach(), Bo.detach()), (round(first, 4), round(last, 4))


@torch.no_grad()
def arm_diags(all_parts, all_lora, k, seed):
    g = torch.Generator(device='cpu').manual_seed(seed + 999)
    stat = {'orig': [0.0, 0.0], 'arm': [0.0, 0.0]}   # [static, total]
    du2 = u2 = 0.0
    r90o, r90l = [], []
    for h in range(NH):
        Av, Bv, Ao, Bo = all_lora[h]
        Wvh_new = WV[h] + Av @ Bv
        Woh_new = Wo[:, h] + Ao @ Bo
        Uh = Vv[:, h] @ Wo[:, h].T
        for _ in range(2 if SMOKE else 4):
            sample = torch.randperm(V, generator=g)[:CTX_M].to(DEV)
            qs = QFULL[sample]
            qs = qs / qs.sum()
            Us = Uh[sample]
            Vh_s = E[sample] @ Wvh_new.T
            if BVb is not None:
                Vh_s = Vh_s + BVb[h]
            Uhat_s = Vh_s @ Woh_new.T
            P_hat, P = branch_patterns(h, sample, all_parts[h], k)
            for tag, Pm, Um in (('orig', P, Us), ('arm', P_hat, Uhat_s)):
                sc, st = dagger(None, None, Pm, Um, qs)
                stat[tag][0] += float(st)
                stat[tag][1] += float(sc + st)
        del Uh
        Co = torch.zeros(D, D, device=DEV)
        Cl = torch.zeros(D, D, device=DEV)
        for i0 in range(0, V, 8192):
            sl = slice(i0, min(i0 + 8192, V))
            U_c = Vv[sl, h] @ Wo[:, h].T
            Vh_c = E[sl] @ Wvh_new.T
            if BVb is not None:
                Vh_c = Vh_c + BVb[h]
            Ul_c = Vh_c @ Woh_new.T
            du2 += float((Ul_c - U_c).pow(2).sum())
            u2 += float(U_c.pow(2).sum())
            qc = QP[sl]
            Co += (U_c * qc[:, None]).T @ U_c
            Cl += (Ul_c * qc[:, None]).T @ Ul_c
        for C, acc in ((Co, r90o), (Cl, r90l)):
            ev = torch.linalg.eigvalsh(C).flip(0).clamp_min(0)
            cs = torch.cumsum(ev, 0) / ev.sum().clamp_min(1e-12)
            acc.append(int((cs < 0.90).sum().item()) + 1)
    return {
        'static_share_orig': round(stat['orig'][0] / max(stat['orig'][1], 1e-12), 4),
        'static_share_arm': round(stat['arm'][0] / max(stat['arm'][1], 1e-12), 4),
        'du_rel': round((du2 / max(u2, 1e-12)) ** 0.5, 4),
        'rank90_orig': r90o,
        'rank90_lora': r90l,
    }


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


for (n, k, arm, r) in JOBS:
    key = f'n{n}_k{k}_{arm}_r{r}'
    if key in res['jobs'] and 'dce' in res['jobs'][key]:
        print(f'{key}: cached, skipping', flush=True)
        continue
    try:
        dict_bits = NHB * dl_sparse_dict(n, ROW, V * k)
        lora_bits = NH * 2 * r * (D + HD) * 32
        row = {'n': n, 'k': k, 'arm': arm, 'r': r,
               'Mbits': round((dict_bits + lora_bits) / 1e6, 1),
               'lora_Mbits': round(lora_bits / 1e6, 1)}
        print(f'=== {key} ({row["Mbits"]} Mbit, of which LoRA {row["lora_Mbits"]})', flush=True)
        fits = get_fits(n, k)

        all_parts, all_lora, losses = [], [], []
        for h in range(NH):
            parts, lora, fl = train_head(h, fits, arm, r, k, seed=0)
            all_parts.append(parts)
            all_lora.append(lora)
            losses.append(fl)
            print(f'  head {h}: dagger-loss {fl[0]:.4f} -> {fl[1]:.4f}', flush=True)
        row['losses'] = losses

        flat = [all_parts[h][br] for h in range(NH) for br in (0, 1)]
        recs = [encode_token(rows(*hb), Dn, b, We, k) for (Dn, We, b), hb in zip(flat, HB)]
        tabs = tables_from(recs)
        del recs
        torch.cuda.empty_cache()
        with lora_applied(all_lora):
            row['dce'] = round(audit_fw(tabs) - CE0, 4)
        print(f'  joint audit (dict scores + LoRA OV): dCE {row["dce"]:+.4f}', flush=True)
        res['jobs'][key] = row
        save()

        if arm == 'joint' and r == 16:
            with lora_applied(all_lora):
                row['dce_exactP_lora'] = round(audit_fw(None) - CE0, 4)
            print(f'  control (EXACT scores + LoRA OV): dCE {row["dce_exactP_lora"]:+.4f}', flush=True)
            save()

        del tabs
        torch.cuda.empty_cache()
        row.update(arm_diags(all_parts, all_lora, k, seed=0))
        print(f'  static share orig {row["static_share_orig"]} -> arm {row["static_share_arm"]}; '
              f'reader change {row["du_rel"]} rel-Frobenius; '
              f'rank90 {row["rank90_orig"]} -> {row["rank90_lora"]}', flush=True)
        del all_parts, all_lora
        torch.cuda.empty_cache()
        save()
    except Exception:
        res['jobs'][key] = {**res['jobs'].get(key, {}), 'error': traceback.format_exc()[-600:]}
        save()
        print(f'  {key} FAILED — continuing', flush=True)
        torch.cuda.empty_cache()

print('\nov-lora arc complete', flush=True)
save()
