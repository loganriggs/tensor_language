"""ROTARY-OBJECTIVE DIAGNOSIS + VARIANTS (tick 163). The factorial (tick 162) shows
rotary-in-objective DEGRADES the flagship arm (+0.0103 vs +0.0054 plain ctx). Cheap arms
(~2.5 min) -> diagnose in a verified, principled way, then sweep variants.

Phase V — VERIFY the offset identity against the audit's own machinery:
    scores_from_factors applies rotary via per-position cos/sin difference tables;
    my objective uses S_D(a,b) = apply_rot(q_a, cos_D, sin_D) . k_b / 128 with D = i - j.
    Compare numerically on a real token sequence (fp32). A convention bug would explain
    everything cheaply — rule it out first.

Phase D — DIAGNOSE with a dense-offset cross-evaluation (no sampling noise):
    Retrain the 000 (plain ctx) and R-uniform-8 dictionaries (deterministic seeds ->
    bit-identical to the factorial arms), then evaluate BOTH under BOTH objectives on a
    dense 128-offset grid. 2x2 verdict:
      R-trained wins rotary-eval but loses dCE  -> objective MISAIMED (context-model wrong)
      R-trained loses even its own rotary-eval  -> OPTIMIZATION NOISE (estimator variance)
    Plus the wash-out meter: the coherent offset sum ||sum_D mu_D||^2 kills every rotary band
    with period << window (analytically ~half of the 64 bands), collapsing the T^2 systematic
    training signal. Measured as signal static share: pre-rotary vs coherent vs incoherent.

Phase X — VARIANTS at the flagship budget (1024, 8), each audited on FineWeb:
    u32_coh    32 uniform offsets, coherent static (variance test)
    tri8_coh   8 triangular offsets (P(D) ~ T-D, matches audit position distribution)
    u8_incoh   8 uniform offsets, INCOHERENT static T^2 * E_D ||mu_D||^2 (keeps all bands)
    tri8_incoh triangular + incoherent
    u8_slow    8 uniform coherent, lr 1e-4, 3000 steps (optimization test)
Writes qk_rot_diag.json; trained dicts saved to qk_rot_diag_dicts.pt (local only).
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
from qk_sae_lib import encode_token

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_rot_diag.json'
NAMES = ('q1', 'k1', 'q2', 'k2')
BRANCHES = (('q1', 'k1'), ('q2', 'k2'))
CTX_M, T_CTX = 1024, 512
N_DICT, K_DICT = 1024, 8

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
NHB = NH * 2
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))

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
COS, SIN = rope_tables(T_CTX, HD, DEV, torch.float32, table_dtype='fp32')


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


def recon(parts, sample, h, k, grad=True):
    """Reconstructed unit-RMS halves + originals for both branches on a token sample."""
    out = []
    for br, (qn, kn) in enumerate(BRANCHES):
        X = torch.cat([TAB[qn][sample, h], TAB[kn][sample, h]], 1)
        Dm, b, We = parts[br]
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = (X - b) @ We.T
        vals, idx = z.abs().topk(k, dim=1)
        coeff = torch.gather(z, 1, idx)
        rec = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
        out.append((unit_rms(rec[:, :HD]), unit_rms(rec[:, HD:]), X[:, :HD], X[:, HD:]))
    return out


def pat_at(halves, Dlt, hat):
    c, s = COS[Dlt], SIN[Dlt]
    P = None
    for (qr, kr, qo, ko) in halves:
        q_, k_ = (qr, kr) if hat else (qo, ko)
        S = apply_rot(q_, c, s) @ k_.T / HD
        P = S if P is None else P * S
    return P


def train_variant(h, fits, k, n_off, dist, static_mode, lr, steps, seed=0):
    """dist: 'uniform'|'tri'; static_mode: 'coh' (A/B split coherent) | 'incoh'."""
    g = torch.Generator(device='cpu').manual_seed(seed + 7 + 100 * h)
    tri_w = (T_CTX - torch.arange(T_CTX).float())
    tri_w = tri_w / tri_w.sum()
    parts, params = {}, []
    for br in (0, 1):
        Dn0, b0, We0 = fits[h * 2 + br]
        Dm = Dn0.clone().requires_grad_(True)
        We = We0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        parts[br] = (Dm, b, We)
        params += [Dm, We, b]
    opt = torch.optim.Adam(params, lr=lr)
    Uh = Vv[:, h] @ Wo[:, h].T
    W2 = Uh.pow(2).sum(1)
    first = last = None
    for step in range(steps):
        sample = torch.randperm(V, generator=g)[:CTX_M].to(DEV)
        qs = QFULL[sample]
        qs = qs / qs.sum()
        Us = Uh[sample]
        qw = qs * W2[sample]
        halves = recon(parts, sample, h, k)
        if dist == 'delta0':
            offs = torch.zeros(1, dtype=torch.long)
        elif dist == 'uniform':
            offs = torch.randperm(T_CTX, generator=g)[:n_off]
        else:
            offs = torch.multinomial(tri_w, n_off, replacement=False, generator=g)
        mus, scat, mu0s, scat0 = [], [], [], []
        for Dlt in offs.tolist():
            Ph = pat_at(halves, Dlt, hat=True)
            with torch.no_grad():
                P = pat_at(halves, Dlt, hat=False)
            dP = Ph - P
            mu = (dP * qs[None, :]) @ Us
            s_ = (dP.pow(2) * qw[None, :]).sum(1)
            scat.append((qs * (s_ - mu.pow(2).sum(1)).clamp_min(0)).sum())
            mus.append(mu)
            with torch.no_grad():
                mu0 = (P * qs[None, :]) @ Us
                s0 = (P.pow(2) * qw[None, :]).sum(1)
                scat0.append((qs * (s0 - mu0.pow(2).sum(1)).clamp_min(0)).sum())
                mu0s.append(mu0)
        scatter = T_CTX * torch.stack(scat).mean()
        if static_mode == 'coh':
            if len(mus) == 1:
                static = T_CTX ** 2 * (qs * mus[0].pow(2).sum(1)).sum()
            else:
                muA = torch.stack(mus[:len(mus) // 2]).mean(0)
                muB = torch.stack(mus[len(mus) // 2:]).mean(0)
                static = T_CTX ** 2 * (qs * (muA * muB).sum(1)).sum()
            with torch.no_grad():
                mu0m = torch.stack(mu0s).mean(0)
                den_st = T_CTX ** 2 * (qs * mu0m.pow(2).sum(1)).sum()
        else:
            static = T_CTX ** 2 * torch.stack([(qs * mu.pow(2).sum(1)).sum() for mu in mus]).mean()
            with torch.no_grad():
                den_st = T_CTX ** 2 * torch.stack(
                    [(qs * mu0.pow(2).sum(1)).sum() for mu0 in mu0s]).mean()
        with torch.no_grad():
            den = (T_CTX * torch.stack(scat0).mean() + den_st).clamp_min(1e-12)
        loss = (scatter + static) / den
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
        Dm, b, We = parts[br]
        out.append(((Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach(),
                    b.detach(), We.detach()))
    return out, (round(first, 4), round(last, 4))


@torch.no_grad()
def dense_eval(all_parts, k, n_samples=4, stride=4, seed=123):
    """Evaluate a dictionary under (a) pre-rotary and (b) dense-grid rotary objectives.
    Returns dict of normalized losses + signal static shares (pre / coherent / incoherent)."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    grid = list(range(0, T_CTX, stride))                 # 128 offsets
    num_pre = den_pre = 0.0
    num_rot_sc = num_rot_st = 0.0
    den_rot_sc = den_rot_st = 0.0
    sig_pre_st = sig_pre_tot = 0.0
    sig_coh_st = sig_incoh_st = sig_rot_tot = 0.0
    for h in range(NH):
        Uh = Vv[:, h] @ Wo[:, h].T
        W2h = Uh.pow(2).sum(1)
        for _ in range(n_samples):
            sample = torch.randperm(V, generator=g)[:CTX_M].to(DEV)
            qs = QFULL[sample]
            qs = qs / qs.sum()
            Us = Uh[sample]
            qw = qs * W2h[sample]
            halves = recon(all_parts[h], sample, h, k, grad=False)
            mu_sum = mu0_sum = None
            sc_n = sc_d = 0.0
            st_incoh_n = st_incoh_d = 0.0
            sig_incoh = 0.0
            for Dlt in grid:
                Ph = pat_at(halves, Dlt, hat=True)
                P = pat_at(halves, Dlt, hat=False)
                dP = Ph - P
                mu = (dP * qs[None, :]) @ Us
                mu0 = (P * qs[None, :]) @ Us
                s_ = (dP.pow(2) * qw[None, :]).sum(1)
                s0 = (P.pow(2) * qw[None, :]).sum(1)
                sc_n += float((qs * (s_ - mu.pow(2).sum(1)).clamp_min(0)).sum())
                sc_d += float((qs * (s0 - mu0.pow(2).sum(1)).clamp_min(0)).sum())
                st_incoh_n += float((qs * mu.pow(2).sum(1)).sum())
                st_incoh_d += float((qs * mu0.pow(2).sum(1)).sum())
                sig_incoh += float((qs * mu0.pow(2).sum(1)).sum())
                mu_sum = mu if mu_sum is None else mu_sum + mu
                mu0_sum = mu0 if mu0_sum is None else mu0_sum + mu0
                if Dlt == 0:
                    num_pre += float(T_CTX * (qs * (s_ - mu.pow(2).sum(1)).clamp_min(0)).sum()
                                     + T_CTX ** 2 * (qs * mu.pow(2).sum(1)).sum())
                    den_pre += float(T_CTX * (qs * (s0 - mu0.pow(2).sum(1)).clamp_min(0)).sum()
                                     + T_CTX ** 2 * (qs * mu0.pow(2).sum(1)).sum())
                    sig_pre_st += float(T_CTX ** 2 * (qs * mu0.pow(2).sum(1)).sum())
                    sig_pre_tot += float(T_CTX * (qs * (s0 - mu0.pow(2).sum(1)).clamp_min(0)).sum()
                                         + T_CTX ** 2 * (qs * mu0.pow(2).sum(1)).sum())
            ng = len(grid)
            num_rot_sc += T_CTX * sc_n / ng
            den_rot_sc += T_CTX * sc_d / ng
            num_rot_st += T_CTX ** 2 * float((qs * (mu_sum / ng).pow(2).sum(1)).sum())
            den_rot_st += T_CTX ** 2 * float((qs * (mu0_sum / ng).pow(2).sum(1)).sum())
            sig_coh_st += T_CTX ** 2 * float((qs * (mu0_sum / ng).pow(2).sum(1)).sum())
            sig_incoh_st += T_CTX ** 2 * sig_incoh / ng
            sig_rot_tot += T_CTX * sc_d / ng
        del Uh, W2h
    return {
        'loss_pre': round(num_pre / max(den_pre, 1e-12), 4),
        'loss_rot_dense': round((num_rot_sc + num_rot_st) / max(den_rot_sc + den_rot_st, 1e-12), 4),
        'sig_static_share_pre': round(sig_pre_st / max(sig_pre_tot, 1e-12), 4),
        'sig_static_coh_over_incoh': round(sig_coh_st / max(sig_incoh_st, 1e-12), 6),
        'sig_static_incoh_over_scatter': round(sig_incoh_st / max(sig_rot_tot, 1e-12), 4),
    }


res = json.load(open(OUT)) if os.path.exists(OUT) else {}
res.setdefault('jobs', {})
res['baseline_ce_fw'] = json.load(open(f'{QK}/qk_pareto_sweep.json'))['baseline_ce_fw']
CE0 = res['baseline_ce_fw']


def save():
    json.dump(res, open(OUT, 'w'), indent=2)


# ---------- Phase V: verify offset identity against scores_from_factors ----------
if 'verify_max_err' not in res:
    Tv = 48
    toks = torch.arange(100, 100 + Tv, device=DEV)[None]           # arbitrary real token ids
    errs = []
    for brname, (qn, kn) in zip((1, 2), BRANCHES):
        s_ref = scores_from_factors(TAB[qn], TAB[kn], toks, HD, table_dtype='fp32')[0]  # (NH,Tv,Tv)
        for h in (0, 4, 8):
            q_ = TAB[qn][toks[0], h]                               # (Tv, HD)
            k_ = TAB[kn][toks[0], h]
            for i in range(0, Tv, 7):
                for j in range(0, i + 1, 5):
                    mine = float(apply_rot(q_[i], COS[i - j], SIN[i - j]) @ k_[j] / HD)
                    errs.append(abs(mine - float(s_ref[h, i, j])))
    res['verify_max_err'] = max(errs)
    print(f'VERIFY offset identity vs scores_from_factors: max abs err {max(errs):.3e}', flush=True)
    save()

# ---------- fits ----------
blob = torch.load(f'{QK}/qk_dict_l0_seed0.pt', map_location=DEV)
fits = [(blob[f'Dn{i}'], blob[f'b{i}'], blob[f'We{i}']) for i in range(NHB)]

VARIANTS = {
    'plain000': dict(n_off=1, dist='delta0', static_mode='coh', lr=3e-4, steps=1500),
    'R_u8_coh': dict(n_off=8, dist='uniform', static_mode='coh', lr=3e-4, steps=1500),
    'u32_coh': dict(n_off=32, dist='uniform', static_mode='coh', lr=3e-4, steps=1500),
    'tri8_coh': dict(n_off=8, dist='tri', static_mode='coh', lr=3e-4, steps=1500),
    'u8_incoh': dict(n_off=8, dist='uniform', static_mode='incoh', lr=3e-4, steps=1500),
    'tri8_incoh': dict(n_off=8, dist='tri', static_mode='incoh', lr=3e-4, steps=1500),
    'u8_slow': dict(n_off=8, dist='uniform', static_mode='coh', lr=1e-4, steps=3000),
}

dicts_blob = {}
for name, vcfg in VARIANTS.items():
    if name in res['jobs'] and 'dce' in res['jobs'][name]:
        print(f'{name}: cached, skipping', flush=True)
        continue
    try:
        print(f'=== {name} {vcfg}', flush=True)
        all_parts, losses = [], []
        for h in range(NH):
            ph, fl = train_variant(h, fits, K_DICT, **vcfg)
            all_parts.append(ph)
            losses.append(fl)
        flat = [all_parts[h][br] for h in range(NH) for br in (0, 1)]
        recs = [encode_token(rows(*hb), Dn, b, We, K_DICT) for (Dn, b, We), hb in zip(flat, HB)]
        tabs = tables_from(recs)
        del recs
        torch.cuda.empty_cache()
        dce = round(audit_fw(tabs) - CE0, 4)
        del tabs
        torch.cuda.empty_cache()
        ev = dense_eval(all_parts, K_DICT)
        row = {'dce': dce, 'losses': losses, **vcfg, **ev}
        res['jobs'][name] = row
        print(f'  dCE {dce:+.4f} | eval: pre {ev["loss_pre"]} rot_dense {ev["loss_rot_dense"]}',
              flush=True)
        for i, p in enumerate(flat):
            for t, ten in zip(('Dn', 'b', 'We'), p):
                dicts_blob[f'{name}_{i}_{t}'] = ten.cpu()
        torch.save(dicts_blob, f'{QK}/qk_rot_diag_dicts.pt')
        del all_parts, flat
        torch.cuda.empty_cache()
        save()
    except Exception:
        res['jobs'][name] = {**res['jobs'].get(name, {}), 'error': traceback.format_exc()[-600:]}
        save()
        print(f'  {name} FAILED — continuing', flush=True)
        torch.cuda.empty_cache()

# MSE-init reference row (no training): dense eval only
if 'mse_init' not in res['jobs']:
    all_parts = [{br: fits[h * 2 + br] for br in (0, 1)} for h in range(NH)]
    ev = dense_eval(all_parts, K_DICT)
    res['jobs']['mse_init'] = {'dce': 0.0076, 'note': 'dce from sweep (linear encoder)', **ev}
    print(f'mse_init eval: pre {ev["loss_pre"]} rot_dense {ev["loss_rot_dense"]}', flush=True)
    save()

print('\nrot-diag complete', flush=True)
save()
