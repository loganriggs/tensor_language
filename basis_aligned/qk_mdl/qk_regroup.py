"""REGROUPING A/B (tick 169b-lite, spec Stage-1 without the value leg yet): same raw
object and identical bits, but rows regrouped ACROSS branches — query-pairs [q1|q2]
and key-pairs [k1|k2] per head (vs the current within-branch [q|k] grouping). Trained
with the exact-moment objective (tick 169a recipe). Tests whether cross-branch
correlation (principal angles: 20-50 of 128 aligned, most on head 3) codes cheaper.
Arms: rg1024 base, rg1024_b256 anchors; comparators em1024 +0.0027 / em1024_b256 +0.0023.
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
OUT = f'{QK}/qk_regroup.json'
NAMES = ('q1', 'k1', 'q2', 'k2')
BRANCHES = (('q1', 'k1'), ('q2', 'k2'))
CTX_M, T_CTX, CTX_STEPS, CTX_LR, N_OFF = 1024, 512, 1500, 3e-4, 8

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
    a0 = m.transformer.h[0].attn
    E = F.rms_norm(m.transformer.wte.weight.detach().float(), (D,))
    Vv = a0.c_v(E).view(V, NH, HD)
    Wo = a0.c_proj.weight.detach().float().view(D, NH, HD)
QFULL = (torch.bincount(FINEWEB.flatten(), minlength=V).float() + 0.5).to(DEV)
QP = QFULL / QFULL.sum()
COS, SIN = rope_tables(T_CTX, HD, DEV, torch.float32, table_dtype='fp32')


def rows(h, qn, kn):
    return torch.cat([TAB[qn][:, h], TAB[kn][:, h]], 1)


def unit_rms(t):
    return t * (t.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt())


def tables_from(recs_by_hb):
    out = {n: TAB[n].clone() for n in NAMES}
    for bi, rec in recs_by_hb.items():
        h, qn, kn = HB[bi]
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


def train_head_incoh(h, fits, k, alpha=1.0, blend=False, seed_off=0):
    """Incoherent-rotary context finetune (diagnosis winner), optional q^alpha query
    weighting (S3) and 0.5-MSE blend. (Dn, b, We) ordering throughout."""
    g = torch.Generator(device='cpu').manual_seed(7 + 100 * h + seed_off)
    parts, params = {}, []
    for br in (0, 1):
        Dn0, b0, We0 = fits[h * 2 + br]
        Dm = Dn0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        We = We0.clone().requires_grad_(True)
        parts[br] = (Dm, b, We)
        params += [Dm, b, We]
    opt = torch.optim.Adam(params, lr=CTX_LR)
    Uh = Vv[:, h] @ Wo[:, h].T
    W2 = Uh.pow(2).sum(1)
    for step in range(CTX_STEPS):
        sample = torch.randperm(V, generator=g)[:CTX_M].to(DEV)
        qs = QFULL[sample]
        qs = qs / qs.sum()
        qi = QFULL[sample].pow(alpha)
        qi = qi / qi.sum()
        Us = Uh[sample]
        qw = qs * W2[sample]
        halves, mse_terms = [], []
        for br, (qn, kn) in enumerate(BRANCHES):
            X = torch.cat([TAB[qn][sample, h], TAB[kn][sample, h]], 1)
            Dm, b, We = parts[br]
            Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
            z = (X - b) @ We.T
            vals, idx = z.abs().topk(k, dim=1)
            coeff = torch.gather(z, 1, idx)
            rec = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
            mse_terms.append((rec - X).pow(2).sum() / X.pow(2).sum().clamp_min(1e-12))
            halves.append((unit_rms(rec[:, :HD]), unit_rms(rec[:, HD:]), X[:, :HD], X[:, HD:]))
        offs = torch.randperm(T_CTX, generator=g)[:N_OFF]
        num = den = 0.0
        for Dlt in offs.tolist():
            c, s = COS[Dlt], SIN[Dlt]
            Ph, P = None, None
            for (qr, kr, qo, ko) in halves:
                Sh = apply_rot(qr, c, s) @ kr.T / HD
                with torch.no_grad():
                    So = apply_rot(qo, c, s) @ ko.T / HD
                Ph = Sh if Ph is None else Ph * Sh
                P = So if P is None else P * So
            dP = Ph - P
            mu = (dP * qs[None, :]) @ Us
            s_ = (dP.pow(2) * qw[None, :]).sum(1)
            num = num + (qi * (T_CTX * (s_ - mu.pow(2).sum(1)).clamp_min(0)
                               + T_CTX ** 2 * mu.pow(2).sum(1))).sum()
            with torch.no_grad():
                mu0 = (P * qs[None, :]) @ Us
                s0 = (P.pow(2) * qw[None, :]).sum(1)
                den = den + (qi * (T_CTX * (s0 - mu0.pow(2).sum(1)).clamp_min(0)
                                   + T_CTX ** 2 * mu0.pow(2).sum(1))).sum()
        loss = num / den.clamp_min(1e-12)
        if blend:
            loss = 0.5 * loss + 0.5 * sum(mse_terms)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    del Uh, W2
    out = []
    for br in (0, 1):
        Dm, b, We = parts[br]
        out.append(((Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach(),
                    b.detach(), We.detach()))
    return out


@torch.no_grad()
def anchor_scores(recs):
    """Full-vocab dagger attribution (Delta=0): per-token combined query+key error score."""
    score = torch.zeros(V, device=DEV)
    tabs = tables_from(recs)
    CHK = 1024
    for h in range(NH):
        Uh = Vv[:, h] @ Wo[:, h].T
        w2 = Uh.pow(2).sum(1)
        qh1o, kh1o = tabs['q1'][:, h], tabs['k1'][:, h]
        qh2o, kh2o = tabs['q2'][:, h], tabs['k2'][:, h]
        q1e, k1e = unit_rms(TAB['q1'][:, h]), unit_rms(TAB['k1'][:, h])
        q2e, k2e = unit_rms(TAB['q2'][:, h]), unit_rms(TAB['k2'][:, h])
        for c0 in range(0, V, CHK):
            sl = slice(c0, min(c0 + CHK, V))
            Ph = (qh1o[sl] @ kh1o.T / HD) * (qh2o[sl] @ kh2o.T / HD)
            P = (q1e[sl] @ k1e.T / HD) * (q2e[sl] @ k2e.T / HD)
            dP = Ph - P
            del Ph, P
            mu = (dP * QP[None, :]) @ Uh
            s_ = (dP.pow(2) * (QP * w2)[None, :]).sum(1)
            mu2 = mu.pow(2).sum(1)
            score[sl] += QP[sl] * (T_CTX * (s_ - mu2).clamp_min(0) + T_CTX ** 2 * mu2)
            score += (dP.pow(2) * (QP[sl][:, None] * QP[None, :] * w2[None, :])).sum(0) * T_CTX
            A = Uh @ mu.T
            score += (dP.T * A * (QP[sl][None, :] * QP[:, None])).sum(1) * T_CTX ** 2
            del dP, mu, A
        del Uh
        torch.cuda.empty_cache()
    return score




N_OFF, Q_SUB, M_SCAT = 8, 8192, 1024

res = json.load(open(OUT)) if os.path.exists(OUT) else {'jobs': {}}
res['baseline_ce_fw'] = json.load(open(f'{QK}/qk_pareto_sweep.json'))['baseline_ce_fw']
CE0 = res['baseline_ce_fw']
QP = QFULL / QFULL.sum()
QP_CPU = QP.cpu()


def save():
    json.dump(res, open(OUT, 'w'), indent=2)


def build_core(k1h, k2h, Vpi):
    # Mc[:, :, kk] = k1^T (k2 * w_kk): 128 small matmuls — no VxHDxHD intermediate,
    # autograd-friendly (nothing large retained).
    return torch.stack([k1h.T @ (k2h * Vpi[:, kk:kk + 1]) for kk in range(HD)], dim=2)


def head_consts(h):
    Vh = Vv[:, h]
    Woh = Wo[:, h]
    Go = Woh.T @ Woh
    Vpi = Vh * QP[:, None]
    Mc0 = build_core(TAB['k1'][:, h], TAB['k2'][:, h], Vpi)
    return Vh, Go, Vpi, Mc0


def contract(Mc, q1r, q2r, chunk=4096):
    out = []
    for c0 in range(0, q1r.shape[0], chunk):
        sl = slice(c0, c0 + chunk)
        t1 = torch.einsum('ci,ijk->cjk', q1r[sl], Mc)
        out.append(torch.einsum('cjk,cj->ck', t1, q2r[sl]))
    return torch.cat(out)



res = json.load(open(OUT)) if os.path.exists(OUT) else {'jobs': {}}
res['baseline_ce_fw'] = json.load(open(f'{QK}/qk_pareto_sweep.json'))['baseline_ce_fw']
CE0 = res['baseline_ce_fw']


def save():
    json.dump(res, open(OUT, 'w'), indent=2)


def rg_rows(h):
    Xqq = torch.cat([TAB['q1'][:, h], TAB['q2'][:, h]], 1)
    Xkk = torch.cat([TAB['k1'][:, h], TAB['k2'][:, h]], 1)
    return Xqq, Xkk


def tables_from_rg(rg):
    out = {n: TAB[n].clone() for n in NAMES}
    for h, (rqq, rkk) in rg.items():
        out['q1'][:, h] = rqq[:, :HD]
        out['q2'][:, h] = rqq[:, HD:]
        out['k1'][:, h] = rkk[:, :HD]
        out['k2'][:, h] = rkk[:, HD:]
    return {n: unit_rms(out[n]) for n in NAMES}


@torch.no_grad()
def anchor_scores_tabs(tabs):
    score = torch.zeros(V, device=DEV)
    for h in range(NH):
        Uh = Vv[:, h] @ Wo[:, h].T
        w2 = Uh.pow(2).sum(1)
        q1e, k1e = unit_rms(TAB['q1'][:, h]), unit_rms(TAB['k1'][:, h])
        q2e, k2e = unit_rms(TAB['q2'][:, h]), unit_rms(TAB['k2'][:, h])
        for c0 in range(0, V, 1024):
            sl = slice(c0, min(c0 + 1024, V))
            Ph = (tabs['q1'][sl, h] @ tabs['k1'][:, h].T / HD) *                  (tabs['q2'][sl, h] @ tabs['k2'][:, h].T / HD)
            P = (q1e[sl] @ k1e.T / HD) * (q2e[sl] @ k2e.T / HD)
            dP = Ph - P
            del Ph, P
            mu = (dP * QP[None, :]) @ Uh
            s_ = (dP.pow(2) * (QP * w2)[None, :]).sum(1)
            mu2 = mu.pow(2).sum(1)
            score[sl] += QP[sl] * (T_CTX * (s_ - mu2).clamp_min(0) + T_CTX ** 2 * mu2)
            score += (dP.pow(2) * (QP[sl][:, None] * QP[None, :] * w2[None, :])).sum(0) * T_CTX
            A = Uh @ mu.T
            score += (dP.T * A * (QP[sl][None, :] * QP[:, None])).sum(1) * T_CTX ** 2
            del dP, mu, A
        del Uh
        torch.cuda.empty_cache()
    return score


def train_head_rg(h, fits_qq, fits_kk, k, seed=0):
    g = torch.Generator(device='cpu').manual_seed(7 + 100 * h + seed * 10000)
    parts, params = {}, []
    for tag, f in (('qq', fits_qq[h]), ('kk', fits_kk[h])):
        Dn0, b0, We0 = f
        Dm = Dn0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        We = We0.clone().requires_grad_(True)
        parts[tag] = (Dm, b, We)
        params += [Dm, b, We]
    opt = torch.optim.Adam(params, lr=CTX_LR)
    Vh, Go, Vpi, Mc0 = head_consts(h)
    Uh = Vh @ Wo[:, h].T
    W2 = Uh.pow(2).sum(1)
    Q1o, Q2o = TAB['q1'][:, h], TAB['q2'][:, h]
    Xqq, Xkk = rg_rows(h)
    for step in range(CTX_STEPS):
        if step % 50 == 0:
            torch.cuda.empty_cache()
        recs = {}
        for tag, X in (('qq', Xqq), ('kk', Xkk)):
            Dm, b, We = parts[tag]
            Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
            z = (X - b) @ We.T
            with torch.no_grad():
                idx = z.abs().topk(k, dim=1).indices
            coeff = torch.gather(z, 1, idx)
            recs[tag] = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
            del z
        q1h, q2h = unit_rms(recs['qq'][:, :HD]), unit_rms(recs['qq'][:, HD:])
        k1h, k2h = unit_rms(recs['kk'][:, :HD]), unit_rms(recs['kk'][:, HD:])
        Mch = build_core(k1h, k2h, Vpi)
        Ml = Mch.detach().requires_grad_(True)
        q1l = q1h.detach().requires_grad_(True)
        q2l = q2h.detach().requires_grad_(True)
        offs = torch.randperm(T_CTX, generator=g)[:N_OFF]
        qsub = torch.multinomial(QP_CPU, Q_SUB, replacement=True, generator=g).to(DEV)
        m0s, den_st = [], 0.0
        with torch.no_grad():
            for Dlt in offs.tolist():
                c, s = COS[Dlt], SIN[Dlt]
                m0 = contract(Mc0, apply_rot(Q1o[qsub], c, s), apply_rot(Q2o[qsub], c, s))
                m0s.append(m0)
                den_st += float(((m0 @ Go) * m0).sum(1).mean())
        den_st = T_CTX ** 2 * den_st / N_OFF
        sample = torch.randperm(V, generator=g)[:M_SCAT].to(DEV)
        qs = QFULL[sample]
        qs = qs / qs.sum()
        Us = Uh[sample]
        qw = qs * W2[sample]
        num_sc = 0.0
        den_sc = 0.0
        for Dlt in offs.tolist():
            c, s = COS[Dlt], SIN[Dlt]
            Ph = ((apply_rot(q1h[sample], c, s) @ k1h[sample].T / HD)
                  * (apply_rot(q2h[sample], c, s) @ k2h[sample].T / HD))
            with torch.no_grad():
                P = ((apply_rot(Q1o[sample], c, s) @ TAB['k1'][sample, h].T / HD)
                     * (apply_rot(Q2o[sample], c, s) @ TAB['k2'][sample, h].T / HD))
            dP = Ph - P
            mu = (dP * qs[None, :]) @ Us
            s_ = (dP.pow(2) * qw[None, :]).sum(1)
            num_sc = num_sc + (qs * (s_ - mu.pow(2).sum(1)).clamp_min(0)).sum()
            with torch.no_grad():
                mu0 = (P * qs[None, :]) @ Us
                s0 = (P.pow(2) * qw[None, :]).sum(1)
                den_sc = den_sc + (qs * (s0 - mu0.pow(2).sum(1)).clamp_min(0)).sum()
        num_sc = T_CTX * num_sc / N_OFF
        den_sc = T_CTX * den_sc / N_OFF
        den = float(den_st + float(den_sc)) or 1e-12
        opt.zero_grad()
        for m0, Dlt in zip(m0s, offs.tolist()):
            c, s = COS[Dlt], SIN[Dlt]
            mh = contract(Ml, apply_rot(q1l[qsub], c, s), apply_rot(q2l[qsub], c, s))
            dm = mh - m0
            piece = (T_CTX ** 2 / N_OFF / den) * ((dm @ Go) * dm).sum(1).mean()
            piece.backward()
        (num_sc / den).backward(retain_graph=True)
        torch.autograd.backward([Mch, q1h, q2h], [Ml.grad, q1l.grad, q2l.grad])
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    del Uh, W2, Mc0
    out = {}
    for tag in ('qq', 'kk'):
        Dm, b, We = parts[tag]
        out[tag] = ((Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach(),
                    b.detach(), We.detach())
    return out


def encode_rg(f, X, k):
    Dn, b, We = f
    return encode_token(X, Dn, b, We, k)


N, K = 1024, 8
if 'rg1024' not in res['jobs'] or 'dce' not in res['jobs'].get('rg1024', {}):
    fits_qq, fits_kk = [], []
    for h in range(NH):
        Xqq, Xkk = rg_rows(h)
        fits_qq.append(train_dict(Xqq, N, K, seed=0))
        fits_kk.append(train_dict(Xkk, N, K, seed=0))
    print('MSE fits done', flush=True)
    rg = {}
    for h in range(NH):
        tr = train_head_rg(h, fits_qq, fits_kk, K)
        Xqq, Xkk = rg_rows(h)
        rg[h] = (encode_rg(tr['qq'], Xqq, K), encode_rg(tr['kk'], Xkk, K))
        print(f'head {h} trained', flush=True)
    tabs = tables_from_rg(rg)
    bits = NHB * dl_sparse_dict(N, ROW, V * K)
    res['jobs']['rg1024'] = {'Mbits': round(bits / 1e6, 1),
                             'dce': round(audit_fw(tabs) - CE0, 4)}
    print(f"rg1024: dCE {res['jobs']['rg1024']['dce']:+.4f}", flush=True)
    save()
    sc = anchor_scores_tabs(tabs)
    top = sc.argsort(descending=True)[:256]
    for h in range(NH):
        rqq, rkk = rg[h]
        Xqq, Xkk = rg_rows(h)
        rqq[top] = Xqq[top]
        rkk[top] = Xkk[top]
    tabs = tables_from_rg(rg)
    bits2 = bits + NHB * 256 * ROW * 32 + 256 * 16
    res['jobs']['rg1024_b256'] = {'Mbits': round(bits2 / 1e6, 1),
                                  'dce': round(audit_fw(tabs) - CE0, 4)}
    print(f"rg1024_b256: dCE {res['jobs']['rg1024_b256']['dce']:+.4f}", flush=True)
    save()
print('regroup A/B complete', flush=True)
