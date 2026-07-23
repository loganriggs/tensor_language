"""EXACT-MOMENT OBJECTIVE (tick 169a, from the circuit-tensor framing): the static
(T^2) term of eq. dagger is a moment-tensor contraction — mu_i = Mhat(q1_i,q2_i,.) −
M(q1_i,q2_i,.) with M = sum_t pi_t k1_t (x) k2_t (x) v_t a 128^3 core per head, and
rotary rotates the QUERY side, so the same cores serve every offset. This replaces the
sampled inner-mu estimator (whose noise ticks 167/168 showed was binding) with the
EXACT value; only the outer query average (pi-sampled, 8192/step) and the scatter term
(1/57 of energy) remain stochastic.
Phase V verifies: exact static vs sampled estimates at M=1k/4k/16k (prediction: sampled
biased UP, shrinking with M). Then trains (1024,8) and (4096,16), audits base + anchor
hybrids against the tick-168 M=4096 numbers.
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
OUT = f'{QK}/qk_exact_moment.json'
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


# ---------- Phase V: verify exact static vs sampled estimates ----------
if 'verify' not in res:
    h = 0
    blob = torch.load(f'{QK}/qk_dict_l0_seed0.pt', map_location=DEV)
    fits0 = [(blob[f'Dn{i}'], blob[f'b{i}'], blob[f'We{i}']) for i in range(NHB)]
    with torch.no_grad():
        Vh, Go, Vpi, Mc0 = head_consts(h)
        recs = [encode_token(rows(*HB[i]), *fits0[i], 8) for i in (2 * h, 2 * h + 1)]
        q1h, k1h = unit_rms(recs[0][:, :HD]), unit_rms(recs[0][:, HD:])
        q2h, k2h = unit_rms(recs[1][:, :HD]), unit_rms(recs[1][:, HD:])
        Mch = build_core(k1h, k2h, Vpi)
        mh = contract(Mch, q1h, q2h)
        m0 = contract(Mc0, TAB['q1'][:, h], TAB['q2'][:, h])
        dm = mh - m0
        exact = float((QP * ((dm @ Go) * dm).sum(1)).sum())
        row = {'exact_static': exact, 'sampled': {}}
        g = torch.Generator(device='cpu').manual_seed(0)
        Uh = Vv[:, h] @ Wo[:, h].T
        for M in (1024, 4096, 16384):
            vals = []
            for _ in range(8 if M < 16384 else 3):
                sample = torch.randperm(V, generator=g)[:M].to(DEV)
                qs = QFULL[sample]
                qs = qs / qs.sum()
                Us = Uh[sample]
                Ph = (q1h[sample] @ k1h[sample].T / HD) * (q2h[sample] @ k2h[sample].T / HD)
                P = ((TAB['q1'][sample, h] @ TAB['k1'][sample, h].T / HD)
                     * (TAB['q2'][sample, h] @ TAB['k2'][sample, h].T / HD))
                mu = ((Ph - P) * qs[None, :]) @ Us
                vals.append(float((qs * mu.pow(2).sum(1)).sum()))
            row['sampled'][str(M)] = round(sum(vals) / len(vals), 6)
        row['exact_static'] = round(exact, 6)
        res['verify'] = row
        print(f"VERIFY head0: exact {exact:.3e} | sampled " +
              " ".join(f"M={m}:{v:.3e}" for m, v in row['sampled'].items()), flush=True)
        save()
        del Mch, mh, m0, Uh
        torch.cuda.empty_cache()


def train_head_exact(h, fits, k, seed=0):
    g = torch.Generator(device='cpu').manual_seed(7 + 100 * h + seed * 10000)
    parts, params = {}, []
    for br in (0, 1):
        Dn0, b0, We0 = fits[h * 2 + br]
        Dm = Dn0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        We = We0.clone().requires_grad_(True)
        parts[br] = (Dm, b, We)
        params += [Dm, b, We]
    opt = torch.optim.Adam(params, lr=CTX_LR)
    Vh, Go, Vpi, Mc0 = head_consts(h)
    Uh = Vh @ Wo[:, h].T
    W2 = Uh.pow(2).sum(1)
    Q1o, Q2o = TAB['q1'][:, h], TAB['q2'][:, h]
    X1 = rows(h, 'q1', 'k1')
    X2 = rows(h, 'q2', 'k2')
    for step in range(CTX_STEPS):
        recs = []
        for br, X in ((0, X1), (1, X2)):
            Dm, b, We = parts[br]
            Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
            z = (X - b) @ We.T
            vals, idx = z.abs().topk(k, dim=1)
            coeff = torch.gather(z, 1, idx)
            recs.append(b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1))
        q1h, k1h = unit_rms(recs[0][:, :HD]), unit_rms(recs[0][:, HD:])
        q2h, k2h = unit_rms(recs[1][:, :HD]), unit_rms(recs[1][:, HD:])
        Mch = build_core(k1h, k2h, Vpi)
        Ml = Mch.detach().requires_grad_(True)
        q1l = q1h.detach().requires_grad_(True)
        q2l = q2h.detach().requires_grad_(True)

        offs = torch.randperm(T_CTX, generator=g)[:N_OFF]
        qsub = torch.multinomial(QP, Q_SUB, replacement=True, generator=g).to(DEV)
        # pass 1 (no grad): original contractions per offset + full denominator
        m0s, den_st = [], 0.0
        with torch.no_grad():
            for Dlt in offs.tolist():
                c, s = COS[Dlt], SIN[Dlt]
                m0 = contract(Mc0, apply_rot(Q1o[qsub], c, s), apply_rot(Q2o[qsub], c, s))
                m0s.append(m0)
                den_st = den_st + float(((m0 @ Go) * m0).sum(1).mean())
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
        # pass 2 (grad, per offset — backward immediately so contraction graphs free)
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
    out = []
    for br in (0, 1):
        Dm, b, We = parts[br]
        out.append(((Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach(),
                    b.detach(), We.detach()))
    return out


RECS_CACHE, ANCH_CACHE = {}, {}


def get_recs(n, k):
    key = (n, k)
    if key not in RECS_CACHE:
        fits = [train_dict(rows(h, qn, kn), n, k, seed=0) for h, qn, kn in HB]
        trained = []
        for h in range(NH):
            trained += train_head_exact(h, fits, k)
            print(f'  head {h} trained', flush=True)
        RECS_CACHE.clear()
        ANCH_CACHE.clear()
        RECS_CACHE[key] = {bi: encode_token(rows(*hb), *f, k)
                           for bi, (f, hb) in enumerate(zip(trained, HB))}
    return RECS_CACHE[key]


JOBS = [('em1024', 1024, 8, 0), ('em1024_b256', 1024, 8, 256),
        ('em4096k16', 4096, 16, 0), ('em4096k16_b1024', 4096, 16, 1024)]
for (name, n, k, B) in JOBS:
    if name in res['jobs'] and 'dce' in res['jobs'][name]:
        print(f'{name}: cached', flush=True)
        continue
    try:
        print(f'=== {name}', flush=True)
        recs = get_recs(n, k)
        bits = NHB * dl_sparse_dict(n, ROW, V * k)
        if B > 0:
            if (n, k) not in ANCH_CACHE:
                ANCH_CACHE[(n, k)] = anchor_scores(recs)
            top = ANCH_CACHE[(n, k)].argsort(descending=True)[:B]
            recs = {bi: r.clone() for bi, r in recs.items()}
            for bi, hb in enumerate(HB):
                recs[bi][top] = rows(*hb)[top]
            bits += NHB * B * ROW * 32 + B * 16
        row = {'n': n, 'k': k, 'B': B, 'Mbits': round(bits / 1e6, 1),
               'dce': round(audit_fw(tables_from(recs)) - CE0, 4)}
        if B > 0:
            del recs
        torch.cuda.empty_cache()
        res['jobs'][name] = row
        print(f'  {name}: dCE {row["dce"]:+.4f} @ {row["Mbits"]} Mbit', flush=True)
        save()
    except Exception:
        res['jobs'][name] = {**res['jobs'].get(name, {}), 'error': traceback.format_exc()[-600:]}
        save()
        print(f'  {name} FAILED — continuing', flush=True)
        torch.cuda.empty_cache()

print('exact-moment arc complete', flush=True)
save()
