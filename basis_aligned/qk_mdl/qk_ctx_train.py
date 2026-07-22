"""OV-CONSIDERED DICTIONARY (tick 159, Logan): train the layer-0 QK dictionary AGAINST the
validated context-expected OV metric (ov_metric_explainer.md eq. †) instead of factor-row MSE,
at identical description length (n=1024, k=8 per head-branch), and audit FineWeb delta-CE.

Objective per head h (both branches jointly — the pattern couples them): on a sampled token set,
  dP = S1_hat*S2_hat - S1*S2   (pattern error, pre-rotary, unit-RMS deployment gauge)
  mu_i = sum_j q_j dP(i,j) u_j ;  s_i = sum_j q_j dP(i,j)^2 ||u_j||^2
  loss = sum_i q_i [ T*(s_i - ||mu_i||^2) + T^2*||mu_i||^2 ]  /  (same functional of the true P)
with u_j = W_o^h W_v^h e_hat_j, q = FineWeb unigram, T = 512. Weight-only + unigram throughout.
Init from the converged MSE dictionaries (qk_dict_l0_seed0.pt); encoder top-k support scheme
unchanged, so bits are identical to the plain arm. Pre-registered: plain-MSE already sits at
+0.006-0.008 and factor FVU correlates 0.95, so the expected gain is small; a negative is a
result. Writes qk_ctx_train.json + qk_dict_l0_ctx.pt.
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward
from tier2_folding import branch_factors, scores_from_factors

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
NAMES = ('q1', 'k1', 'q2', 'k2')
BRANCHES = (('q1', 'k1'), ('q2', 'k2'))
K = 8
M = 1024                 # tokens per training sample
STEPS = 1500
T_CTX = 512.0

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))

TAB = {}
for br, (qn, kn) in enumerate(BRANCHES, start=1):
    qh, kh = branch_factors(m, br)
    TAB[qn], TAB[kn] = qh.float().to(DEV), kh.float().to(DEV)
HB = [(h, qn, kn) for h in range(NH) for (qn, kn) in BRANCHES]
blob = torch.load(f'{QK}/qk_dict_l0_seed0.pt', map_location=DEV)

with torch.no_grad():
    a = m.transformer.h[0].attn
    E = F.rms_norm(m.transformer.wte.weight.detach().float(), (D,))
    Vv = a.c_v(E).view(V, NH, HD)
    Wo = a.c_proj.weight.detach().float().view(D, NH, HD)
QFULL = (torch.bincount(FINEWEB.flatten(), minlength=V).float() + 0.5).to(DEV)


def unit_rms(t):
    return t * (t.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt())


def recon_rows(sample, Dm, We, b):
    """Reconstruct (M, 256) rows via the deployment encoder (top-k, signed magnitudes)."""
    X = torch.cat([TAB[qn][sample, h_cur], TAB[kn_][sample, h_cur]], 1)
    Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    z = (X - b) @ We.T
    vals, idx = z.abs().topk(K, dim=1)
    coeff = torch.gather(z, 1, idx)
    return b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1), X


trained = {}
g = torch.Generator(device='cpu').manual_seed(0)
for h_cur in range(NH):
    params = []
    parts = {}
    for br, (qn, kn_) in enumerate(BRANCHES):
        bi = h_cur * 2 + br
        Dm = blob[f'Dn{bi}'].clone().requires_grad_(True)
        We = blob[f'We{bi}'].clone().requires_grad_(True)
        b = blob[f'b{bi}'].clone().requires_grad_(True)
        parts[br] = (Dm, We, b)
        params += [Dm, We, b]
    opt = torch.optim.Adam(params, lr=3e-4)
    Uh_full = Vv[:, h_cur] @ Wo[:, h_cur].T                     # (V, D) OV vectors, this head
    first = last = None
    for step in range(STEPS):
        sample = torch.randperm(V, generator=g)[:M].to(DEV)
        qs = QFULL[sample]
        qs = qs / qs.sum()
        Us = Uh_full[sample]
        w2 = Us.pow(2).sum(1)
        Shat, S = [], []
        for br, (qn, kn_) in enumerate(BRANCHES):
            rec, X = recon_rows(sample, *parts[br])
            qh_, kh_ = unit_rms(rec[:, :HD]), unit_rms(rec[:, HD:])
            Shat.append(qh_ @ kh_.T / HD)
            S.append(X[:, :HD] @ X[:, HD:].T / HD)
        P = S[0] * S[1]
        dP = Shat[0] * Shat[1] - P
        def ctx(matrix):
            mu = (matrix * qs[None, :]) @ Us
            mu2 = mu.pow(2).sum(1)
            s_ = (matrix.pow(2) * (qs * w2)[None, :]).sum(1)
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
    print(f'head {h_cur}: ctx loss {first:.4f} -> {last:.4f}', flush=True)
    for br in (0, 1):
        bi = h_cur * 2 + br
        Dm, We, b = parts[br]
        trained[f'Dn{bi}'] = (Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach()
        trained[f'We{bi}'] = We.detach()
        trained[f'b{bi}'] = b.detach()

torch.save({k: v.cpu() for k, v in trained.items()}, f'{QK}/qk_dict_l0_ctx.pt')

# full-vocab reconstruction + FineWeb audit at identical bits
out = {n: TAB[n].clone() for n in NAMES}
with torch.no_grad():
    for bi, (h, qn, kn_) in enumerate(HB):
        X = torch.cat([TAB[qn][:, h], TAB[kn_][:, h]], 1)
        Dn, We, b = trained[f'Dn{bi}'], trained[f'We{bi}'], trained[f'b{bi}']
        z = (X - b) @ We.T
        vals, idx = z.abs().topk(K, dim=1)
        coeff = torch.gather(z, 1, idx)
        rec = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
        out[qn][:, h] = rec[:, :HD]
        out[kn_][:, h] = rec[:, HD:]
TABS = {n: unit_rms(out[n]) for n in NAMES}


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


CE0 = audit_fw(None)
d = audit_fw(TABS) - CE0
res = {'baseline_ce_fw': round(CE0, 4), 'dce_fw': round(d, 4), 'Mbits': 455.4,
       'objective': 'context-expected OV (eq. dagger)', 'init': 'seed-0 MSE dicts',
       'steps': STEPS, 'M': M, 'lr': 3e-4}
json.dump(res, open(f'{QK}/qk_ctx_train.json', 'w'), indent=2)
print(f'\nOV-context-trained dict (n=1024 k=8, 455.4 Mbit): dCE fw {d:+.4f} '
      f'(plain-MSE linear arm: +0.0076)', flush=True)
