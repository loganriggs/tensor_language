"""CE-POLISH UPPER-BOUND DIAGNOSTIC (Logan 2026-07-22): how much held-out cross-entropy is left
on the table by training the layer-0 QK dictionary on weight-space MSE instead of CE?

Frozen-support CE training through the frozen model (the ov_sparse.py recipe, adapted from the
value path to the query/key path): supports = each token's top-8 atoms from the seed-0 linear
encoder (deployment assignment, FROZEN); trainable = atoms + coefficients + biases, per
head-branch (n=1024, k=8). Model bf16, params frozen, Adam + cosine + grad clipping.

Data hygiene: FineWeb 600 seqs split — TRAIN = seqs 300..599, AUDIT = seqs 0..299 (153,600
held-out predictions; disjoint). All CE in the same bf16 grad-enabled forward (its own baseline).
NOTE: this arm is NOT weight-only (train tokens enter the fit) — it is an upper-bound diagnostic
only, per Logan. ~12M trainable params on 154k train tokens: overfitting is expected; we track
held-out dCE every 150 steps and report the MINIMUM (that is the upper bound).
Writes qk_ce_polish.json.
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
from tier2_folding import branch_factors

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_ce_polish.json'
BRANCHES = (('q1', 'k1'), ('q2', 'k2'))
N_ATOMS, K = 1024, 8
STEPS, EVAL_EVERY = 1200, 150

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
AUDIT, TRAIN = FW[:300], FW[300:].to(DEV)

TAB = {}
for br, (qn, kn) in enumerate(BRANCHES, start=1):
    qh, kh = branch_factors(m, br)
    TAB[qn], TAB[kn] = qh.float().to(DEV), kh.float().to(DEV)
HB = [(h, qn, kn) for h in range(NH) for (qn, kn) in BRANCHES]

# frozen supports + trainable parts from the saved seed-0 dictionary (deployment assignment)
blob = torch.load(f'{QK}/qk_dict_l0_seed0.pt', map_location=DEV)
SUP, CF, DN, BB = [], [], [], []
with torch.no_grad():
    for bi, (h, qn, kn) in enumerate(HB):
        X = torch.cat([TAB[qn][:, h], TAB[kn][:, h]], 1)
        Dn, b, We = blob[f'Dn{bi}'], blob[f'b{bi}'], blob[f'We{bi}']
        z = (X - b) @ We.T
        _, idx = z.abs().topk(K, dim=1)
        SUP.append(idx)                                             # (V, K) frozen
        CF.append(torch.gather(z, 1, idx).clone().requires_grad_(True))
        DN.append(Dn.clone().requires_grad_(True))
        BB.append(b.clone().requires_grad_(True))
PARAMS = CF + DN + BB

m.to(torch.bfloat16)
for p in m.parameters():
    p.requires_grad_(False)


def unit_rms(t):
    return t * (t.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt())


def live_qk(tokens, h_idx):
    """Reconstructed unit-RMS q_hat/k_hat rows for one head from trainable dict parts."""
    out = []
    for br in (0, 1):
        bi = h_idx * 2 + br
        rec = BB[bi] + (CF[bi][tokens].unsqueeze(-1) * DN[bi][SUP[bi][tokens]]).sum(-2)
        out.append((unit_rms(rec[..., :HD]), unit_rms(rec[..., HD:])))
    return out                                                      # [(q1,k1),(q2,k2)] per head


def forward(tokens, live=False):
    x = m.transformer.wte(tokens)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    v1 = None
    B, T = tokens.shape
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        h = F.rms_norm(x, (x.size(-1),))
        cos, sin = rope_tables(T, HD, tokens.device, x.dtype, 'bf16')
        cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]

        def qk(lin):
            z = lin(h).view(B, T, NH, HD)
            return apply_rot(F.rms_norm(z, (HD,)), cosr, sinr)

        v = a.c_v(h).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        mask = torch.tril(torch.ones(T, T, device=tokens.device, dtype=torch.bool))
        if li == 0 and live:
            qs = [torch.empty(B, T, NH, HD, device=DEV, dtype=torch.float32) for _ in range(4)]
            for hh in range(NH):
                (q1, k1), (q2, k2) = live_qk(tokens, hh)
                for t_, val in zip(qs, (q1, k1, q2, k2)):
                    t_[:, :, hh] = val
            q, k_, q2_, k2_ = [apply_rot(t_.to(x.dtype), cosr, sinr) for t_ in qs]
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k_) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2_, k2_) / HD
        else:
            q, k_ = qk(a.c_q), qk(a.c_k)
            q2_, k2_ = qk(a.c_q2), qk(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k_) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2_, k2_) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    x = F.rms_norm(x, (x.size(-1),))
    return 30 * torch.tanh(m.lm_head(x) / 30)


@torch.no_grad()
def ce(live=False):
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        logits = forward(b[:, :-1], live=live).float()
        tot += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               b[:, 1:].reshape(-1)).item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


CE0 = ce(live=False)
d_before = ce(live=True) - CE0
res = {'baseline_ce': round(CE0, 4), 'dce_mse_dict': round(d_before, 4),
       'n_train_preds': int(TRAIN.shape[0] * 512), 'n_audit_preds': int(AUDIT.shape[0] * 512),
       'curve': []}
print(f'baseline {CE0:.4f} (bf16); MSE-trained dict dCE {d_before:+.4f}', flush=True)

opt = torch.optim.Adam(PARAMS, lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
g = torch.Generator(); g.manual_seed(0)
best = d_before
for step in range(STEPS):
    bb = TRAIN[torch.randint(0, len(TRAIN), (4,), generator=g)]
    logits = forward(bb[:, :-1], live=True).float()
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), bb[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(PARAMS, 1.0)
    opt.step(); sched.step()
    if (step + 1) % EVAL_EVERY == 0:
        d = ce(live=True) - CE0
        best = min(best, d)
        res['curve'].append({'step': step + 1, 'train_ce': round(loss.item(), 4),
                             'dce_heldout': round(d, 4)})
        print(f'  step {step + 1}: train CE {loss.item():.4f}  held-out dCE {d:+.4f}', flush=True)
        json.dump(res, open(OUT, 'w'), indent=2)

res['dce_ce_polished_best'] = round(best, 4)
res['gap_closed'] = round(d_before - best, 4)
json.dump(res, open(OUT, 'w'), indent=2)
print(f'\nMSE dict {d_before:+.4f} -> CE-polished best {best:+.4f} '
      f'(upper-bound gap {d_before - best:+.4f})', flush=True)
print(f'wrote {OUT}', flush=True)
