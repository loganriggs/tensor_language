"""CLASS-PAIR CIRCUITS (new arc, Logan directive 2026-07-20): TN-native
meaningful circuits at layer 0 with falsifiable monosemanticity scoring.

The layer-0 pattern is an exact tensor P(t_i, t_j, D). Coarsen its token
indices by the embedding classes (kmeans-256, reused from ngram2): each
(head, class_q, class_k) BLOCK is a candidate circuit atom "when a class-A
token queries a class-B key". For the top-energy blocks:
  causal probe = zero the pattern entries of that block only ->
  effect vector = mean Dlogits over affected query positions, scored by
    - CONCENTRATION: share of |Dlogit| mass in top-20 tokens (monosemantic
      effects are concentrated; diffuse effects falsify the atom)
    - CONSISTENCY: mean pairwise cosine of per-position Dlogit vectors
      (a real atom does the same thing every time it fires)
    - named output tokens (what the circuit promotes/suppresses)
Estimation: block energy E[pat^2] by class pair over audit chunks; causal
probes on the same chunks (deterministic audit slice)."""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens
from transformers import AutoTokenizer

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/cp_circuits.json'
TOPB = 14
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
tok = AutoTokenizer.from_pretrained('gpt2')
AUDIT = build_eval_tokens(n_chunks=20, seq_len=513)[4:20]
CLS = torch.load(f'{QK}/ngram2_pairclass.pt')['cls']          # (V,) kmeans-256 on emb
E_hat = F.rms_norm(m.transformer.wte.weight.detach().float(), (D,))


def class_exemplars(c, k=5):
    ids = (CLS == c).nonzero().squeeze(1)
    if len(ids) == 0:
        return []
    mu = E_hat[ids].mean(0)
    sims = F.cosine_similarity(E_hat[ids], mu[None], dim=1)
    return [tok.decode([ids[j]]) for j in sims.topk(min(k, len(ids))).indices.tolist()]


@torch.no_grad()
def forward(idx, block=None):
    """block: (head, cq, ck) -> zero layer-0 pattern entries of that block.
    Returns logits and (if block) the affected-query-position mask."""
    B, T = idx.shape
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0, v1 = x, None
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    cls_pos = CLS.to(DEV)[idx]                                # (B,T)
    affected = None
    pat0 = None
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        h = F.rms_norm(x, (x.size(-1),))
        qn = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosb, sinb)
        q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
        v = a.c_v(h).view(B, T, NH, HD)
        v1 = v if v1 is None else v1
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        if li == 0:
            pat0 = pat
            if block is not None:
                hh, cq, ck = block
                bm = (cls_pos[:, :, None] == cq) & (cls_pos[:, None, :] == ck)  # (B,T,T)
                pat = pat.clone()
                pat[:, hh] = pat[:, hh].masked_fill(bm, 0.0)
                affected = bm.any(-1)                          # (B,T) queries touched
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
        x = x + a.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    xf = F.rms_norm(x, (x.size(-1),))
    return 30 * torch.tanh(m.lm_head(xf) / 30), pat0, affected


# ---- stage 1: block energy over the audit slice ----
energy = torch.zeros(NH, 256, 256, device=DEV)
count = torch.zeros(256, 256, device=DEV)
base_logits = []
for i in range(0, len(AUDIT), 4):
    b = AUDIT[i:i + 4].to(DEV)
    idx = b[:, :-1]
    lg, pat0, _ = forward(idx)
    base_logits.append(lg.cpu())
    cls_pos = CLS.to(DEV)[idx]
    B, T = idx.shape
    code = cls_pos[:, :, None] * 256 + cls_pos[:, None, :]     # (B,T,T)
    tri = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool), -1)[None]
    codef = code[:, tri[0]].reshape(-1)
    for hh in range(NH):
        pf = pat0[:, hh][:, tri[0]].reshape(-1).float()
        energy[hh].view(-1).index_add_(0, codef, pf * pf)
    count.view(-1).index_add_(0, codef, torch.ones_like(codef, dtype=torch.float))
mean_energy = energy / count.clamp_min(1)[None]
# rank blocks by TOTAL energy (mass = frequency x strength), skip near-empty cells
tot_energy = energy.clone()
tot_energy[:, count < 50] = 0.0
flat = tot_energy.view(NH, -1)
vals, ids = flat.view(-1).topk(TOPB)
blocks = [(int(i // (256 * 256)), int((i % (256 * 256)) // 256), int(i % 256))
          for i in ids.tolist()]
print('top blocks (head, cq, ck) by pattern-energy mass:', blocks, flush=True)

# ---- stage 2: causal probes with monosemanticity scoring ----
res = {'blocks': []}
for (hh, cq, ck) in blocks:
    dvecs = []
    n_aff = 0
    for bi, i in enumerate(range(0, len(AUDIT), 4)):
        b = AUDIT[i:i + 4].to(DEV)
        idx = b[:, :-1]
        lg, _, aff = forward(idx, block=(hh, cq, ck))
        dl = (lg - base_logits[bi].to(DEV)).float()            # (B,T,V)
        am = aff & (torch.arange(idx.shape[1], device=DEV)[None] < idx.shape[1])
        if am.any():
            dvecs.append(dl[am].cpu())
            n_aff += int(am.sum())
    if not dvecs:
        continue
    DL = torch.cat(dvecs)                                       # (N, V)
    mean_dl = DL.mean(0)
    mass = mean_dl.abs()
    conc = float(mass.topk(20).values.sum() / mass.sum().clamp_min(1e-9))
    Nc = min(len(DL), 200)
    sub = F.normalize(DL[:Nc], dim=1)
    cons = float((sub @ sub.T).mean())
    top_up = [tok.decode([w]) for w in mean_dl.topk(6).indices.tolist()]
    top_dn = [tok.decode([w]) for w in (-mean_dl).topk(6).indices.tolist()]
    entry = {'head': hh, 'cq': cq, 'ck': ck,
             'cq_exemplars': class_exemplars(cq), 'ck_exemplars': class_exemplars(ck),
             'n_affected_positions': n_aff,
             'mean_abs_dlogit': round(float(mass.mean()), 5),
             'concentration_top20': round(conc, 3),
             'consistency_cos': round(cons, 3),
             'promotes_on_ablate': top_up, 'suppresses_on_ablate': top_dn}
    res['blocks'].append(entry)
    print(f"H{hh} [{','.join(entry['cq_exemplars'][:3])}] -> [{','.join(entry['ck_exemplars'][:3])}]: "
          f"conc {conc:.2f} cons {cons:.2f} n={n_aff} "
          f"suppresses {top_dn[:3]}", flush=True)
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=2)
print('cp circuits done', flush=True)
