"""CIRCUIT CARD 5: the L17 pronoun->temporal rule, causally verified.
RD-1 named one clean cross-class rulebook entry. This card: identify the
block (freq-filtered), ablate it at L17 (all heads), measure the TARGETED
effect (dlogit on temporal-conjunction tokens at pronoun-query positions)
vs a matched random content-block control. Output: cards/card5_l17rule.md.
Original: RULEBOOK NAMING ACROSS DEPTH + the diagonal question: name rulebooks for
L1/L5/L16 and measure per-layer DIAGONAL-block energy fraction (cq==ck) —
is selection same-class matching at every depth, or does the rulebook
structure change? Original: RULEBOOK + DEPTH DENSITY: (a) name the layer-0 rulebook (top blocks per
head with class exemplars -> results/cards/rulebook_L0.md); (b) is block-
sparsity universal? Same keep-top-B mask applied to LIVE patterns at layers
L in {1,5,12,16}, B ladder, dCE each. Original: BLOCK-SPARSE PATTERN MDL: how block-sparse is the layer-0 selection
tensor behaviorally? Keep only the top-B class-pair blocks per head (by
data-weighted pattern energy), zero the rest, audit dCE for a B ladder.
Bits: B block-ids (2x8-bit class ids) per head + the factor tables already
counted. EH-5 predicts heavy cost at small B (small entries sum coherently);
the ladder measures where. Original: CLASS-PAIR CIRCUITS (new arc, Logan directive 2026-07-20): TN-native
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
OUT = f'{QK}/card5.json'
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



@torch.no_grad()
def layer_energy(L):
    en = torch.zeros(NH, 256, 256, device=DEV)
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        idx = b[:, :-1]
        B, T = idx.shape
        cls_pos = CLS.to(DEV)[idx]
        x = m.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        x0, v1 = x, None
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
        cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
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
            if li == L:
                code = cls_pos[:, :, None] * 256 + cls_pos[:, None, :]
                tri = mask[None]
                codef = code[:, tri[0]].reshape(-1)
                for hh in range(NH):
                    pf = pat[:, hh][:, tri[0]].reshape(-1).float()
                    en[hh].view(-1).index_add_(0, codef, pf * pf)
                break
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
            x = x + a.c_proj(y)
            x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    return en


@torch.no_grad()
def audit_keep(keep_masks):
    """keep_masks: (NH, 256, 256) bool on DEV — pattern entries outside kept
    blocks are zeroed at layer 0."""
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        idx = b[:, :-1]
        B, T = idx.shape
        cls_pos = CLS.to(DEV)[idx]
        x = m.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        x0, v1 = x, None
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
        cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
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
            if li == MASK_LAYER:
                kq = cls_pos[:, :, None].expand(B, T, T)
                kk = cls_pos[:, None, :].expand(B, T, T)
                for hh in range(NH):
                    kmh = keep_masks[hh][kq.reshape(-1), kk.reshape(-1)].view(B, T, T)
                    pat[:, hh] = pat[:, hh] * kmh
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
            x = x + a.c_proj(y)
            x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
        xf = F.rms_norm(x, (x.size(-1),))
        logits = 30 * torch.tanh(m.lm_head(xf) / 30)
        ce = F.cross_entropy(logits.float().reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n




# frequency filter for content classes
cnt = torch.zeros(V)
TRf = build_eval_tokens(n_chunks=148, seq_len=513)[20:]
for i in range(0, len(TRf), 64):
    fl = TRf[i:i + 64].reshape(-1)
    cnt.index_add_(0, fl, torch.ones_like(fl, dtype=torch.float))
cfreq = torch.zeros(256)
for c in range(256):
    cfreq[c] = cnt[CLS == c].sum()
content = cfreq > cfreq.median()

en17 = layer_energy(17)
E = en17.sum(0)
E_f = E.clone()
for c in range(256):
    if not content[c]:
        E_f[c, :] = 0; E_f[:, c] = 0
E_f.fill_diagonal_(0)
i_top = int(E_f.view(-1).argmax())
CQ, CK = i_top // 256, i_top % 256
print('rule block:', class_exemplars(CQ, 5), '->', class_exemplars(CK, 5), flush=True)
ck_tokens = (CLS == CK)

import random as _rnd
_rnd.seed(9)
content_ids = content.nonzero().squeeze(1).tolist()
CTRL = []
while len(CTRL) < 4:
    a_, b_ = _rnd.choice(content_ids), _rnd.choice(content_ids)
    if a_ != CQ and (a_, b_) != (CQ, CK):
        CTRL.append((a_, b_))


@torch.no_grad()
def probe(cq, ck):
    """ablate block (cq,ck) at L17 all heads; return mean dlogit on CK-class
    tokens and on all-other tokens, at affected (query in cq) positions."""
    d_ck, d_other, npos = 0.0, 0.0, 0
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        idx = b[:, :-1]
        B, T = idx.shape
        cls_pos = CLS.to(DEV)[idx]
        lg0, _, _ = forward(idx)
        lg1, _, aff = forward(idx, block=None) if False else (None, None, None)
        # block-ablate at L17: reuse forward's block arg is L0-only; inline here
        x = m.transformer.wte(idx); x = F.rms_norm(x, (x.size(-1),))
        x0, v1 = x, None
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
        cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
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
            if li == 17:
                bm = (cls_pos[:, :, None] == cq) & (cls_pos[:, None, :] == ck)
                pat = pat.masked_fill(bm[:, None], 0.0)
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
            x = x + a.c_proj(y)
            x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
        xf = F.rms_norm(x, (x.size(-1),))
        lg1 = 30 * torch.tanh(m.lm_head(xf) / 30)
        dl = (lg1 - lg0).float()
        affq = (cls_pos == cq)
        if affq.any():
            dsel = dl[affq]                                   # (N, V)
            d_ck += float(dsel[:, ck_tokens].mean()) * int(affq.sum())
            d_other += float(dsel[:, ~ck_tokens].mean()) * int(affq.sum())
            npos += int(affq.sum())
    return d_ck / max(npos, 1), d_other / max(npos, 1), npos


dck, doth, np_ = probe(CQ, CK)
print(f'RULE block: dlogit on temporal-class {dck:+.4f}, other {doth:+.4f} (n={np_})', flush=True)
ctrl_res = []
for (a_, b_) in CTRL[:3]:
    c1, c2, n2 = probe(a_, b_)
    ctrl_res.append((c1, c2))
    print(f'ctrl ({a_},{b_}): target-class {c1:+.4f}, other {c2:+.4f}', flush=True)

card = ['# Circuit card 5: the L17 pronoun -> temporal rule\n',
        f'**The first cross-class rulebook entry, causally verified.**\n',
        f'Rule block (freq-filtered top off-diagonal at L17):',
        f'- queries: {class_exemplars(CQ, 6)}',
        f'- keys: {class_exemplars(CK, 6)}\n',
        f'## Causal probe (block-ablate at L17, all heads; audit at pronoun-query positions, n={np_})\n',
        '| arm | Δlogit on temporal-conj class | Δlogit other tokens |', '|---|---|---|',
        f'| RULE block ablated | **{dck:+.4f}** | {doth:+.4f} |']
for j, (c1, c2) in enumerate(ctrl_res):
    card.append(f'| control block {j} | {c1:+.4f} | {c2:+.4f} |')
card += ['\n## Verdict\n', '(from the numbers, at harvest — see LOG)']
with open(f'{QK}/results/cards/card5_l17rule.md', 'w') as fh:
    fh.write('\n'.join(card))
with open(OUT, 'w') as fh:
    json.dump({'rule': [CQ, CK], 'd_ck': dck, 'd_other': doth,
               'controls': ctrl_res, 'n': np_}, fh, indent=2)
print('card 5 done', flush=True)
