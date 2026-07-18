"""CIRCUIT CARD 4: the two-head interference, dissected. Card 1 found that
killing {H5,H7} hurts LESS than killing H7 alone. Hypothesis: with H7 dead,
H5's noisy carriage actively misleads; removing or CLEANING it repairs.
Arms on the corpus repeat harness (16 random A+A seqs, 2nd-half CE):
H7-dead / H7+H5-dead / H7-dead+H5-CLEANED / H7+H0-dead / H7+H3-dead (controls).
If clean ~= dead < H7-alone, the interference channel is H5 noise.
Original card 2 docstring follows.
H5 (the match head) attends CORRECTLY on repeats (53x signature) yet zeroing
it barely hurts — while CLEANING its carried content (cond-mean identity) or
low-rank-filtering its output IMPROVES copying (WW-6/7). This card shows all
of that on a single legible sequence with set-ablation controls.
Output: results/cards/card2_denoising.md"""
import json
import os
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens
from transformers import AutoTokenizer

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
os.makedirs(f'{QK}/results/cards', exist_ok=True)
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
tok = AutoTokenizer.from_pretrained('gpt2')

g = torch.Generator(); g.manual_seed(7)
REP = torch.randint(0, V, (16, 256), generator=g).repeat(1, 2)
@torch.no_grad()
def forward(idx, kill_heads=None, grab=False, vbar5=None, rank2_h5=None):
    """kill_heads: list of (layer, head) whose scores are zeroed.
    grab: capture L5 patterns + per-head outputs at L5."""
    B, T = idx.shape
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0, v1 = x, None
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    grabbed = {}
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        h = F.rms_norm(x, (x.size(-1),))
        qn = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosb, sinb)
        q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
        v = a.c_v(h).view(B, T, NH, HD)
        v1 = v if v1 is None else v1
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        if vbar5 is not None and li == 5:
            v = v.clone()
            v[:, :, 5] = vbar5[idx.cpu()].to(DEV, v.dtype)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        if kill_heads:
            for (kl, kh) in kill_heads:
                if kl == li:
                    s1 = s1.clone(); s2 = s2.clone()
                    s1[:, kh] = 0.0; s2[:, kh] = 0.0
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        if rank2_h5 is not None and li == 5:
            Wo5 = a.c_proj.weight.detach().float()[:, 5 * HD:6 * HD]
            o5 = y[:, :, 5].float() @ Wo5.T
            proj = torch.einsum('btd,kd->btk', o5, rank2_h5)
            o5f = torch.einsum('btk,kd->btd', proj, rank2_h5)
            y = y.clone()
            # remove head-5 slice, add filtered version through c_proj externally
            delta = (o5f - o5).to(x.dtype)
            x = x + delta
        if grab and li == 5:
            grabbed['pat'] = pat[0].clone()
            Wo = a.c_proj.weight.detach().float()
            for hh in (5, 7):
                grabbed[f'out{hh}'] = (y[0, :, hh].float()
                                        @ Wo[:, hh * HD:(hh + 1) * HD].T)
        x = x + a.c_proj(y.reshape(B, T, -1))
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),))
                        )
    xf = F.rms_norm(x, (x.size(-1),))
    logits = 30 * torch.tanh(m.lm_head(xf) / 30)
    return logits, grabbed



# H5 output PCs for rank-2 filter (small sample of natural data)
EST = build_eval_tokens(n_chunks=36, seq_len=513)[20:][:, :-1]
o5s = []
for i in range(0, 16, 4):
    id2 = EST[i:i + 4].to(DEV)
    _, G2 = forward(id2, grab=True)
    o5s.append(G2['out5'].reshape(-1, D))
O5 = torch.cat(o5s)
C5 = (O5.T @ O5) / len(O5)
r2basis = torch.linalg.eigh(C5)[1][:, -2:].T.contiguous()      # (2, D)
# vbar for H5 (cond-mean v-content by token)
acc = torch.zeros(V, HD); c_ = torch.zeros(V)
for i in range(0, len(EST), 4):
    id2 = EST[i:i + 4].to(DEV)
    B2, T2 = id2.shape
    x2 = m.transformer.wte(id2); x2 = F.rms_norm(x2, (x2.size(-1),))
    x0, v1 = x2, None
    mask = torch.tril(torch.ones(T2, T2, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T2, HD, DEV, x2.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    x = x2
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        h = F.rms_norm(x, (x.size(-1),))
        qn = lambda lin: apply_rot(F.rms_norm(lin(h).view(B2, T2, NH, HD), (HD,)), cosb, sinb)
        q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
        v = a.c_v(h).view(B2, T2, NH, HD)
        v1 = v if v1 is None else v1
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        if li == 5:
            fl = id2.reshape(-1).cpu()
            acc.index_add_(0, fl, v[:, :, 5].reshape(-1, HD).float().cpu())
            c_.index_add_(0, fl, torch.ones_like(fl, dtype=torch.float))
            break
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B2, T2, -1))
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
vbar5 = acc / c_.clamp_min(1)[:, None]
vbar5[c_ == 0] = acc.sum(0) / c_.sum()
print('h5 stats built', flush=True)




@torch.no_grad()
def rep_ce(**kw):
    tot, n = 0.0, 0
    for i in range(0, len(REP), 4):
        b = REP[i:i + 4].to(DEV)
        lg, _ = forward(b[:, :-1], **kw)
        lp = lg.float()
        tgt = b[:, 1:]
        ce = F.cross_entropy(lp[:, 256:].reshape(-1, V), tgt[:, 256:].reshape(-1))
        tot += ce.item() * tgt[:, 256:].numel(); n += tgt[:, 256:].numel()
    return tot / n


base = rep_ce()
arms = {
    'live': 0.0,
    'H7 dead': rep_ce(kill_heads=[(5, 7)]) - base,
    'H7 + H5 dead': rep_ce(kill_heads=[(5, 7), (5, 5)]) - base,
    'H7 dead + H5 CLEANED': rep_ce(kill_heads=[(5, 7)], vbar5=vbar5) - base,
    'H7 + H0 dead (control)': rep_ce(kill_heads=[(5, 7), (5, 0)]) - base,
    'H7 + H3 dead (control)': rep_ce(kill_heads=[(5, 7), (5, 3)]) - base,
    'H5 dead alone (ref)': rep_ce(kill_heads=[(5, 5)]) - base,
}
card = ["# Circuit card 4: the two-head interference, dissected\n",
        "**Why does killing BOTH heads hurt less than killing H7 alone?** (card 1's",
        "anomaly, taken to corpus scale on the repeat harness; baseline CE "
        f"{base:.3f} on second halves of 16 random A+A sequences)\n",
        "| arm | ΔCE (repeat 2nd half) |", "|---|---|"]
for name, v in arms.items():
    card.append(f"| {name} | {v:+.4f} |")
res = {k: round(v, 4) for k, v in arms.items()}
res['baseline'] = base
import json as _json
with open(f'{QK}/card4.json', 'w') as fh:
    _json.dump(res, fh, indent=2)
verdict_data = (arms['H7 + H5 dead'], arms['H7 dead + H5 CLEANED'], arms['H7 dead'])
card += ["\n## Verdict\n", "(written at harvest from the numbers — see LOG)"]
with open(f'{QK}/results/cards/card4_interference.md', 'w') as fh:
    fh.write('\n'.join(card))
for name, v in arms.items():
    print(f'{name}: {v:+.4f}', flush=True)
print('card 4 done', flush=True)
