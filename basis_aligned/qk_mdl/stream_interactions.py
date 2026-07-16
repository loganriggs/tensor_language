"""Logan's method B: interaction-norm map. The residual x_i at layer L is an
EXACT linear sum of streams (embedding path incl. the lambdas-x0 skip, plus
attn-out and mlp-out of every lower layer). c_q/c_k are linear, and both
RMSNorms are per-position scales shared across streams, so each branch score
S(i,j) = sum_ab q_a(i)·k_b(j) decomposes exactly over stream pairs (a,b).
This script measures E[(q_a(i)·k_b(j))^2] over sampled causal pairs — which
input interactions each layer's selection actually consumes (and how deep the
window is, method C's observational version).
GATES: (1) sum of streams == x to fp tolerance; (2) sum of pair-scores ==
live score on a sample.
Output: stream_interactions.pt (energy tensors) + json summary."""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
NL = cfg['n_layer']
TOK = build_eval_tokens(n_chunks=16, seq_len=513)[:, :-1]
P = 4096  # sampled causal pairs per sequence-batch

# energy[L][br] : (NH, S_L, S_L) accumulators (S_L = 1 + 2L streams)
energy = {}
count = {}
gate_max = 0.0


@torch.no_grad()
def run_batch(idx, first):
    global gate_max
    B, T = idx.shape
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    streams = [x.clone()]          # stream 0: embedding path
    names = ['emb']
    v1 = None
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    g = torch.Generator(device='cpu'); g.manual_seed(17)
    qi = torch.randint(1, T, (P,), generator=g).to(DEV)
    kj = (torch.rand(P, generator=g).to(DEV) * qi).long()   # kj < qi (causal)
    bi = torch.randint(0, B, (P,), generator=g).to(DEV)
    for li, blk in enumerate(m.transformer.h):
        lam0, lam1 = blk.lambdas[0], blk.lambdas[1]
        x = lam0 * x + lam1 * x0
        streams = [lam0 * s for s in streams]
        streams[0] = streams[0] + lam1 * x0
        a = blk.attn
        rms_x = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
        h = x * rms_x
        # gate 1: stream sum equals x
        d = (sum(streams) - x).abs().max().item()
        gate_max = max(gate_max, d)
        if li >= 1:
            S = len(streams)
            if li not in energy:
                energy[li] = {br: torch.zeros(NH, S, S, device=DEV) for br in (0, 1)}
                count[li] = 0
            for br, (cq, ck) in enumerate(((a.c_q, a.c_k), (a.c_q2, a.c_k2))):
                qf = cq(h).view(B, T, NH, HD)
                kf = ck(h).view(B, T, NH, HD)
                rq = qf.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
                rk = kf.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
                qs = torch.stack([apply_rot(cq(s * rms_x).view(B, T, NH, HD) * rq, cosb, sinb)
                                  for s in streams])   # (S,B,T,NH,HD)
                ks = torch.stack([apply_rot(ck(s * rms_x).view(B, T, NH, HD) * rk, cosb, sinb)
                                  for s in streams])
                qp = qs[:, bi, qi]                     # (S,P,NH,HD)
                kp = ks[:, bi, kj]
                dots = torch.einsum('aphd,bphd->abphd', qp, kp).sum(-1) / HD  # (S,S,P,NH)
                # gate 2: pair-sum equals live score on these samples
                qfull = apply_rot((qf * rq), cosb, sinb)[bi, qi]
                kfull = apply_rot((kf * rk), cosb, sinb)[bi, kj]
                sfull = (qfull * kfull).sum(-1) / HD   # (P,NH)
                d2 = (dots.sum((0, 1)) - sfull).abs().max().item()
                gate_max = max(gate_max, d2)
                energy[li][br] += dots.pow(2).mean(2).permute(2, 0, 1)
            count[li] += 1
        # live forward to next layer
        qn = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosb, sinb)
        q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
        v = a.c_v(h).view(B, T, NH, HD)
        v1 = v if v1 is None else v1
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        attn_out = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        x = x + attn_out
        streams.append(attn_out)
        names.append(f'attn{li}')
        rms2 = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
        mlp_out = blk.mlp(x * rms2)
        x = x + mlp_out
        streams.append(mlp_out)
        names.append(f'mlp{li}')
    return names


for i in range(0, len(TOK), 4):
    names = run_batch(TOK[i:i + 4].to(DEV), i == 0)
    print(f'  batch {i//4 + 1}/{len(TOK)//4}, gate_max {gate_max:.2e}', flush=True)

out = {str(L): {str(br): (energy[L][br] / count[L]).cpu() for br in (0, 1)}
       for L in energy}
torch.save({'energy': out, 'note': 'E[(q_a·k_b)^2] per (head, stream_a, stream_b)'},
           f'{QK}/stream_interactions.pt')

# summary: per layer, share of score energy by stream-pair category
summary = {'gate_max': gate_max}
for L in sorted(energy):
    S = energy[L][0].shape[-1]
    nm = ['emb'] + [f'{t}{l}' for l in range(L) for t in ('attn', 'mlp')]
    E = (energy[L][0] + energy[L][1]).sum(0)   # (S,S) over heads+branches
    tot = E.sum().item()
    ee = E[0, 0].item() / tot
    # emb x (anything recent = within 2 layers) vs older
    rec = [i for i, n in enumerate(nm) if n != 'emb' and int(n[4:] if n.startswith('attn') else n[3:]) >= L - 2]
    old = [i for i in range(1, S) if i not in rec]
    def share(rows, cols):
        return E[rows][:, cols].sum().item() / tot
    summary[L] = {
        'emb×emb': round(ee, 4),
        'emb×recent+recent×emb': round(share([0], rec) + share(rec, [0]), 4),
        'emb×old+old×emb': round(share([0], old) + share(old, [0]), 4),
        'recent×recent': round(share(rec, rec), 4),
        'old×old+old×recent': round(share(old, old) + share(old, rec) + share(rec, old), 4),
        'top_pairs': sorted([(f'{nm[a]}×{nm[b]}', round(E[a, b].item() / tot, 4))
                             for a in range(S) for b in range(S)],
                            key=lambda t: -t[1])[:6],
    }
    print(f'L{L}: emb×emb {ee:.2f} | top: {summary[L]["top_pairs"][:3]}', flush=True)
with open(f'{QK}/stream_interactions.json', 'w') as fh:
    json.dump(summary, fh, indent=2)
print('stream interactions done', flush=True)
