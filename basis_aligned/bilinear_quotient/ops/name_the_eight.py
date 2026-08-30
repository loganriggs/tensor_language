# NAME THE EIGHT: what are the top-8 loss-gradient directions at blocks 5 and 6 that select mlp4/mlp5's units?
#
# BENCHMARK_BACKLOG rung 17. §2110: a selector that scores each mlp4/mlp5 unit by how much its Down column writes into
# the top-8 eigenvectors of the site's loss-gradient Gramian reproduces the certified +0.124 / +0.075 nat (§2106) and
# beats wider heads. Sixteen vectors carry the frontier's first observability-arc gain. This is the descriptive pass
# that turns them from a selector into an object: are the block-5 and block-6 eights the same subspace, who reads
# them at attn5, are they logit directions, and which tokens load on them. Weights + two Gramians + one pass over the
# FIT rows for token loadings; no fits, no arms.
#
# REGISTERED PREDICTIONS:
#   (a) ONE OBJECT, NOT TWO: the top-8 subspaces at block 5's and block 6's input overlap at >= 0.5 (mean squared
#       cosine of the 8 principal angles). If FALSE the two sites weight different directions and "the eight" is two
#       different objects, one per MLP.
#   (b) attn5 READS THEM: at least one of attn5's nine heads has q, k or v projection energy on the block-5 eight
#       >= 2x its energy on random unit directions (mean of 64 draws). §2102 found attn5 amplifies mlp4's error
#       8.6x; this asks whether the amplification is a head-level read of exactly these directions. If FALSE the
#       eight are read diffusely and the amplification is not a nameable head.
#   (c) NOT LOGIT DIRECTIONS: the block-5 eight's energy in the top-64 right-singular subspace of lm_head is <= 0.2
#       (observability v1 measured 0.086 for the top-8 at block 2; the number is a check that the object is not
#       simply "the unembedding").
#
# Descriptive: per-direction top-12 tokens by |projection| of the block-5 stream over the FIT rows (positions >= 64);
# per-direction eigenvalue share; the eight's overlap with the block-1..4 heads (is it the massive direction?).
# Self-reviewed. Writes name_the_eight_results.json.
import json, sys, time, torch
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import os

if os.environ.get('BQLIB_DRYRUN') == '1':
    _bq = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    if not os.path.exists(_bq + 'metric_units_top8_results.json'):
        print('DRYRUN FAIL: S2110 artifact absent'); raise SystemExit(1)
    print('DRYRUN OK: weights + two Gramians + one token-loading pass')
    raise SystemExit(0)

import torch.nn.functional as F
import tiktoken
from bilin18_joint_removal import m, FW, DEV

D = 1152; TT = 256; SKIP = 64; CA, CB = 300, 512; NH = 9; HD = 128
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'name_the_eight_results.json'
t0 = time.time()
TOKF = torch.cat([FW[i:i + 4, :257] for i in range(CA, CB, 4)]).to(DEV)
ENC = tiktoken.get_encoding('gpt2')


def gramian_and_stream(site):
    G = torch.zeros(D, D, device=DEV, dtype=torch.float64); n = 0; streams = []; toks = []
    for b0 in range(0, TOKF.shape[0], 4):
        idx = TOKF[b0:b0 + 4, :-1]; tg = TOKF[b0:b0 + 4, 1:].reshape(-1)
        with torch.enable_grad():
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None; leaf = None
            for li, blk in enumerate(m.transformer.h):
                if li == site:
                    x = x.detach().requires_grad_(True); leaf = x
                x, v1 = blk(x, v1, x0)
            lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
            ce = F.cross_entropy(lg.view(-1, lg.size(-1)), tg, reduction='none').view(idx.shape[0], TT)
            ce[:, SKIP:].sum().backward()
        g = leaf.grad[:, SKIP:].reshape(-1, D).double(); G += g.T @ g; n += g.shape[0]
        streams.append(leaf.detach()[:, SKIP:].reshape(-1, D).float().cpu()); toks.append(idx[:, SKIP:].reshape(-1).cpu())
    e, Q = torch.linalg.eigh(G / n); e, Q = e.flip(0).clamp_min(0), Q.flip(1)
    return e, Q.float(), torch.cat(streams), torch.cat(toks)


e5, Q5, X5, T5 = gramian_and_stream(5)
e6, Q6, X6, T6 = gramian_and_stream(6)
P5, P6 = Q5[:, :8], Q6[:, :8]
# (a) subspace overlap: mean squared cosine of principal angles = ||P5^T P6||_F^2 / 8
overlap_56 = float(((P5.T @ P6) ** 2).sum() / 8)
# (c) lm_head row space
lm = m.lm_head.weight.detach().float()
Vlm = torch.linalg.svd(lm, full_matrices=False)[2][:64].T
lm_overlap = float(((Vlm.T @ P5) ** 2).sum() / 8)
# (b) attn5 head reads: energy of W_q/W_k/W_v (per head) applied to each direction, vs random directions
a5 = m.transformer.h[5].attn
gen = torch.Generator(device='cpu').manual_seed(8)
Rnd = F.normalize(torch.randn(D, 64, generator=gen), dim=0).to(DEV)
reads = {}
for name, lin in (('q', a5.c_q), ('k', a5.c_k), ('q2', a5.c_q2), ('k2', a5.c_k2), ('v', a5.c_v)):
    W = lin.weight.detach().float()                      # (NH*HD, D)
    def head_energy(P):
        Y = (W @ P).view(NH, HD, -1)                     # per head, per direction
        return (Y ** 2).sum(1).mean(1)                   # mean over directions -> (NH,)
    reads[name] = {'eight': head_energy(P5).tolist(), 'random': head_energy(Rnd).tolist()}
ratios = {name: [round(a / max(b, 1e-12), 3) for a, b in zip(v['eight'], v['random'])] for name, v in reads.items()}
best_ratio = max(r for v in ratios.values() for r in v)
best = max(((name, h, r) for name, v in ratios.items() for h, r in enumerate(v)), key=lambda z: z[2])
# descriptive: token loadings on the block-5 eight
proj = X5 @ P5.cpu()                                     # (N, 8)
tokens = {}
for d in range(8):
    order = proj[:, d].abs().argsort(descending=True)[:400]
    seen = {}
    for i in order.tolist():
        t = int(T5[i]); seen.setdefault(t, [0, 0.0]); seen[t][0] += 1; seen[t][1] += float(proj[i, d])
    top = sorted(seen.items(), key=lambda kv: -kv[1][0])[:12]
    tokens[str(d)] = [{'tok': ENC.decode([t]), 'count': c, 'mean_proj': round(s / c, 3)} for t, (c, s) in top]
# is any of the eight the massive/DC direction? overlap with the mean stream direction
mu5 = F.normalize(X5.mean(0), dim=0)
dc_overlap = [round(float((P5[:, d].cpu() @ mu5) ** 2), 4) for d in range(8)]
pa = overlap_56 >= 0.5
pb = best_ratio >= 2.0
pc = lm_overlap <= 0.2
out = {'eigen_share_top8_block5': [round(float(v), 4) for v in (e5[:8] / e5.sum())],
       'eigen_share_top8_block6': [round(float(v), 4) for v in (e6[:8] / e6.sum())],
       'overlap_block5_block6_top8': round(overlap_56, 4), 'lm_head_top64_overlap_block5_top8': round(lm_overlap, 4),
       'attn5_head_read_ratio_eight_over_random': ratios, 'best_read': {'proj': best[0], 'head': best[1], 'ratio': best[2]},
       'dc_direction_overlap_per_direction': dc_overlap, 'top_tokens_block5': tokens,
       'pred_a_one_object': bool(pa), 'pred_b_attn5_reads_them': bool(pb), 'pred_c_not_logit_directions': bool(pc),
       'self_reviewed': True, 'runtime_s': round(time.time() - t0, 1)}
json.dump(out, open(OUT, 'w'), indent=1)
print(f'eigen share top-8: block5 {out["eigen_share_top8_block5"]} | block6 {out["eigen_share_top8_block6"]}')
print(f'(a) block-5/6 top-8 overlap {overlap_56:.3f} >= 0.5: {"HELD" if pa else "FAILED"}')
print(f'(b) best attn5 head read ratio {best[0]} head {best[1]} = {best[2]:.2f} >= 2: {"HELD" if pb else "FAILED"}   all: ' +
      ' '.join(f'{k}:{max(v):.2f}' for k, v in ratios.items()))
print(f'(c) lm_head overlap {lm_overlap:.3f} <= 0.2: {"HELD" if pc else "FAILED"}   dc overlap per dir {dc_overlap}')
for d in range(8):
    print(f'  dir {d}: ' + ', '.join(f"{t['tok']!r}x{t['count']}({t['mean_proj']:+.1f})" for t in tokens[str(d)][:8]))
print(f'wrote {OUT} ({time.time() - t0:.0f}s)')
