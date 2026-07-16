"""Top-MLP arc, probe 1: bilin18's MLP is PURE bilinear (Down(Lx ⊙ Rx), not
gated), so MLP outputs decompose EXACTLY over stream pairs like QK scores did.
For the contextual top MLPs (L13, L16, L17) and bottom contrasts (L2, L5):
Down-column-weighted hidden energy per stream pair at sampled positions.
GATES: stream sum == x; pair-sum of hidden == full hidden."""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
TEST_L = [2, 5, 13, 16, 17]
P = 512
OUT = f'{QK}/mlp_stream_interactions.json'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
TOK = build_eval_tokens(n_chunks=16, seq_len=513)[:, :-1]

energy = {}
count = {}
gate_max = 0.0


@torch.no_grad()
def run_batch(idx):
    global gate_max
    B, T = idx.shape
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    streams = [x.clone()]
    v1 = None
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    g = torch.Generator(); g.manual_seed(11)
    pi = torch.randint(0, T, (P,), generator=g).to(DEV)
    bi = torch.randint(0, B, (P,), generator=g).to(DEV)
    for li, blk in enumerate(m.transformer.h):
        lam0, lam1 = blk.lambdas[0], blk.lambdas[1]
        x = lam0 * x + lam1 * x0
        streams = [lam0 * s for s in streams]
        streams[0] = streams[0] + lam1 * x0
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
        attn_out = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        x = x + attn_out
        streams.append(attn_out)
        # MLP input read = x after attention; streams must include attn_out
        d = (sum(streams) - x).abs().max().item()
        gate_max = max(gate_max, d)
        rms2 = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
        if li in TEST_L:
            S = len(streams)
            if li not in energy:
                energy[li] = torch.zeros(S, S, device=DEV)
                count[li] = 0
            mlp = blk.mlp
            wcol = mlp.Down.weight.pow(2).sum(0).sqrt()          # (4608,)
            sp = torch.stack([s[bi, pi] * rms2[bi, pi] for s in streams])   # (S,P,D)
            Ls = torch.einsum('spd,hd->sph', sp, mlp.Left.weight)
            Rs = torch.einsum('spd,hd->sph', sp, mlp.Right.weight)
            hid_full = (Ls.sum(0) * Rs.sum(0))
            hid_check = torch.einsum('aph,bph->ph', Ls, Rs)
            gate_max = max(gate_max, (hid_full - hid_check).abs().max().item())
            e = torch.zeros(S, S, device=DEV)
            # per-pair weighted energy: sum_h (Ls_a * Rs_b * wcol)^2 over p
            for aa in range(S):
                contrib = Ls[aa][None] * Rs * wcol[None, None]   # (S,P,H)
                e_row = contrib.pow(2).sum((1, 2))
                e[aa] = e_row
            energy[li] += e
            count[li] += 1
        mlp_out = blk.mlp(x * rms2)
        x = x + mlp_out
        streams.append(mlp_out)


for i in range(0, len(TOK), 4):
    run_batch(TOK[i:i + 4].to(DEV))
    print(f'  batch {i//4 + 1}/{len(TOK)//4} gate {gate_max:.2e}', flush=True)

summary = {'gate_max': gate_max}
for L in sorted(energy):
    S = energy[L].shape[0]
    nm = ['emb'] + [f'{t}{l}' for l in range(L) for t in ('attn', 'mlp')] + [f'attn{L}']
    E = energy[L] / count[L]
    tot = E.sum().item()
    pairs = sorted([(f'{nm[a]}×{nm[b]}', round(E[a, b].item() / tot, 4))
                    for a in range(S) for b in range(S)], key=lambda t: -t[1])[:8]
    rec = [i for i, n in enumerate(nm) if n != 'emb'
           and int(n[4:] if n.startswith('attn') else n[3:]) >= L - 2]
    r_share = E[rec][:, rec].sum().item() / tot
    summary[L] = {'recent_x_recent': round(r_share, 4), 'top_pairs': pairs}
    print(f'L{L}: recent×recent {r_share:.2f} | top {pairs[:4]}', flush=True)
with open(OUT, 'w') as fh:
    json.dump(summary, fh, indent=2)
print('mlp stream interactions done', flush=True)
