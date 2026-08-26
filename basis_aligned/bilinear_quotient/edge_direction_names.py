# edge_direction_names: NAME THE EDGE DIRECTIONS (user directive 2026-08-26: "we're
# defining the clusters with respect to the composition, so we should have examples").
# S1468 showed the block-1 edges are ~95% mean transport + a small low-rank signal, so
# the nameable objects are the CENTERED (signal) directions of each composed edge:
#   pattern edge CP = [c_q1; c_k1; c_q2_1; c_k2_1] @ Down0   (stacked, 4x1152 x 4608)
#   values  edge CV = c_v1 @ Down0                            (1152 x 4608)
#   mlp1    edge CM = [Left1; Right1] @ Down0                 (2x4608 x 4608)
# Method: h0 token table TAB[t] = mean mlp0-hidden when token t is input (960 rows,
# freq-weighted center removed), rms-whitened; SVD of each edge (whitened); for each
# top-4 right vector: top/bottom-25 tokens by projection of TAB (freq >= 5), decoded;
# head/map loading from the left vector; stability = top-30 token overlap between
# two disjoint 480-row halves.
#
# Registered predictions:
#   pred_a token spectra are STABLE (median top-30 overlap across directions >= .50)
#          — namability requires stability.
#   pred_b pattern-edge direction loadings are concentrated: each top-4 direction puts
#          >= 30% of its left-vector energy in ONE (map, head) block of 36.
#   pred_c the three edges' top-4 signal subspaces OVERLAP: mean pairwise principal
#          energy >= .40 (the same mlp0 signal feeds all three readers).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256; HD = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'edge_direction_names_results.json'
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')


@torch.no_grad()
def h0_table(rows):
    tsum = torch.zeros(50257, HD); tcnt = torch.zeros(50257)
    for i in range(0, rows.shape[0], 8):
        idx = rows[i:i + 8, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x
        blk = H[0]
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        at = blk.attn
        cos, sin = at.rotary(at.c_q(xin).view(-1, T, 9, 128))
        q = are(F.rms_norm(at.c_q(xin).view(-1, T, 9, 128), (128,)), cos, sin)
        k = are(F.rms_norm(at.c_k(xin).view(-1, T, 9, 128), (128,)), cos, sin)
        q2 = are(F.rms_norm(at.c_q2(xin).view(-1, T, 9, 128), (128,)), cos, sin)
        k2 = are(F.rms_norm(at.c_k2(xin).view(-1, T, 9, 128), (128,)), cos, sin)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
            * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
        tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        pat = pat.masked_fill(~tril, 0.0)
        v = at.c_v(xin).view(-1, T, 9, 128)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(v.dtype), v)
        xx = xm + at.c_proj(y.reshape(-1, T, D))
        z = F.rms_norm(xx, (D,))
        h = (blk.mlp.Left(z).float() * blk.mlp.Right(z).float()).reshape(-1, HD).cpu()
        toks = rows[i:i + 8, :-1].reshape(-1)
        tsum.index_add_(0, toks, h)
        tcnt.index_add_(0, toks, torch.ones(toks.shape[0]))
    return tsum, tcnt


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(960, skip=80)[:, :T + 1].contiguous()
    tsA, tcA = h0_table(ROWS[:480])
    tsB, tcB = h0_table(ROWS[480:])
    print("tables built", flush=True)
    ts, tc = tsA + tsB, tcA + tcB
    TAB = torch.where(tc.unsqueeze(1) > 0, ts / tc.clamp_min(1).unsqueeze(1),
                      torch.zeros(1, HD))
    mu = (ts.sum(0) / tc.sum()).to(DEV)                  # freq-weighted mean h0
    rms = (TAB.to(DEV) ** 2 * (tc.to(DEV) / tc.sum()).unsqueeze(1)).sum(0).sqrt() \
        .clamp_min(1e-6)
    TABc = TAB.to(DEV) - mu
    halves = []
    for tsX, tcX in ((tsA, tcA), (tsB, tcB)):
        TX = torch.where(tcX.unsqueeze(1) > 0, tsX / tcX.clamp_min(1).unsqueeze(1),
                         torch.zeros(1, HD)).to(DEV) - mu
        halves.append((TX, tcX))

    Wd0 = H[0].mlp.Down.weight.float().to(DEV)
    at1 = H[1].attn
    EDGES = {
        'pattern': torch.cat([at1.c_q.weight.float().to(DEV) @ Wd0,
                              at1.c_k.weight.float().to(DEV) @ Wd0,
                              at1.c_q2.weight.float().to(DEV) @ Wd0,
                              at1.c_k2.weight.float().to(DEV) @ Wd0], 0),
        'values': at1.c_v.weight.float().to(DEV) @ Wd0,
        'mlp1': torch.cat([H[1].mlp.Left.weight.float().to(DEV) @ Wd0,
                           H[1].mlp.Right.weight.float().to(DEV) @ Wd0], 0),
    }
    MAPNAMES = ['q', 'k', 'q2', 'k2']

    ok = tc.to(DEV) >= 5
    out = {'directions': {}}
    tops = {}
    stab_all = []
    conc_flags = []
    for nm, M in EDGES.items():
        Mw = M * rms.unsqueeze(0)
        U, S, Vt = torch.linalg.svd(Mw, full_matrices=False)
        tops[nm] = Vt[:4]                                # whitened h0-space dirs
        dirs = []
        for d in range(4):
            vec = (Vt[d] / rms)                          # back to raw h0 space? project
            proj = TABc @ (Vt[d] * rms)                  # whitened inner product
            proj = torch.where(ok, proj, torch.zeros_like(proj))
            top = proj.argsort(descending=True)[:25].cpu().tolist()
            bot = proj.argsort()[:25].cpu().tolist()
            ov = []
            for TX, tcX in halves:
                pX = TX @ (Vt[d] * rms)
                pX = torch.where(tcX.to(DEV) >= 3, pX, torch.zeros_like(pX))
                ov.append(set(pX.argsort(descending=True)[:30].cpu().tolist()))
            stab = len(ov[0] & ov[1]) / 30.0
            stab_all.append(stab)
            entry = {'sv_share': round(float(S[d] / S.sum()), 4),
                     'stability_top30': round(stab, 3),
                     'top_tokens': [ENC.decode([t]) for t in top],
                     'bottom_tokens': [ENC.decode([t]) for t in bot]}
            if nm == 'pattern':
                lv = U[:, d].view(4, 9, 128)
                en = (lv ** 2).sum(-1)                   # [4 maps, 9 heads]
                fi = int(en.argmax())
                mi, hi = fi // 9, fi % 9
                share = float(en.flatten()[fi] / en.sum())
                entry['top_block'] = f'{MAPNAMES[mi]}@head1.{hi}'
                entry['block_share'] = round(share, 3)
                conc_flags.append(share >= 0.30)
            dirs.append(entry)
        out['directions'][nm] = dirs
        print(f"{nm} done", flush=True)

    import itertools
    pens = []
    for a, b in itertools.combinations(EDGES, 2):
        G = tops[a] @ tops[b].T
        pens.append(float((G ** 2).sum()) / 4.0)
    mean_pe = sum(pens) / len(pens)

    import statistics
    med_stab = statistics.median(stab_all)
    pa = med_stab >= 0.50
    pb = all(conc_flags)
    pc = mean_pe >= 0.40
    out.update({'median_stability': round(med_stab, 3),
                'pattern_block_concentrated': conc_flags,
                'mean_pairwise_principal_energy': round(mean_pe, 3),
                'pred_a_stable_50': bool(pa), 'pred_b_concentrated_30': bool(pb),
                'pred_c_shared_subspace_40': bool(pc),
                'runtime_s': round(time.time() - t0, 1)})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"stab {med_stab:.3f} conc {conc_flags} pe {mean_pe:.3f}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
