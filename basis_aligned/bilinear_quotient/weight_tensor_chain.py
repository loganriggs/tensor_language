# weight_tensor_chain: DECOMPOSE THE UPSTREAM MODULES BY THEIR OWN WEIGHT STRUCTURE AND
# COMPOSE (user directive 2026-08-25: PCs are not mlp3's natural features; mlp3 is
# itself many features; run the same metric on mlp1/mlp2/mlp3 and compose to degree 4).
# All pure-bilinear: out = Down(Left(x) * Right(x)). Three parts, ALL FROM WEIGHTS:
#  (1) Per-module self-structure for mlp1, mlp2, mlp3: for each of the top-16 output
#      directions g (left singular vectors of Down), the eigen-spectrum of
#      B_g = sym(Left^T diag(Down^T g) Right) — concentration = top-8 |eig| mass.
#  (2) mlp4's interaction tensor re-expressed with mlp3's NATIVE output features
#      (Down_3's top-64 left singular vectors) instead of variance PCs — block mass +
#      hub check vs §1425.
#  (3) Chain composition: for mlp4's top native-basis interaction pair involving an
#      mlp3 feature g, report g's own top eigen-pairs in mlp3 — the explicit degree-4
#      sentence.
# Feature bases for attn4/mlp0/mlp2 sides in part (2): attn4 = c_proj_4 row-space top-64
# right singular vectors; mlp0/mlp2 = their Down top-64 left singular vectors (native).
#
# Registered predictions:
#   pred_a mlp3's per-output eigen-spectra are CONCENTRATED: top-8 |eig| mass >= .50
#          averaged over its top-16 output directions (few input-pairs per feature).
#   pred_b the native basis preserves mlp3's dominance in mlp4's tensor: mlp3-involved
#          block mass >= .55 (PC basis gave .655).
#   pred_c the hub survives the basis change: >= 5 of mlp4's top-20 pairs share one
#          single native mlp3 feature.
import json, time, sys, torch
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'weight_tensor_chain_results.json'
R = 64
H = m.transformer.h
t0 = time.time()


def parts(L):
    mlp = H[L].mlp
    return (mlp.Left.weight.float().to(DEV), mlp.Right.weight.float().to(DEV),
            mlp.Down.weight.float().to(DEV))


@torch.no_grad()
def main():
    # ---- (1) self-structure of mlp1/2/3
    self_struct = {}
    for L in (1, 2, 3):
        Lw, Rw, Dw = parts(L)
        Ug, Sg, _ = torch.linalg.svd(Dw, full_matrices=False)
        conc = []
        for a in range(16):
            g = Ug[:, a]
            w = Dw.T @ g                                   # [hidden]
            Bg = Lw.T @ (w.unsqueeze(1) * Rw)              # [D, D]
            Bg = 0.5 * (Bg + Bg.T)
            ev = torch.linalg.eigvalsh(Bg)
            mag = ev.abs()
            conc.append(float(mag.topk(8).values.sum() / mag.sum()))
        self_struct[f'mlp{L}'] = {'down_spectrum_top8_frac':
                                  round(float(Sg[:8].sum() / Sg.sum()), 4),
                                  'mean_top8_eig_mass': round(sum(conc) / len(conc), 4),
                                  'per_dir': [round(c, 3) for c in conc]}
        print(f"mlp{L}: mean top-8 eig mass {self_struct[f'mlp{L}']['mean_top8_eig_mass']}",
              flush=True)

    # ---- (2) mlp4 tensor in NATIVE bases
    bases = {}
    for name, L in (('m0', 0), ('m2', 2), ('m3', 3)):
        _, _, Dw = parts(L)
        bases[name] = torch.linalg.svd(Dw, full_matrices=False)[0][:, :R]   # [D, R]
    cp4 = H[4].attn.c_proj.weight.float().to(DEV)          # [D, D_in]
    bases['a4'] = torch.linalg.svd(cp4, full_matrices=False)[0][:, :R]
    order = ['a4', 'm0', 'm2', 'm3']
    Fall = torch.cat([bases[n] for n in order], 1)          # [D, 4R]
    L4, R4, D4 = parts(4)
    Ug4 = torch.linalg.svd(D4, full_matrices=False)[0][:, :R]
    A = L4 @ Fall; B = R4 @ Fall
    Q = Ug4.T @ D4                                          # [R, hidden]
    I = torch.einsum('ah,hi,hj->aij', Q, A, B)
    I = 0.5 * (I + I.transpose(1, 2))
    M2 = (I ** 2).sum(0)
    total = float(M2.sum())
    blocks = {}
    for bi, ni in enumerate(order):
        for bj, nj in enumerate(order):
            if bj < bi:
                continue
            sl_i = slice(bi * R, (bi + 1) * R); sl_j = slice(bj * R, (bj + 1) * R)
            mass = float(M2[sl_i, sl_j].sum())
            if bi != bj:
                mass += float(M2[sl_j, sl_i].sum())
            blocks[f'{ni}x{nj}'] = round(mass / total, 4)
    m3_mass = sum(v for k, v in blocks.items() if 'm3' in k)
    print("native block mass:", blocks, flush=True)

    fam = [n for n in order for _ in range(R)]
    fl = torch.triu(M2)
    v, ix = fl.flatten().topk(20)
    top_pairs = []
    cnt = {}
    for val, ii in zip(v.tolist(), ix.tolist()):
        i2, j2 = ii // (4 * R), ii % (4 * R)
        na = f'{fam[i2]}#{i2 % R}'; nb = f'{fam[j2]}#{j2 % R}'
        top_pairs.append([na, nb, round(val / total, 4)])
        for nm in (na, nb):
            if nm.startswith('m3'):
                cnt[nm] = cnt.get(nm, 0) + 1
    hub = max(cnt.items(), key=lambda kv: kv[1]) if cnt else (None, 0)

    # ---- (3) compose the hub (or top m3 feature) through mlp3 -> degree-4 sentence
    chain = None
    if hub[0]:
        a_idx = int(hub[0].split('#')[1])
        g = bases['m3'][:, a_idx]
        Lw3, Rw3, Dw3 = parts(3)
        w = Dw3.T @ g
        Bg = Lw3.T @ (w.unsqueeze(1) * Rw3)
        Bg = 0.5 * (Bg + Bg.T)
        ev, evec = torch.linalg.eigh(Bg)
        mag = ev.abs()
        topk = mag.topk(8)
        chain = {'m3_feature': hub[0], 'hub_count_top20': hub[1],
                 'top8_eig_mass': round(float(topk.values.sum() / mag.sum()), 4),
                 'top_eigs': [round(float(ev[i]), 4) for i in topk.indices.tolist()]}
        print(f"chain: {chain}", flush=True)

    pa = sum(self_struct['mlp3']['per_dir']) / 16 >= 0.50
    pb = m3_mass >= 0.55
    pc = hub[1] >= 5
    out = {'self_struct': self_struct, 'native_blocks': blocks,
           'm3_involved_mass': round(m3_mass, 4), 'top_pairs': top_pairs,
           'hub': {'feature': hub[0], 'count': hub[1]}, 'chain_decomposition': chain,
           'pred_a_mlp3_concentrated': bool(pa), 'pred_b_m3_mass_55': bool(pb),
           'pred_c_hub_survives': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} ({m3_mass:.3f}) | pred_c {pc} ({hub})")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
