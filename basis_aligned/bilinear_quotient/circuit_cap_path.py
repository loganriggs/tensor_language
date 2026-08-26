# circuit_cap_path: CLOSE THE FALSE-NEGATIVE GAP WITH A PATH-AWARE SCORE (S1506:
# the direct output-subspace score plateaus at rho ~.74 / FN ~.62 — the damage tail
# arrives through indirect paths). New score: capture the FINAL-residual-stream
# delta under ensemble removal at capitalized-target positions (96 fit rows, one
# clean + one removal pass), top-32 SVD of the deltas -> subspace P; s2_w =
# fraction of unembedding row w inside P. Ground truth: the per-token damage saved
# in circuit_cap_fn_tokens.npz (S1506; no new causal runs needed).
#
# Registered predictions:
#   pred_a Spearman(s2, damage) >= .80 (the path-aware score beats the direct .74).
#   pred_b FN rate (top-50 damaged below the s2 top quartile) <= .45.
#   pred_c FP rate <= .30.
import json, time, sys, re, torch
import numpy as np
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'circuit_cap_path_results.json'
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
CONSTS = torch.load(PT + 'opt_ablation_consts_all.pt', map_location='cpu')
ENSEMBLE = {13: [0, 5], 14: [4, 6, 7], 15: [3], 16: [0, 3, 4, 5], 17: [0, 1, 2]}
HOOK = {'on': False}


def mk_hook(L):
    def hook(mod, args):
        if not HOOK['on']:
            return None
        hs = ENSEMBLE.get(L, [])
        if not hs:
            return None
        x = args[0].clone()
        for hh in hs:
            x[:, :, hh * 128:(hh + 1) * 128] = \
                CONSTS[f'head{L}.{hh}'].to(DEV).float().to(x.dtype)
        return (x,)
    return hook


@torch.no_grad()
def fwd_resid(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return F.rms_norm(x, (D,))


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    FR = cl.fineweb_rows(96, skip=80)[:, :T + 1].contiguous()
    CAPSET = torch.zeros(50257, dtype=torch.bool)
    for t in range(50257):
        if re.match(r'^ [A-Z]', ENC.decode([t])):
            CAPSET[t] = True
    hooks = [H[L].attn.c_proj.register_forward_pre_hook(mk_hook(L))
             for L in ENSEMBLE]

    deltas = []
    for i in range(0, 96, 8):
        bb = FR[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous(); tg = bb[:, 1:].to(DEV)
        HOOK['on'] = False
        r0 = fwd_resid(idx)
        HOOK['on'] = True
        r1 = fwd_resid(idx)
        HOOK['on'] = False
        pm = CAPSET.to(DEV)[tg]
        pm[:, :64] = False
        d = (r0 - r1)[pm]
        if d.shape[0]:
            deltas.append(d[::3].float().cpu())
    for hk in hooks:
        hk.remove()
    DL = torch.cat(deltas).to(DEV)
    U_, S_, V_ = torch.svd_lowrank(DL, q=48, niter=4)
    P = V_[:, :32]
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    s2 = ((WU @ P).norm(dim=1) / WU.norm(dim=1).clamp_min(1e-6)).cpu()
    print(f"delta subspace from {DL.shape[0]} positions", flush=True)

    z = np.load(PT + 'circuit_cap_fn_tokens.npz')
    toks = torch.tensor(z['tokens']); rise = torch.tensor(z['rise'])
    sv = s2[toks]
    rs = torch.argsort(torch.argsort(sv)).float()
    rr = torch.argsort(torch.argsort(rise)).float()
    n = len(toks)
    rho = 1 - 6 * float(((rs - rr) ** 2).sum()) / max(n * (n * n - 1), 1)
    q75 = float(sv.quantile(0.75))
    top_dmg = rise.argsort(descending=True)[:50]
    fn = float((sv[top_dmg] < q75).float().mean())
    med_r = float(rise.median())
    top_sc = sv.argsort(descending=True)[:50]
    fp = float((rise[top_sc] <= med_r).float().mean())

    pa = rho >= 0.80
    pb = fn <= 0.45
    pc = fp <= 0.30
    out = {'spearman': round(rho, 3), 'fn': round(fn, 3), 'fp': round(fp, 3),
           'direct_ref': {'spearman': 0.739, 'fn': 0.62, 'fp': 0.26},
           'pred_a_rho_80': bool(pa), 'pred_b_fn_45': bool(pb),
           'pred_c_fp_30': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(out)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
