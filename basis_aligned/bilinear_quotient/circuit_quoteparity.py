# circuit_quoteparity: DOES THE CLOSE-QUOTE ENSEMBLE TRACK PARITY? Class split:
# positions whose target is a close-quote token, divided by whether the context
# holds an ODD number of quote characters (a close is properly pending) vs EVEN.
# Removal: the verified close_quote ensemble. If the circuit implements
# quote-STATE, damage concentrates on the proper (odd) subclass.
#
# Registered predictions:
#   pred_a clean CE is lower on the proper subclass (the model tracks parity).
#   pred_b ensemble-removal damage on proper >= 2x improper.
#   pred_c global rise <= .02.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'circuit_quoteparity_results.json'
NR = 1920
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
CONSTS = torch.load(PT + 'opt_ablation_consts_all.pt', map_location='cpu')
ENSEMBLE = [(12, 6), (13, 3), (14, 8), (10, 7), (10, 6)]
HOOK = {'on': False}


def mk_hook(L):
    def hook(mod, args):
        if not HOOK['on']:
            return None
        hs = [hh for (LL, hh) in ENSEMBLE if LL == L]
        if not hs:
            return None
        x = args[0].clone()
        for hh in hs:
            x[:, :, hh * 128:(hh + 1) * 128] = \
                CONSTS[f'head{L}.{hh}'].to(DEV).float().to(x.dtype)
        return (x,)
    return hook


@torch.no_grad()
def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()
    import re
    QMASK = torch.zeros(50257, dtype=torch.bool)
    QCOUNT = torch.zeros(50257)
    for t in range(50257):
        s = ENC.decode([t])
        nq = s.count('"') + s.count('\u201c') + s.count('\u201d')
        QCOUNT[t] = nq
        if re.match(r'^["\u201d]$|^ ?"$', s):
            QMASK[t] = True
    hooks = [H[L].attn.c_proj.register_forward_pre_hook(mk_hook(L))
             for L, _ in ENSEMBLE]

    def measure(on):
        HOOK['on'] = on
        gs = 0.0; gn = 0
        sp = 0.0; np_ = 0; si = 0.0; ni = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            gs += float(ce[mk].sum()); gn += int(mk.sum())
            qc = QCOUNT.to(DEV)[idx]
            cum = torch.cumsum(qc, dim=1)
            odd = (cum % 2) == 1
            cls = QMASK.to(DEV)[tg] & mk
            proper = cls & odd
            improper = cls & ~odd
            sp += float(ce[proper].sum()); np_ += int(proper.sum())
            si += float(ce[improper].sum()); ni += int(improper.sum())
        HOOK['on'] = False
        return gs / max(gn, 1), sp / max(np_, 1), si / max(ni, 1), np_, ni

    g0, p0, i0, npos, nim = measure(False)
    g1, p1, i1, _, _ = measure(True)
    for hk in hooks:
        hk.remove()
    dp = p1 - p0; di = i1 - i0
    pa = p0 < i0
    pb = dp >= 2 * max(di, 1e-6)
    pc = (g1 - g0) <= 0.02
    out = {'clean': {'global': round(g0, 4), 'proper': round(p0, 4),
                     'improper': round(i0, 4), 'n_proper': npos,
                     'n_improper': nim},
           'rises': {'proper': round(dp, 4), 'improper': round(di, 4),
                     'global': round(g1 - g0, 4)},
           'pred_a_model_tracks_parity': bool(pa),
           'pred_b_proper_2x': bool(pb), 'pred_c_global_le_02': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(out)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
