# fitted_payload: the §1294 logged route to per-head accounting — since no WEIGHTS-derived
# subspace is "the induction part" of a fetcher (cuts 1-4), fit it EMPIRICALLY: PCA the
# fetchers' (8.3+8.4) c_proj-input slices AT INDUCTION TARGETS on fit rows, then mask that
# rank-r subspace of the head's output everywhere and measure. Rank sweep r in {1,2,4,8,
# 16,32} answers "how many dimensions is the fetcher's task-part?" with a dose-response
# curve. Null: same-rank PCA fitted at random positions. Anchor: whole-slice mean ablation.
#
# Registered predictions:
#   pred_a FITTED r=16 IS MOST OF THE PART: >= 60% of the whole-slice induction damage
#          with elsewhere <= 40% of whole-slice elsewhere.
#   pred_b NULL GAP: random-position-fitted r=16 captures <= 30% of the fitted version's
#          induction damage.
#   pred_c LOW ELBOW: r=8 already reaches >= 80% of r=32's induction damage (the payload
#          READ OUT at targets is much lower-rank than the full identity code — §1292's
#          full-rank verdict applies to the code, this tests the delivered slice).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'fitted_payload_results.json'
NFIT = 96; NR = 192; W = 128; L8 = 8
HEADS = (3, 4)
RANKS = (1, 2, 4, 8, 16, 32)
H = m.transformer.h


@torch.no_grad()
def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


def induction_mask(toks, tgt):
    TGT = torch.zeros_like(toks, dtype=torch.bool)
    for b0 in range(0, toks.shape[0], 64):
        tb = toks[b0:b0 + 64]; gb = tgt[b0:b0 + 64]
        eq = (tb.unsqueeze(1) == tb.unsqueeze(2)) & (gb.unsqueeze(1) == gb.unsqueeze(2))
        q_i = torch.arange(T).view(1, T, 1); p_i = torch.arange(T).view(1, 1, T)
        band = (q_i < p_i) & (q_i >= p_i - W)
        TGT[b0:b0 + 64] = (eq & band).any(1)
    TGT[:, :16] = False
    return TGT


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NFIT + NR)[:, :T + 1].contiguous()
    FITR, EVR = ROWS[:NFIT], ROWS[NFIT:]
    toks_f = FITR[:, :-1]; tgt_f = FITR[:, 1:]
    TGT_F = induction_mask(toks_f, tgt_f)
    toks_e = EVR[:, :-1]; tgt_e = EVR[:, 1:]
    TGT = induction_mask(toks_e, tgt_e)
    ELSE = ~TGT; ELSE[:, :16] = False
    print(f"fit targets {int(TGT_F.sum())} | eval targets {int(TGT.sum())}", flush=True)

    # capture 8.3+8.4 c_proj-input slices on fit rows
    caps = []
    hk = H[L8].attn.c_proj.register_forward_pre_hook(
        lambda mod, args: caps.append(args[0].detach().float().cpu()))
    for i in range(0, NFIT, 4):
        fwd(FITR[i:i + 4, :-1].to(DEV).contiguous())
    hk.remove()
    Y = torch.cat(caps, 0)                                     # (NFIT, T, D)
    seg = torch.cat([Y[:, :, h * 128:(h + 1) * 128] for h in HEADS], -1)  # (NFIT,T,256)
    A_fit = seg[TGT_F]                                         # (n_t, 256)
    g = torch.Generator().manual_seed(29)
    RANDF = torch.zeros_like(TGT_F)
    flat = torch.randperm(TGT_F.numel(), generator=g)[:int(TGT_F.sum())]
    RANDF.view(-1)[flat] = True
    RANDF[:, :16] = False
    A_rnd = seg[RANDF]
    mu_fit = A_fit.mean(0); mu_rnd = A_rnd.mean(0)
    _, _, V_fit = torch.linalg.svd(A_fit - mu_fit, full_matrices=False)
    _, _, V_rnd = torch.linalg.svd(A_rnd - mu_rnd, full_matrices=False)
    print("bases fitted", flush=True)

    ymean_all = Y.mean((0, 1))                                 # for whole-slice anchor

    MODE = {'kind': None, 'B': None}

    def hook(mod, args):
        if MODE['kind'] is None:
            return args
        y = args[0].clone()
        s3 = y[:, :, HEADS[0] * 128:(HEADS[0] + 1) * 128]
        s4 = y[:, :, HEADS[1] * 128:(HEADS[1] + 1) * 128]
        if MODE['kind'] == 'whole':
            y[:, :, HEADS[0] * 128:(HEADS[0] + 1) * 128] = ymean_all[HEADS[0] * 128:(HEADS[0] + 1) * 128].to(y.dtype).to(y.device)
            y[:, :, HEADS[1] * 128:(HEADS[1] + 1) * 128] = ymean_all[HEADS[1] * 128:(HEADS[1] + 1) * 128].to(y.dtype).to(y.device)
            return (y,)
        B = MODE['B'].to(y.device)                             # (r, 256)
        cat = torch.cat([s3, s4], -1).float()                  # (B,T,256)
        proj = torch.einsum('btd,rd->btr', cat, B)
        comp = torch.einsum('btr,rd->btd', proj, B).to(y.dtype)
        cat2 = (cat.to(y.dtype) - comp)
        y[:, :, HEADS[0] * 128:(HEADS[0] + 1) * 128] = cat2[:, :, :128]
        y[:, :, HEADS[1] * 128:(HEADS[1] + 1) * 128] = cat2[:, :, 128:]
        return (y,)

    hk = H[L8].attn.c_proj.register_forward_pre_hook(hook)

    def ce_sets():
        tots = {'t': 0.0, 'e': 0.0}; ns = {'t': 0, 'e': 0}
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV); idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            lse = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1), reduction='none').view(tg.shape)
            for name, mask in (('t', TGT), ('e', ELSE)):
                mm = mask[i:i + 8].to(DEV)
                tots[name] += float(lse[mm].sum()); ns[name] += int(mm.sum())
        return {k: tots[k] / max(ns[k], 1) for k in tots}

    MODE['kind'] = None; base = ce_sets()
    MODE['kind'] = 'whole'; anchor = ce_sets()
    aw_t = anchor['t'] - base['t']; aw_e = anchor['e'] - base['e']
    print(f"base {base} | whole-slice dmg ind {aw_t:.4f} else {aw_e:.4f}", flush=True)
    res = {'whole': {'d_ind': round(aw_t, 4), 'd_else': round(aw_e, 4)}}
    for tag, V in (('fit', V_fit), ('rnd', V_rnd)):
        for r in RANKS:
            MODE['kind'] = 'sub'; MODE['B'] = V[:r]
            rr = ce_sets()
            key = f'{tag}_r{r}'
            res[key] = {'d_ind': round(rr['t'] - base['t'], 4), 'd_else': round(rr['e'] - base['e'], 4)}
            print(f"{key}: ind {res[key]['d_ind']} else {res[key]['d_else']}", flush=True)
    hk.remove()

    f16, n16, f8, f32 = res['fit_r16'], res['rnd_r16'], res['fit_r8'], res['fit_r32']
    pa = f16['d_ind'] >= 0.6 * aw_t and f16['d_else'] <= 0.4 * max(aw_e, 1e-4)
    pb = n16['d_ind'] <= 0.3 * max(f16['d_ind'], 1e-4)
    pc = f8['d_ind'] >= 0.8 * max(f32['d_ind'], 1e-4)
    out = {'n_fit': NFIT, 'n_rows': NR, 'base': {k: round(v, 4) for k, v in base.items()},
           'conds': res,
           'pred_a_fitted16': bool(pa), 'pred_b_null_gap': bool(pb), 'pred_c_low_elbow': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a fitted16 {pa} | pred_b null {pb} | pred_c elbow {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
