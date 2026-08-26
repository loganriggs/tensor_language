# deep_trained: A TRAINED SMALL-BILINEAR REPLACEMENT CLASS FOR THE DEEP MLPS (the
# joint replacement model's remaining ~1.6 CE lives in mlps 4-13; five fitted
# quadratic bases all failed ~.45-.52. New move = the S1489 lesson at full module
# grain: TRAIN a rank-64 bilinear replacement — stand(z) = Down64((L64 z) * (R64 z))
# + b, ~3.5 Mbit — against full-model CE with the target MLP replaced, everything
# else live. Targets: mlp7, mlp8, mlp9, mlp14. Adam 3e-3, 400 steps, batch 8, 480
# rows; init from the module's own top-64 SVD factors. Held-out fid vs frozen
# anchors, NR=960.
#
# Registered predictions:
#   pred_a trained mlp14 recovery >= .55 (fitted methods stall at .35).
#   pred_b trained mlp7 >= .60 (fitted .52).
#   pred_c trained beats the linall ridge by >= .10 recovery at all four targets.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'deep_trained_results.json'
NR = 960; NTR = 480; STEPS = 400; RK = 64
H = m.transformer.h
TARGETS = [7, 8, 9, 14]
LINALL_REF = {7: 0.4595, 8: 0.4568, 9: 0.4819, 14: 0.3401}
STAND = {'L': None, 'mod': None}


def stand_hook_for(L):
    def hook(mod, args, output):
        if STAND['L'] == L:
            z = args[0]
            sm = STAND['mod']
            return (sm['dn']((sm['l'](z).float() * sm['r'](z).float())
                            .to(sm['dn'].weight.dtype)) + sm['b']).to(output.dtype)
        return None
    return hook


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    for p in m.parameters():
        p.requires_grad_(False)
    TRR = cl.fineweb_rows(NTR, skip=80)[:, :T + 1].contiguous()
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()
    sw = json.load(open(PT + 'optimal_ablation_all_results.json'))['results']
    hooks = [H[L].mlp.register_forward_hook(stand_hook_for(L)) for L in TARGETS]

    def ce_eval():
        s_ = 0.0; n_ = 0
        with torch.no_grad():
            for i in range(0, NR, 8):
                bb = EVR[i:i + 8].to(DEV)
                idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
                lo = fwd(idx).float()
                ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                     reduction='none').view(tg.shape)
                mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
                s_ += float(ce[mk].sum()); n_ += int(mk.sum())
        return s_ / max(n_, 1)

    STAND['L'] = None
    clean = ce_eval()
    print(f"clean {clean:.4f}", flush=True)
    res = {'clean': round(clean, 4)}
    fids = {}; recs = {}
    for L in TARGETS:
        blk = H[L]
        with torch.no_grad():
            Ul, Sl, Vl = torch.svd_lowrank(blk.mlp.Left.weight.float(), q=RK + 16,
                                           niter=4)
            Ur, Sr, Vr = torch.svd_lowrank(blk.mlp.Right.weight.float(), q=RK + 16,
                                           niter=4)
        lm_ = torch.nn.Linear(D, RK, bias=False).to(DEV)
        rm_ = torch.nn.Linear(D, RK, bias=False).to(DEV)
        dn_ = torch.nn.Linear(RK, D, bias=False).to(DEV)
        lm_.weight.data = (Sl[:RK].sqrt().unsqueeze(1) * Vl[:, :RK].T).to(DEV)
        rm_.weight.data = (Sr[:RK].sqrt().unsqueeze(1) * Vr[:, :RK].T).to(DEV)
        dn_.weight.data = torch.zeros(D, RK, device=DEV)
        b_ = torch.nn.Parameter(blk.mlp.Down_bias.detach().float().clone())
        params = list(lm_.parameters()) + list(rm_.parameters()) \
            + list(dn_.parameters()) + [b_]
        opt = torch.optim.Adam(params, lr=3e-3)
        g = torch.Generator().manual_seed(3 + L)
        STAND['L'] = L
        STAND['mod'] = {'l': lm_, 'r': rm_, 'dn': dn_, 'b': b_}
        for step in range(STEPS):
            sel = torch.randint(0, NTR, (8,), generator=g)
            bb = TRR[sel].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            loss = ce[:, 64:].mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 100 == 0:
                print(f"  L{L} step {step} loss {float(loss):.4f}", flush=True)
        ce_tr = ce_eval()
        STAND['L'] = None; STAND['mod'] = None
        a = sw[f'mlp{L}']
        fids[L] = (a['ce_opt'] - ce_tr) / max(a['ce_opt'] - clean, 1e-6)
        recs[L] = (a['ce_mean'] - ce_tr) / max(a['ce_mean'] - clean, 1e-6)
        res[f'mlp{L}'] = {'ce': round(ce_tr, 4), 'fid_opt': round(fids[L], 4),
                          'recovery': round(recs[L], 4),
                          'linall_ref': LINALL_REF[L]}
        print(f"mlp{L}: {res[f'mlp{L}']}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    for hk in hooks:
        hk.remove()

    pa = recs[14] >= 0.55
    pb = recs[7] >= 0.60
    pc = all(recs[L] >= LINALL_REF[L] + 0.10 for L in TARGETS)
    out = {'res': res, 'mbits_per_module': round((3 * RK * D + D) * 16 / 1e6, 2),
           'data_budget': {'steps': STEPS, 'batch': 8, 'lr': 3e-3, 'rows': NTR},
           'pred_a_mlp14_55': bool(pa), 'pred_b_mlp7_60': bool(pb),
           'pred_c_beats_linall_10': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
