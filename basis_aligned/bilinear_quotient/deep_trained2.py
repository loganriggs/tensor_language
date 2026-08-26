# deep_trained2: THE RETRY WITH A FAITHFUL INIT (S1532: zero-Down init never
# escaped bias-only output). Init = the module's OWN TOP-64 HIDDEN UNITS (ranked by
# std(h) x ||Down column||, measured on 96 rows) — an exact sub-MLP restricted to
# those units, itself a describable class. Arms per target: zero-shot sub-MLP fid,
# then CE-finetuned (300 steps, Adam 1e-3 — lower lr, warm start). Targets mlp7/8/
# 9/14, held-out NR=960, frozen anchors.
#
# Registered predictions:
#   pred_a zero-shot top-64-unit sub-MLP >= .30 recovery for >= 3 of 4 targets.
#   pred_b CE-finetuning adds >= .10 recovery at all four.
#   pred_c trained mlp7 >= .55.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'deep_trained2_results.json'
NR = 960; NTR = 480; STEPS = 300; RK = 64
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
    fids = {}; recs = {}; zs_recs = {}
    # hidden std for unit ranking (96 rows)
    HR = cl.fineweb_rows(96, skip=80)[:, :T + 1].contiguous()
    hstd = {}
    with torch.no_grad():
        acc1 = {L: 0 for L in TARGETS}; acc2 = {L: 0 for L in TARGETS}; n0 = 0
        zs = {L: [] for L in TARGETS}
        pre_hooks = []
        store = {}
        def mk_pre(L):
            def hk(mod, args):
                store.setdefault(L, []).append(args[0].detach())
                return None
            return hk
        for L in TARGETS:
            pre_hooks.append(H[L].mlp.register_forward_pre_hook(mk_pre(L)))
        for i in range(0, 96, 8):
            store.clear()
            fwd(HR[i:i + 8, :-1].to(DEV).contiguous())
            for L in TARGETS:
                z = store[L][0]
                h = (H[L].mlp.Left(z).float() * H[L].mlp.Right(z).float()) \
                    .reshape(-1, H[L].mlp.Left.weight.shape[0])
                acc1[L] = acc1[L] + h.sum(0)
                acc2[L] = acc2[L] + (h * h).sum(0)
            n0 += 8 * T
        for hk in pre_hooks:
            hk.remove()
        for L in TARGETS:
            mu = acc1[L] / n0
            hstd[L] = (acc2[L] / n0 - mu * mu).clamp_min(0).sqrt()
    print("unit stats done", flush=True)

    for L in TARGETS:
        blk = H[L]
        with torch.no_grad():
            score = hstd[L] * blk.mlp.Down.weight.float().norm(dim=0)
            topu = score.argsort(descending=True)[:RK]
        lm_ = torch.nn.Linear(D, RK, bias=False).to(DEV)
        rm_ = torch.nn.Linear(D, RK, bias=False).to(DEV)
        dn_ = torch.nn.Linear(RK, D, bias=False).to(DEV)
        lm_.weight.data = blk.mlp.Left.weight.float()[topu].clone()
        rm_.weight.data = blk.mlp.Right.weight.float()[topu].clone()
        dn_.weight.data = blk.mlp.Down.weight.float()[:, topu].clone()
        b_ = torch.nn.Parameter(blk.mlp.Down_bias.detach().float().clone())
        STAND['L'] = L
        STAND['mod'] = {'l': lm_, 'r': rm_, 'dn': dn_, 'b': b_}
        ce_zs = ce_eval()
        a = sw[f'mlp{L}']
        zs_recs[L] = (a['ce_mean'] - ce_zs) / max(a['ce_mean'] - clean, 1e-6)
        print(f"mlp{L} zero-shot ce {ce_zs:.4f} rec {zs_recs[L]:.4f}", flush=True)

        params = list(lm_.parameters()) + list(rm_.parameters()) \
            + list(dn_.parameters()) + [b_]
        opt = torch.optim.Adam(params, lr=1e-3)
        g = torch.Generator().manual_seed(3 + L)
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
        fids[L] = (a['ce_opt'] - ce_tr) / max(a['ce_opt'] - clean, 1e-6)
        recs[L] = (a['ce_mean'] - ce_tr) / max(a['ce_mean'] - clean, 1e-6)
        res[f'mlp{L}'] = {'ce_zeroshot': round(ce_zs, 4),
                          'rec_zeroshot': round(zs_recs[L], 4),
                          'ce_trained': round(ce_tr, 4),
                          'fid_opt': round(fids[L], 4),
                          'rec_trained': round(recs[L], 4),
                          'linall_ref': LINALL_REF[L]}
        print(f"mlp{L}: {res[f'mlp{L}']}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    for hk in hooks:
        hk.remove()

    pa = sum(1 for L in TARGETS if zs_recs[L] >= 0.30) >= 3
    pb = all(recs[L] >= zs_recs[L] + 0.10 for L in TARGETS)
    pc = recs[7] >= 0.55
    out = {'res': res, 'mbits_per_module': round((3 * RK * D + D) * 16 / 1e6, 2),
           'data_budget': {'steps': STEPS, 'batch': 8, 'lr': 1e-3, 'rows': NTR},
           'pred_a_zeroshot_30_3of4': bool(pa), 'pred_b_train_adds_10': bool(pb),
           'pred_c_mlp7_55': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
