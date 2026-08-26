# deep_units_b: THE UNIT-TRUNCATION CURVE (S1533: top-64 units recover .30-.42
# zero-shot at mlp7/8/9 — unit-grain concentration in the module's own basis, which
# every rotated-basis method missed). Zero-shot sub-MLPs at K in {64, 256, 1024}
# for TARGETS below (no training — S1533 showed CE-finetuning harms warm starts).
# Unit ranking: std(h) x ||Down column||, 96 rows. Frozen anchors, NR=960.
#
# Registered predictions:
#   pred_a median recovery at K=256 (5.6% of units) >= .55 across this part's targets.
#   pred_b median at K=1024 >= .75.
#   pred_c every target >= .20 at K=64 (concentration is universal in the deep stack).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'deep_units_b_results.json'
NR = 960; NTR = 480; STEPS = 300; RK = 64
H = m.transformer.h
TARGETS = [11, 12, 13, 14, 15, 16, 17]
LINALL_REF = {}
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

    KS = (64, 256, 1024)
    RECS = {}
    for L in TARGETS:
        blk = H[L]
        with torch.no_grad():
            score = hstd[L] * blk.mlp.Down.weight.float().norm(dim=0)
            order = score.argsort(descending=True)
        a = sw[f'mlp{L}']
        RECS[L] = {}
        for K in KS:
            topu = order[:K]
            lm_ = torch.nn.Linear(D, K, bias=False).to(DEV)
            rm_ = torch.nn.Linear(D, K, bias=False).to(DEV)
            dn_ = torch.nn.Linear(K, D, bias=False).to(DEV)
            lm_.weight.data = blk.mlp.Left.weight.float()[topu].clone()
            rm_.weight.data = blk.mlp.Right.weight.float()[topu].clone()
            dn_.weight.data = blk.mlp.Down.weight.float()[:, topu].clone()
            b_ = blk.mlp.Down_bias.detach().float().clone()
            STAND['L'] = L
            STAND['mod'] = {'l': lm_, 'r': rm_, 'dn': dn_, 'b': b_}
            ce_zs = ce_eval()
            STAND['L'] = None; STAND['mod'] = None
            rec = (a['ce_mean'] - ce_zs) / max(a['ce_mean'] - clean, 1e-6)
            RECS[L][K] = round(rec, 4)
            res[f'mlp{L}_K{K}'] = {'ce': round(ce_zs, 4), 'rec': RECS[L][K]}
            print(f"mlp{L} K={K}: rec {rec:.4f}", flush=True)
            json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    import statistics
    med256 = statistics.median([RECS[L][256] for L in TARGETS])
    med1024 = statistics.median([RECS[L][1024] for L in TARGETS])
    pa = med256 >= 0.55
    pb = med1024 >= 0.75
    pc = all(RECS[L][64] >= 0.20 for L in TARGETS)
    out = {'res': res, 'recs': {str(L): RECS[L] for L in TARGETS},
           'median_256': round(med256, 4), 'median_1024': round(med1024, 4),
           'pred_a_med256_55': bool(pa), 'pred_b_med1024_75': bool(pb),
           'pred_c_all_K64_20': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
