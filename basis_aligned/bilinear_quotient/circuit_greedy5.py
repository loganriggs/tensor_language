# circuit_greedy5: THE RECIPE ON SCREEN-4'S STRONGEST (pronouns 41.5x, however
# 12.6x, open_paren 5.1x, hyphen 7.2x at top-5).
# lesson: top-k by weight score dilutes with generalists). For 4 mid-tier classes
# (digits, the, is, and): candidates = top-15 heads by the weights-only score;
# greedily add a candidate only if class-rise grows AND selectivity (class/global)
# does not drop below 2. Screening evals at NR=240; the final ensemble re-verified
# at NR=960.
#
# Registered predictions:
#   pred_a all 4 beat their screen top-5 selectivity.
#   pred_b median ensemble size <= 4.
#   pred_c pronouns >= 60x.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'circuit_greedy5_results.json'
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
CONSTS = torch.load(PT + 'opt_ablation_consts_all.pt', map_location='cpu')
HSET = {'set': []}
HOOKS = []


def mk_hook(L):
    def hook(mod, args):
        hs = [hh for (LL, hh) in HSET['set'] if LL == L]
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


def rx_masks():
    V = 50257
    def rx(pat):
        v = torch.zeros(V, dtype=torch.bool)
        for t in range(V):
            if re.match(pat, ENC.decode([t])):
                v[t] = True
        return v
    return {'pronouns': rx(r'^ (he|she|they|He|She|They)$'),
            'however': rx(r'^ (However|Meanwhile|Moreover|Furthermore|Additionally)$|^(However|Meanwhile|Moreover)$'),
            'open_paren': rx(r'^ ?\($'),
            'hyphen': rx(r'^-$|^ ?--$|^\u2013$|^\u2014$| ?\u2014$')}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(960, skip=7000)[:, :T + 1].contiguous()
    SCR = ROWS[:240]
    CLS = rx_masks()
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    for L in range(18):
        HOOKS.append(H[L].attn.c_proj.register_forward_pre_hook(mk_hook(L)))

    def measure(rows, mask_v):
        gs = 0.0; gn = 0; cs = 0.0; cn = 0
        for i in range(0, rows.shape[0], 8):
            bb = rows[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            cm = mask_v.to(DEV)[tg] & mk
            gs += float(ce[mk & ~cm].sum()); gn += int((mk & ~cm).sum())
            cs += float(ce[cm].sum()); cn += int(cm.sum())
        return gs / max(gn, 1), cs / max(cn, 1)

    TOP5_VERIFIED = {'pronouns': 41.45, 'however': 12.55, 'open_paren': 5.08,
                     'hyphen': 7.19}
    res = {}
    sizes = []
    for cname, v in CLS.items():
        u = WU[v.to(DEV)].mean(0); u = u / u.norm()
        sc = torch.zeros(18, 9)
        for L in range(18):
            W = H[L].attn.c_proj.weight.float().to(DEV)
            for hh in range(9):
                sc[L, hh] = float((u @ W[:, hh * 128:(hh + 1) * 128]).norm())
        cand = [(int(i) // 9, int(i) % 9)
                for i in sc.flatten().argsort(descending=True)[:15]]
        HSET['set'] = []
        g0, c0 = measure(SCR, v)
        best = {'set': [], 'cls_rise': 0.0, 'sel': 0.0}
        cur = []
        for hd in cand:
            trial = cur + [hd]
            HSET['set'] = trial
            g1, c1 = measure(SCR, v)
            dr_c = c1 - c0; dr_g = g1 - g0
            sel = dr_c / max(dr_g, 1e-6)
            ok_sel = (not cur) or sel >= 0.9 * best['sel']
            ok_gain = dr_c >= 1.15 * best['cls_rise'] if cur else dr_c > 0.005
            if ok_sel and ok_gain and sel >= 2.0:
                cur = trial
                best = {'set': list(trial), 'cls_rise': dr_c, 'sel': sel}
            HSET['set'] = []
        # verify at full NR
        HSET['set'] = best['set']
        G0, C0 = 0, 0
        HSET['set'] = []
        G0, C0 = measure(ROWS, v)
        HSET['set'] = best['set']
        G1, C1 = measure(ROWS, v)
        HSET['set'] = []
        selv = (C1 - C0) / max(G1 - G0, 1e-6)
        res[cname] = {'heads': [f'{L}.{h}' for L, h in best['set']],
                      'size': len(best['set']),
                      'screen_sel': round(best['sel'], 2),
                      'verified_sel': round(selv, 2),
                      'verified_cls_rise': round(C1 - C0, 4),
                      'top5_ref_sel': TOP5_VERIFIED[cname]}
        sizes.append(len(best['set']))
        print(cname, res[cname], flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    for hk in HOOKS:
        hk.remove()

    import statistics
    wins = sum(1 for cn in res if res[cn]['verified_sel'] > res[cn]['top5_ref_sel'])
    pa = wins >= 4
    pb = statistics.median(sizes) <= 4
    pc = res['pronouns']['verified_sel'] >= 60
    out = {'res': res, 'wins_vs_top5': wins, 'median_size': statistics.median(sizes),
           'pred_a_all4_beat': bool(pa), 'pred_b_size_le_4': bool(pb),
           'pred_c_pronouns_60x': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"wins {wins} sizes {sizes}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
