# circuit_verify_mid_b: NR=1920 VERIFICATION OF THE MID-TIER REGISTRY (S1530-31
# hardened the >50x tier; this completes the registry: every claim large-row
# verified). Heads PARSED from the source results jsons (S1530 lesson). Part b.
#
# Registered predictions:
#   pred_a every ensemble retains >= half its claimed selectivity.
#   pred_b class-rise replicates within 35% at every circuit.
#   pred_c selectivity ordering preserved (Spearman >= .7).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'circuit_verify_mid_b_results.json'
NR = 1920
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
CONSTS = torch.load(PT + 'opt_ablation_consts_all.pt', map_location='cpu')
HSET = {'set': []}


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


def rx(pat):
    v = torch.zeros(50257, dtype=torch.bool)
    for t in range(50257):
        if re.match(pat, ENC.decode([t])):
            v[t] = True
    return v


def parse(hs):
    return [(int(s.split('.')[0]), int(s.split('.')[1])) for s in hs]


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()
    g2 = json.load(open(PT + 'circuit_greedy2_results.json'))['res']
    g3 = json.load(open(PT + 'circuit_greedy3_results.json'))['res']
    nc = json.load(open(PT + 'circuit_novelcap_results.json'))
    MASKS = {
        'digits': rx(r'^ ?[0-9]+$'), 'the': rx(r'^ the$| The$|^The$'),
        'is': rx(r'^ is$| was$| are$'), 'and': rx(r'^ and$|^ or$|^ but$'),
        'close_quote': rx(r'^["\u201d]$|^ ?"$'),
        'dollar': rx(r'^ ?[$\u00a3\u20ac]$'),
        'said': rx(r'^ (said|says|told|asked|replied)$'),
        'months': rx(r'^ (January|February|March|April|May|June|July|August|September|October|November|December)$'),
    }
    SPECS = {}
    for cn in ['close_quote', 'dollar', 'said', 'months']:
        if cn in g2:
            SPECS[cn] = {'heads': parse(g2[cn]['heads']),
                         'ref': g2[cn]['verified_sel'],
                         'ref_rise': g2[cn]['verified_cls_rise'],
                         'mask': MASKS[cn]}
        elif cn in g3:
            SPECS[cn] = {'heads': parse(g3[cn]['heads']),
                         'ref': g3[cn]['verified_sel'],
                         'ref_rise': g3[cn]['verified_cls_rise'],
                         'mask': MASKS[cn]}
    hooks = [H[L].attn.c_proj.register_forward_pre_hook(mk_hook(L))
             for L in range(18)]

    def measure(hset, mask_v):
        HSET['set'] = hset
        gs = 0.0; gn = 0; cs = 0.0; cn_ = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            cm = mask_v.to(DEV)[tg] & mk
            gs += float(ce[mk & ~cm].sum()); gn += int((mk & ~cm).sum())
            cs += float(ce[cm].sum()); cn_ += int(cm.sum())
        HSET['set'] = []
        return gs / max(gn, 1), cs / max(cn_, 1)

    res = {}
    for cn, spec in SPECS.items():
        g0, c0 = measure([], spec['mask'])
        g1, c1 = measure(spec['heads'], spec['mask'])
        sel = (c1 - c0) / max(g1 - g0, 1e-6)
        res[cn] = {'sel_1920': round(sel, 2), 'sel_ref': spec['ref'],
                   'rise_1920': round(c1 - c0, 4), 'rise_ref': spec['ref_rise']}
        print(cn, res[cn], flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    for hk in hooks:
        hk.remove()

    ok_a = all(res[cn]['sel_1920'] >= 0.5 * res[cn]['sel_ref'] for cn in res)
    ok_b = all(abs(res[cn]['rise_1920'] - res[cn]['rise_ref'])
               <= 0.35 * max(res[cn]['rise_ref'], 1e-6) for cn in res)
    sels = [res[cn]['sel_1920'] for cn in res]
    refs = [res[cn]['sel_ref'] for cn in res]
    def ranks(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r_ = [0] * len(v)
        for i, j in enumerate(s):
            r_[j] = i
        return r_
    ra, rb = ranks(sels), ranks(refs)
    n = len(sels)
    rho = 1 - 6 * sum((ra[i] - rb[i]) ** 2 for i in range(n)) \
        / max(n * (n * n - 1), 1)
    pa = ok_a
    pb = ok_b
    pc = rho >= 0.7
    out = {'res': res, 'ordering_spearman': round(rho, 3),
           'pred_a_half_retained': bool(pa), 'pred_b_rise_35': bool(pb),
           'pred_c_ordering_7': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
