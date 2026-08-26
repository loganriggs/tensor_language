# circuit_screen3: CONTEXTUAL CIRCUITS (the frontier past target-token classes:
# classes defined by CONTEXT patterns). Classes per position j:
#   copy      — target occurred in the previous 64 context tokens AND is an
#               INFREQUENT token (eval-count rank > 500): content copying, not
#               function-word recurrence (S1520 fix).
#   induction — some k<j has idx[k]==idx[j] and idx[k+1]==tg[j] (AB...A -> B).
#   novel_cap — target is a capitalized word NOT present in context (anti-copy).
# Weights-only scores cannot see context, so both methods here are data-driven:
#   method T (target-attribution): head's mean contribution to the CORRECT target
#     logit at class positions (per-position direction u_{tg[j]}).
#   method P (pattern-heuristic): head's mean attention mass on context positions
#     holding the target token (copy/induction) or uniformly (novel_cap: excluded,
#     method T only).
# Top-5 per method, graded by optimal-constant removal selectivity (NR=480).
#
# circuit_verify_g5: NR=1920 VERIFICATION of greedy5's >50x claims (S1523 rule).
# Heads PARSED from circuit_greedy5_results.json.
# Registered predictions:
#   pred_a pronouns >= 100x at NR=1920 (the 275x survives at least at 100x scale).
#   pred_b open_paren >= 30x.
#   pred_c both class rises replicate within 35%.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'circuit_verify_g5_results.json'
NR = 1920
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
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
def fwd_plain(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def class_masks(idx, tg):
    """idx, tg: [B, T] on DEV. Returns dict of [B, T] bool masks."""
    B = idx.shape[0]
    copy = torch.zeros(B, T, dtype=torch.bool, device=DEV)
    induc = torch.zeros(B, T, dtype=torch.bool, device=DEV)
    for lag in range(1, 65):
        past = torch.roll(idx, lag, dims=1)
        past[:, :lag] = -1
        copy |= (past == tg)
        pastn = torch.roll(idx, lag - 1, dims=1)   # token AFTER the matched one
        pastn[:, :max(lag - 1, 1)] = -1
        induc |= (past == idx) & (pastn == tg) & (lag >= 2)
    CAPV = torch.zeros(50257, dtype=torch.bool)
    for t in range(50257):
        pass
    return {'copy': copy, 'induction': induc}


CAPV = None
FREQOK = None   # infrequent-token mask (rank > 500 by eval count)


@torch.no_grad()
def main():
    global CAPV
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()
    CAPV = torch.zeros(50257, dtype=torch.bool)
    for t in range(50257):
        if re.match(r'^ [A-Z]', ENC.decode([t])):
            CAPV[t] = True
    CAPV = CAPV.to(DEV)
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    global FREQOK
    ecnt = torch.zeros(50257)
    for i in range(0, NR, 8):
        ecnt.index_add_(0, EVR[i:i + 8, :-1].reshape(-1), torch.ones(8 * T))
    thresh = ecnt.sort(descending=True).values[500]
    FREQOK = (ecnt < thresh).to(DEV)
    hooks = [H[L].attn.c_proj.register_forward_pre_hook(mk_hook(L))
             for L in range(18)]

    def masks_for(idx, tg):
        mm = class_masks(idx, tg)
        mm['copy'] = mm['copy'] & FREQOK[tg]
        mm['novel_cap'] = CAPV[tg] & ~mm['copy']
        for k in mm:
            mm[k][:, :64] = False
        return mm

    import re as _re
    def rx(pat):
        v = torch.zeros(50257, dtype=torch.bool)
        for t in range(50257):
            if _re.match(pat, ENC.decode([t])):
                v[t] = True
        return v
    import json as _j
    g5 = _j.load(open(PT + 'circuit_greedy5_results.json'))['res']
    def parse(hs):
        return [(int(s.split('.')[0]), int(s.split('.')[1])) for s in hs]
    SPECS = {
        'pronouns': {'mask': rx(r'^ (he|she|they|He|She|They)$'),
                     'heads': parse(g5['pronouns']['heads']),
                     'ref': g5['pronouns']['verified_sel'],
                     'ref_rise': g5['pronouns']['verified_cls_rise']},
        'open_paren': {'mask': rx(r'^ ?\($'),
                       'heads': parse(g5['open_paren']['heads']),
                       'ref': g5['open_paren']['verified_sel'],
                       'ref_rise': g5['open_paren']['verified_cls_rise']},
    }

    def measure(hset, mask_v):
        HSET['set'] = hset
        gs = 0.0; gn = 0; cs = 0.0; cn = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd_plain(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            cm = mask_v.to(DEV)[tg] & mk
            gs += float(ce[mk & ~cm].sum()); gn += int((mk & ~cm).sum())
            cs += float(ce[cm].sum()); cn += int(cm.sum())
        HSET['set'] = []
        return gs / max(gn, 1), cs / max(cn, 1)

    res = {}
    for cn_, spec in SPECS.items():
        g0, c0 = measure([], spec['mask'])
        g1, c1 = measure(spec['heads'], spec['mask'])
        sel = (c1 - c0) / max(g1 - g0, 1e-6)
        res[cn_] = {'sel_1920': round(sel, 2), 'sel_960_ref': spec['ref'],
                    'rise_class': round(c1 - c0, 4),
                    'rise_global': round(g1 - g0, 4)}
        print(cn_, res[cn_], flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    pa = res['pronouns']['sel_1920'] >= 100
    pb = res['open_paren']['sel_1920'] >= 30
    pc = all(abs(res[cn]['rise_class'] - SPECS[cn]['ref_rise'])
             <= 0.35 * max(SPECS[cn]['ref_rise'], 1e-6) for cn in SPECS)
    out = {'res': res,
           'pred_a_pronouns_100': bool(pa), 'pred_b_openparen_30': bool(pb),
           'pred_c_rises_35': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
