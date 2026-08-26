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
# circuit_h55: PROFILE THE PRIMARY MOVER (S1525: head 5.5 carries the context-
# matching for copy AND induction). Removal arms: {5.5} scored on copy / induction
# / cap_copied / cap_novel / global.
# Registered predictions:
#   pred_a 5.5's cap_copied damage >= 3x its cap_novel damage (it moves COPIED
#          content — context machinery, not capitalization machinery).
#   pred_b induction damage >= .04 (it also drives the AB...A->B pattern).
#   pred_c global <= .01.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'circuit_h55_results.json'
NR = 960
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

    ENS = {'m55': [(5, 5)]}

    def measure(hset):
        HSET['set'] = hset
        gs = 0.0; gn = 0
        cs = {k2: 0.0 for k2 in ('copy', 'induction', 'cap_copied', 'cap_novel')}
        cn = {k2: 0 for k2 in ('copy', 'induction', 'cap_copied', 'cap_novel')}
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            mm = masks_for(idx, tg)
            capm = CAPV[tg].clone(); capm[:, :64] = False
            mm['cap_copied'] = capm & mm['copy']
            mm['cap_novel'] = capm & ~mm['copy']
            del mm['novel_cap']
            lo = fwd_plain(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            anym = mm['copy'] | mm['induction']
            gs += float(ce[mk & ~anym].sum()); gn += int((mk & ~anym).sum())
            for k2 in cs:
                cs[k2] += float(ce[mm[k2]].sum()); cn[k2] += int(mm[k2].sum())
        HSET['set'] = []
        return gs / max(gn, 1), {k2: cs[k2] / max(cn[k2], 1) for k2 in cs}

    g0, c0 = measure([])
    res = {'clean': {'global': round(g0, 4),
                     **{k2: round(v, 4) for k2, v in c0.items()}}}
    rises = {}
    for nm, ens in ENS.items():
        g1, c1 = measure(ens)
        rises[nm] = {'global': round(g1 - g0, 4),
                     **{k2: round(c1[k2] - c0[k2], 4) for k2 in c1}}
        res[nm] = rises[nm]
        print(nm, rises[nm], flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    pa = rises['m55']['cap_copied'] >= 3 * max(rises['m55']['cap_novel'], 1e-6)
    pb = rises['m55']['induction'] >= 0.04
    pc = rises['m55']['global'] <= 0.01
    out = {'res': res, 'pred_a_copied_3x': bool(pa),
           'pred_b_induction_04': bool(pb), 'pred_c_global_le_01': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
