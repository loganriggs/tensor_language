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
# circuit_copy_verify (built on the copy2 harness): NR=1920 verification of the
# S1522 near-noise selectivity denominators + a movers-only arm.
# Arms: clean / copy_P ensemble / copy_T ensemble / movers {5.5, 10.8} only.
# Registered predictions:
#   pred_a copy_P selectivity >= 50x at NR=1920 (the 138x was not a noise artifact).
#   pred_b copy_T class rise >= .45 replicates.
#   pred_c the two movers alone produce >= .10 class rise (they carry >= half of
#          the P-ensemble's damage).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'circuit_copy_verify_results.json'
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

    ENS = {'copy_P': [(5, 5), (10, 8), (16, 3), (16, 4), (17, 1)],
           'copy_T': [(14, 4), (13, 0), (16, 3), (14, 6), (8, 3)],
           'movers': [(5, 5), (10, 8)]}

    def measure(hset):
        HSET['set'] = hset
        gs = 0.0; gn = 0; cs = 0.0; cn = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            mm = masks_for(idx, tg)
            pm = mm['copy']
            lo = fwd_plain(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            gs += float(ce[mk & ~pm].sum()); gn += int((mk & ~pm).sum())
            cs += float(ce[pm].sum()); cn += int(pm.sum())
        HSET['set'] = []
        return gs / max(gn, 1), cs / max(cn, 1), cn

    g0, c0, npos = measure([])
    print(f"clean: cls {c0:.4f} (n={npos})", flush=True)
    res = {'clean': {'global': round(g0, 4), 'cls': round(c0, 4),
                     'n_class_positions': npos}}
    for nm, ens in ENS.items():
        g1, c1, _ = measure(ens)
        sel = (c1 - c0) / max(g1 - g0, 1e-6)
        res[nm] = {'rise_class': round(c1 - c0, 4),
                   'rise_global': round(g1 - g0, 4), 'selectivity': round(sel, 2)}
        print(nm, res[nm], flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    pa = res['copy_P']['selectivity'] >= 50
    pb = res['copy_T']['rise_class'] >= 0.45
    pc = res['movers']['rise_class'] >= 0.10
    out = {'res': res, 'pred_a_P_50x': bool(pa), 'pred_b_T_45': bool(pb),
           'pred_c_movers_10': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
