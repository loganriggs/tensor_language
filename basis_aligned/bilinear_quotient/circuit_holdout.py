# circuit_holdout: HELD-OUT CLASS-HALF GENERALIZATION (S1517: per-token grading is
# vacuous for closed classes; the right test for them is member-level holdout).
# For six circuits: re-select the ensemble using only HALF the class's tokens
# (mask A: even-indexed members) — candidates by the weights-only score of mask-A's
# mean unembedding, greedy top-4 — then measure removal damage separately on
# mask A (fit half), mask B (HELD-OUT half), and frequency-matched non-member
# control tokens.
#
# Registered predictions:
#   pred_a held-out-half damage >= .7x fit-half damage for >= 4 of 6 circuits.
#   pred_b control-token damage <= .3x held-out damage for >= 5 of 6.
#   pred_c the small closed classes (months, said) also generalize (>= .5x).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'circuit_holdout_results.json'
NR = 960
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


CIRCUITS = {
    'months': {'mask': rx(r'^ (January|February|March|April|May|June|July|August|September|October|November|December)$'),
               'heads': [(14, 7)]},
    'said': {'mask': rx(r'^ (said|says|told|asked|replied)$'),
             'heads': [(11, 3), (9, 1), (11, 5), (12, 2)]},
    'is': {'mask': rx(r'^ is$| was$| are$'), 'heads': [(11, 3), (15, 5)]},
    'the': {'mask': rx(r'^ the$| The$|^The$'),
            'heads': [(7, 3), (10, 8), (11, 7)]},
    'and': {'mask': rx(r'^ and$|^ or$|^ but$'),
            'heads': [(10, 5), (16, 8), (7, 3), (9, 5)]},
    'digits': {'mask': rx(r'^ ?[0-9]+$'),
               'heads': [(7, 3), (6, 5), (12, 6), (11, 5)]},
}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()
    hooks = [H[L].attn.c_proj.register_forward_pre_hook(mk_hook(L))
             for L in range(18)]
    WU = m.lm_head.weight.float().to(DEV)[:50257]

    # eval-frequency for controls
    ecnt = torch.zeros(50257)
    for i in range(0, NR, 8):
        ecnt.index_add_(0, EVR[i:i + 8, :-1].reshape(-1), torch.ones(8 * T))

    def per_token():
        tsum = torch.zeros(50257); tn = torch.zeros(50257)
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            tgf = tg.cpu().reshape(-1)
            tsum.index_add_(0, tgf, (ce * mk).cpu().reshape(-1))
            tn.index_add_(0, tgf, mk.cpu().reshape(-1).float())
        return tsum, tn

    HSET['set'] = []
    ts0, tn0 = per_token()
    print("clean measured", flush=True)

    res = {}
    for cname, spec in CIRCUITS.items():
        members = torch.nonzero(spec['mask']).flatten().tolist()
        members = [t for t in members if ecnt[t] >= 5]
        if len(members) < 4:
            res[cname] = {'skip': True, 'n_members': len(members)}
            continue
        A = members[0::2]; B = members[1::2]
        mA = torch.zeros(50257, dtype=torch.bool); mA[torch.tensor(A)] = True
        # select ensemble from mask A only (weights-only + greedy top-4 by score)
        u = WU[mA.to(DEV)].mean(0); u = u / u.norm()
        sc = torch.zeros(18, 9)
        for L in range(18):
            W = H[L].attn.c_proj.weight.float().to(DEV)
            for hh in range(9):
                sc[L, hh] = float((u @ W[:, hh * 128:(hh + 1) * 128]).norm())
        ens = [(int(i) // 9, int(i) % 9)
               for i in sc.flatten().argsort(descending=True)[:4]]
        HSET['set'] = ens
        ts1, _ = per_token()
        HSET['set'] = []

        def dmg(tok_list):
            tt = torch.tensor(tok_list)
            w = tn0[tt]
            d = (ts1[tt] - ts0[tt]) / tn0[tt].clamp_min(1)
            return float((d * w).sum() / w.sum().clamp_min(1))
        dA = dmg(A); dB = dmg(B)
        # frequency-matched controls: non-members nearest in eval count
        ctrl = []
        for t in B:
            diffs = (ecnt - ecnt[t]).abs()
            diffs[spec['mask']] = 1e9
            ctrl.append(int(diffs.argmin()))
        dC = dmg(ctrl)
        res[cname] = {'ensemble': [f'{L}.{h}' for L, h in ens],
                      'n_members': len(members),
                      'dmg_fit_half': round(dA, 4),
                      'dmg_heldout_half': round(dB, 4),
                      'dmg_controls': round(dC, 4),
                      'holdout_ratio': round(dB / max(dA, 1e-6), 3),
                      'control_ratio': round(dC / max(dB, 1e-6), 3)}
        print(cname, res[cname], flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    for hk in hooks:
        hk.remove()

    done = [cn for cn in res if 'holdout_ratio' in res[cn]]
    n_a = sum(1 for cn in done if res[cn]['holdout_ratio'] >= 0.7)
    n_b = sum(1 for cn in done if res[cn]['control_ratio'] <= 0.3)
    pa = n_a >= 4
    pb = n_b >= 5
    small = [cn for cn in ('months', 'said') if cn in done]
    pc = all(res[cn]['holdout_ratio'] >= 0.5 for cn in small) and len(small) == 2
    out = {'res': res, 'n_holdout_07': n_a, 'n_control_03': n_b,
           'pred_a_holdout_4of6': bool(pa), 'pred_b_controls_5of6': bool(pb),
           'pred_c_small_classes': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
