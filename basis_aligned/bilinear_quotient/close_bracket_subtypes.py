# close_bracket_subtypes: THE USER'S REFINEMENT of §1340. The close-bracket target class
# is heterogeneous: the target token can be plain ")", a double close "))" (depth >= 2),
# or a compound ")," / ")." / ")\"" etc. where the model must place trailing punctuation
# along with the close. The user conjectures that heads BEYOND 13.8 assist on the
# compound forms even though 13.8 owns the aggregate (96.5%).
#
# Decomposition: partition targets by the TARGET TOKEN's decoded form:
#   plain    exactly ")"
#   double   contains "))"
#   comma    ")" followed by "," in the same token (")," etc.)
#   period   ")" followed by "." (").", ")." variants)
#   quote    ")" adjacent to a quote char
#   other    remaining ")"-containing tokens
# Per subtype, damage at targets from: 13.8 solo | full a13 | a13 minus 13.8 (the 8
# helpers jointly) | a12 (neighbor) | a14 (neighbor). Share_138 = dmg(13.8)/dmg(a13).
#
# Registered predictions:
#   pred_a 13.8's CORE IS THE PLAIN CLOSE: share_138 >= 0.90 on the "plain" subtype.
#   pred_b (USER CONJECTURE, registered on his behalf): at least one compound subtype
#          (double/comma/period/quote) has share_138 <= plain's share − 0.15 — helpers
#          carry a real slice of the compound job.
#   pred_c THE DOUBLE CLOSE IS MOST HELPER-DEPENDENT: "))" has the LOWEST share_138 of
#          all subtypes with n >= 30 (two-level depth needs the most extra machinery).
# Diagnostics per subtype: n, base CE, and the neighbor-layer damages (a12/a14) —
# if a compound subtype's helper slice sits OUTSIDE L13, the neighbors will show it.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'close_bracket_subtypes_results.json'
NMEAN = 24; NR = 1920; L13 = 13
H = m.transformer.h
CUR = {'mode': None, 'heads': None, 'hmean': None, 'lmean': None, 'layer': None}


def cproj_hook_13(mod, args):
    """Head-level ablation inside L13 (modes 'heads')."""
    if CUR['mode'] != 'heads':
        return None
    y = args[0].clone()
    for h in CUR['heads']:
        y[..., h * 128:(h + 1) * 128] = CUR['hmean'][h].to(y.dtype)
    return (y,)


def mk_layer_hook(L):
    def hook(mod, args, out):
        if CUR['mode'] == 'layer' and CUR['layer'] == L:
            y = out[0] if isinstance(out, tuple) else out
            rep = CUR['lmean'][L].to(y.dtype).expand_as(y)
            return (rep,) + tuple(out[1:]) if isinstance(out, tuple) else rep
        return out
    return hook


@torch.no_grad()
def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    close_t = set(); open_t = set()
    sub_of = {}
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        if '(' in d:
            open_t.add(tok)
        if ')' not in d:
            continue
        close_t.add(tok)
        ds = d
        if ds.strip() == ')':
            sub_of[tok] = 'plain'
        elif '))' in ds:
            sub_of[tok] = 'double'
        elif '),' in ds:
            sub_of[tok] = 'comma'
        elif ').' in ds:
            sub_of[tok] = 'period'
        elif ')"' in ds or ")'" in ds or '")' in ds or "')" in ds:
            sub_of[tok] = 'quote'
        else:
            sub_of[tok] = 'other'
    close_ids = torch.tensor(sorted(close_t)); open_ids = torch.tensor(sorted(open_t))
    SUBS = ('plain', 'double', 'comma', 'period', 'quote', 'other')

    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]

    # means: per-head at L13 c_proj input; per-layer outputs for a12/a13/a14
    capsH = []
    capsL = {L: [] for L in (12, 13, 14)}
    hs = [H[L13].attn.c_proj.register_forward_pre_hook(
        lambda mod, args: capsH.append(args[0].detach().float().reshape(-1, 9, 128).mean(0)))]
    for L in (12, 13, 14):
        def mk(L):
            def h(mod, args, out):
                y = out[0] if isinstance(out, tuple) else out
                capsL[L].append(y.detach().float().mean((0, 1)))
                return out
            return h
        hs.append(H[L].attn.register_forward_hook(mk(L)))
    CUR['mode'] = None
    for i in range(0, NMEAN, 4):
        fwd(MEANR[i:i + 4, :-1].to(DEV).contiguous())
    for h in hs:
        h.remove()
    CUR['hmean'] = torch.stack(capsH).mean(0)
    CUR['lmean'] = {L: torch.stack(capsL[L]).mean(0) for L in (12, 13, 14)}

    # masks
    toks = EVR[:, :-1]; tgt = EVR[:, 1:]
    is_open = torch.isin(toks, open_ids); is_close = torch.isin(toks, close_ids)
    depth = torch.zeros_like(toks)
    d_run = torch.zeros(toks.shape[0], dtype=torch.long)
    for p in range(toks.shape[1]):
        d_run = (d_run + is_open[:, p].long() - is_close[:, p].long()).clamp_min(0)
        depth[:, p] = d_run
    TARGET = torch.isin(tgt, close_ids) & (depth > 0)
    TARGET[:, :64] = False
    submask = {s: TARGET & torch.zeros_like(TARGET) for s in SUBS}
    sub_lookup = torch.zeros(50257, dtype=torch.long)
    names = {s: i + 1 for i, s in enumerate(SUBS)}
    for tok, s in sub_of.items():
        sub_lookup[tok] = names[s]
    tgt_sub = sub_lookup[tgt]
    for s in SUBS:
        submask[s] = TARGET & (tgt_sub == names[s])
        print(f"{s}: n {int(submask[s].sum())}", flush=True)

    hook13 = H[L13].attn.c_proj.register_forward_pre_hook(cproj_hook_13)
    hooksL = [H[L].attn.register_forward_hook(mk_layer_hook(L)) for L in (12, 14)]

    ARMS = {'base': ('none', None), 'h138': ('heads', {8}),
            'a13_full': ('heads', set(range(9))), 'a13_minus138': ('heads', set(range(9)) - {8}),
            'a12': ('layer', 12), 'a14': ('layer', 14)}

    def ce_run(arm):
        mode, spec = ARMS[arm]
        if mode == 'none':
            CUR['mode'] = None
        elif mode == 'heads':
            CUR['mode'] = 'heads'; CUR['heads'] = spec
        else:
            CUR['mode'] = 'layer'; CUR['layer'] = spec
        sums = {s: 0.0 for s in SUBS}; ns = {s: 0 for s in SUBS}
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            for s in SUBS:
                mm = submask[s][i:i + 8].to(DEV)
                sums[s] += float(ce[mm].sum()); ns[s] += int(mm.sum())
        return {s: sums[s] / max(ns[s], 1) for s in SUBS}, ns

    res = {}
    for arm in ARMS:
        r, ns = ce_run(arm)
        res[arm] = {s: round(v, 4) for s, v in r.items()}
        print(f"{arm}: " + " ".join(f"{s} {r[s]:.3f}" for s in SUBS), flush=True)
    for h in [hook13] + hooksL:
        h.remove()

    out_subs = {}
    for s in SUBS:
        d138 = res['h138'][s] - res['base'][s]
        d13 = res['a13_full'][s] - res['base'][s]
        dhelp = res['a13_minus138'][s] - res['base'][s]
        out_subs[s] = {'n': ns[s], 'base_ce': res['base'][s],
                       'dmg_138': round(d138, 4), 'dmg_a13': round(d13, 4),
                       'dmg_helpers_L13': round(dhelp, 4),
                       'dmg_a12': round(res['a12'][s] - res['base'][s], 4),
                       'dmg_a14': round(res['a14'][s] - res['base'][s], 4),
                       'share_138': round(d138 / max(d13, 1e-4), 4)}
    big = {s: v for s, v in out_subs.items() if v['n'] >= 30}
    plain_share = out_subs['plain']['share_138']
    pa = plain_share >= 0.90
    compounds = [s for s in ('double', 'comma', 'period', 'quote') if s in big]
    pb = any(big[s]['share_138'] <= plain_share - 0.15 for s in compounds)
    pc = ('double' in big and
          big['double']['share_138'] == min(v['share_138'] for v in big.values()))
    out = {'n_rows': NR, 'subtypes': out_subs,
           'pred_a_plain_core': bool(pa), 'pred_b_user_helpers': bool(pb),
           'pred_c_double_most_dependent': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    for s in SUBS:
        v = out_subs[s]
        print(f"{s}: n {v['n']} | 13.8 {v['dmg_138']:+.3f} of a13 {v['dmg_a13']:+.3f} "
              f"(share {v['share_138']}) | helpers {v['dmg_helpers_L13']:+.3f} "
              f"| a12 {v['dmg_a12']:+.3f} a14 {v['dmg_a14']:+.3f}")
    print(f"pred_a plain-core {pa} | pred_b user-helpers {pb} | pred_c double {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
