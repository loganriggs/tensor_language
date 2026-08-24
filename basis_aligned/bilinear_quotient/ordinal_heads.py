# ordinal_heads: head decomposition of the increment carrier attn8 (§1275), widened to 480
# rows, with the three-way dedup: digit circuit vs fetchers vs new head.
#
# Two target classes: INC (next token = digit d with d-1 in prior 128) and PLAIN (next
# token = digit with NO predecessor digit in prior 128 — pure digit formatting).
#
# Registered predictions:
#   pred_a ONE HEAD: top head of attn8 >= 60% of the layer's INC-target damage.
#   pred_b NOT THE FETCHERS (bet): the winner is not head 3 or 4 — increment is its own
#          machinery, not the copy fetchers moonlighting.
#   pred_c INCREMENT-SPECIFIC: the winner's INC concentration >= 2x its PLAIN concentration
#          (if FALSE and they are equal, the "increment circuit" deduplicates INTO the digit
#          circuit — logged either way; both-halves gate on INC required regardless).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ordinal_heads_results.json'
NMEAN = 24; NR = 480
H = m.transformer.h
ABL = {'kind': None, 'layer': -1, 'mean': None}


def hook_attn(L):
    def h(mod, args, out):
        if ABL['kind'] == 'attn' and ABL['layer'] == L:
            x1, v1 = out
            return (ABL['mean'].to(x1.dtype).expand_as(x1), v1)
        return out
    return h


def hook_mlp(L):
    def h(mod, args, out):
        if ABL['kind'] == 'mlp' and ABL['layer'] == L:
            return ABL['mean'].to(out.dtype).expand_as(out)
        return out
    return h


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
    dig = {d: set() for d in range(1, 10)}
    for tok in range(50257):
        try:
            s = enc.decode([tok]).strip()
        except Exception:
            continue
        for d in range(1, 10):
            if s in (str(d), f"{d}.", f"{d})"):
                dig[d].add(tok)
    dig = {d: torch.tensor(sorted(v)) for d, v in dig.items()}
    alldig = torch.cat([dig[d] for d in range(1, 10)])
    print(f"digit id counts: {[len(dig[d]) for d in range(1,10)]}", flush=True)

    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]

    # attn13 mean output (full and per-head slices via c_proj structure: capture pre-proj y)
    capy = []
    at13 = H[8].attn
    def cap_hook(mod, args, out):
        return out
    # capture y before c_proj by hooking c_proj input
    ys = []
    def cproj_hook(mod, args):
        ys.append(args[0].detach().float().mean((0, 1)))
    hh = at13.c_proj.register_forward_pre_hook(cproj_hook)
    for i in range(0, NMEAN, 4):
        fwd(MEANR[i:i + 4, :-1].to(DEV).contiguous())
    hh.remove()
    ymean = torch.stack(ys).mean(0)                            # (D,) head-major pre-proj mean

    HSEL = {'head': -1}
    def abl_hook(mod, args):
        if HSEL['head'] >= 0:
            y = args[0].clone()
            h = HSEL['head']
            y[:, :, h * 128:(h + 1) * 128] = ymean[h * 128:(h + 1) * 128].to(y.dtype)
            return (y,)
        return args
    hh = at13.c_proj.register_forward_pre_hook(abl_hook)

    tgt_all = EVR[:, 1:]
    QT = torch.zeros_like(tgt_all, dtype=torch.bool)      # INC targets
    anyprev = torch.zeros_like(tgt_all, dtype=torch.bool)
    for d in range(2, 10):
        is_d = torch.isin(tgt_all, dig[d])
        prev = torch.isin(EVR[:, :-1], dig[d - 1])
        ctx = torch.zeros_like(prev)
        for w in range(1, 129):
            sh = torch.zeros_like(prev)
            sh[:, w:] = prev[:, :-w]
            ctx |= sh
        QT |= (is_d & ctx)
    anydig_prev = torch.isin(EVR[:, :-1], alldig)
    ctxall = torch.zeros_like(anydig_prev)
    for w in range(1, 129):
        sh = torch.zeros_like(anydig_prev)
        sh[:, w:] = anydig_prev[:, :-w]
        ctxall |= sh
    BT = torch.isin(tgt_all, alldig) & ~ctxall             # PLAIN: digit target, no digit context
    QT[:, :64] = False; BT[:, :64] = False
    print(f"INC targets {int(QT.sum())} | PLAIN targets {int(BT.sum())}", flush=True)
    ELSE = ~QT & ~BT
    ELSE[:, :64] = False

    def ce_sets(head, half=None):
        HSEL['head'] = head
        lo_r, hi_r = (0, NR) if half is None else ((0, NR // 2) if half == 0 else (NR // 2, NR))
        tots = {k: 0.0 for k in ('q', 'b', 'e')}; ns = {k: 0 for k in tots}
        for i in range(lo_r, hi_r, 8):
            bb = EVR[i:i + 8].to(DEV); idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            lse = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1), reduction='none').view(tg.shape)
            for name, mask in (('q', QT), ('b', BT), ('e', ELSE)):
                mm = mask[i:i + 8].to(DEV)
                tots[name] += float(lse[mm].sum()); ns[name] += int(mm.sum())
        return {k: tots[k] / max(ns[k], 1) for k in tots}

    base = ce_sets(-1)
    print(f"base {base}", flush=True)
    per = {}
    for h in range(9):
        r = ce_sets(h)
        dq = r['q'] - base['q']; db = r['b'] - base['b']; de = r['e'] - base['e']
        per[h] = {'dmg_q': round(dq, 4), 'dmg_b': round(db, 4), 'dmg_e': round(de, 5),
                  'conc_q': round(dq / max(de, 1e-4), 2), 'conc_b': round(db / max(de, 1e-4), 2)}
        print(f"h{h}: q {dq:.4f} (conc {per[h]['conc_q']}) | b {db:.4f} (conc {per[h]['conc_b']}) | e {de:.5f}", flush=True)
    tot_q = sum(v['dmg_q'] for v in per.values())
    win = max(per, key=lambda h: per[h]['dmg_q'])
    share = per[win]['dmg_q'] / max(tot_q, 1e-6)
    halves = []
    for half in (0, 1):
        b2 = ce_sets(-1, half); r2 = ce_sets(win, half)
        c = (r2['q'] - b2['q']) / max(r2['e'] - b2['e'], 1e-4)
        halves.append(round(c, 2))
    hh.remove()
    out = {'n_rows': NR, 'base': {k: round(v, 4) for k, v in base.items()},
           'per_head': per, 'winner': win, 'winner_share': round(share, 3),
           'halves_conc': halves,
           'pred_a_one_head': bool(share >= 0.60),
           'pred_b_halves_gate': bool(all(c >= 3 for c in halves)),
           'pred_c_inc_specific': bool(per[win]['conc_q'] >= 2 * max(per[win]['conc_b'], 1e-6)),\n           'pred_b_not_fetchers': bool(win not in (3, 4)),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"winner h{win} share {share:.3f} | halves {halves} | bracket conc {per[win]['conc_b']}")
    print(f"pred_a one-head {out['pred_a_one_head']} | halves {out['pred_b_halves_gate']} | not-fetchers {out['pred_b_not_fetchers']} | inc-specific {out['pred_c_inc_specific']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
