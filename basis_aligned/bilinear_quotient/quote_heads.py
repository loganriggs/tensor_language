# quote_heads: SOP step 3 for the quote-state carrier — decompose attn13 into heads on
# WIDENED data (240 rows, ~200+ targets per the more-data rule), with the certification
# gate and the bclose dedup comparison.
#
# Per-head instrument: replace ONE head's slice of attn13's output with its mean (same
# global-mean idiom, head-slice granularity). Concentration per SOP.
#
# Registered predictions:
#   pred_a ONE HEAD: the top head of attn13 carries >= 60% of the layer's target damage.
#   pred_b BOTH-HALVES GATE (SOP certification): the top head's concentration >= 3 on
#          rows 0-119 and rows 120-239 independently.
#   pred_c DELIMITER DEDUP: the same head, mean-ablated, damages BRACKET-close targets
#          (next token is a closing bracket/paren with an opener in context) with
#          concentration >= 3 as well — quote-close and bracket-close are ONE machine
#          (if FALSE, they are separate circuits sharing a layer; logged either way).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'quote_heads_results.json'
NMEAN = 24; NR = 240
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
    def ids_for(chars):
        out = set()
        for t in range(50257):
            try:
                d = enc.decode([t])
            except Exception:
                continue
            if any(c in d for c in chars):
                out.add(t)
        return torch.tensor(sorted(out))
    qids = ids_for(['"'])
    bids = ids_for([')', ']', '}'])
    oids = ids_for(['(', '[', '{'])
    print(f"quote ids {len(qids)} | close-bracket ids {len(bids)} | open ids {len(oids)}", flush=True)

    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]

    # attn13 mean output (full and per-head slices via c_proj structure: capture pre-proj y)
    capy = []
    at13 = H[13].attn
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

    def targets_for(close_ids, open_ids_ctx):
        tgt_all = EVR[:, 1:]
        isq = torch.isin(tgt_all, close_ids)
        ctx = torch.zeros_like(isq)
        tok_iso = torch.isin(EVR[:, :-1], open_ids_ctx)
        for w in range(1, 65):
            sh = torch.zeros_like(tok_iso)
            sh[:, w:] = tok_iso[:, :-w]
            ctx |= sh
        TGT = isq & ctx
        TGT[:, :64] = False
        return TGT

    QT = targets_for(qids, qids)
    BT = targets_for(bids, oids)
    print(f"quote targets {int(QT.sum())} | bracket targets {int(BT.sum())}", flush=True)
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
           'pred_c_bracket_too': bool(per[win]['conc_b'] >= 3),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"winner h{win} share {share:.3f} | halves {halves} | bracket conc {per[win]['conc_b']}")
    print(f"pred_a one-head {out['pred_a_one_head']} | pred_b halves {out['pred_b_halves_gate']} | pred_c bracket {out['pred_c_bracket_too']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
