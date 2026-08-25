# newline_pair_quote: CHARACTERIZE THE §1415 FINDS. (1) Pair structure: 8.2+11.0 joint
# removal vs solos — two-crew redundancy echo? (2) 8.2 subtype split: single '\n' tokens
# vs paragraph tokens (decode contains '\n\n' or is all-newlines) vs mixed word+newline
# tokens. (3) 10.6 side split: quote targets by parity (cumulative quote count in the
# row: even = OPEN side, odd = CLOSE side; 13.8 owns closes). NR=960, skip=5600 (same
# rows as the screen for comparability). Assumptions registered.
#
# Registered predictions:
#   pred_a joint 8.2+11.0 newline damage >= 1.3x the larger solo (partial redundancy,
#          crews echo).
#   pred_b 8.2 has subtype structure: best/worst subtype damage ratio >= 2x.
#   pred_c 10.6 is the OPENER: open-side damage >= 2x close-side damage.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'newline_pair_quote_results.json'
NMEAN = 24; NR = 960
H = m.transformer.h
HEADS = [(8, 2), (11, 0), (10, 6)]
CUR = {'head': None, 'mean': {}}


def mk_hook(L):
    def hook(mod, args):
        if not CUR['head']:
            return None
        y = None
        for (LL, h) in CUR['head']:
            if LL == L:
                if y is None:
                    y = args[0].clone()
                y[..., h * 128:(h + 1) * 128] = CUR['mean'][L][h].to(y.dtype)
        return None if y is None else (y,)
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
    nl = set(); pu = set(); qu = set()
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        ds = d.strip()
        if '\n' in d:
            nl.add(tok)
        if '"' in d or "'" in d:
            qu.add(tok)
        elif ds and all(not c.isalnum() for c in ds) and '\n' not in d \
                and not any(c in ds for c in ')]}(['):
            pu.add(tok)
    IDS = {'newline': torch.tensor(sorted(nl)), 'punct': torch.tensor(sorted(pu)),
           'quote': torch.tensor(sorted(qu))}

    ROWS = cl.fineweb_rows(NMEAN + NR, skip=5600)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]

    need_layers = sorted({L for (L, h) in HEADS})
    caps = {L: [] for L in need_layers}
    hks = [H[L].attn.c_proj.register_forward_pre_hook(
        (lambda LL: lambda mod, args: caps[LL].append(
            args[0].detach().float().reshape(-1, 9, 128).mean(0)))(L))
        for L in need_layers]
    CUR['head'] = None
    for i in range(0, NMEAN, 4):
        fwd(MEANR[i:i + 4, :-1].to(DEV).contiguous())
    for hk in hks:
        hk.remove()
    for L in need_layers:
        CUR['mean'][L] = torch.stack(caps[L]).mean(0).to(DEV)

    tgt_all = EVR[:, 1:]; toks = EVR[:, :-1]
    qids = IDS['quote']
    isq = torch.isin(toks, qids)
    par = (isq.long().cumsum(1) % 2) == 1
    masks = {}
    defs = {'nl_single': torch.isin(tgt_all, IDS['nl_single']),
            'nl_para': torch.isin(tgt_all, IDS['nl_para']),
            'nl_mixed': torch.isin(tgt_all, IDS['nl_mixed']),
            'q_open': torch.isin(tgt_all, qids) & ~par,
            'q_close': torch.isin(tgt_all, qids) & par}
    for cls, TARGET in defs.items():
        TARGET = TARGET.clone()
        TARGET[:, :64] = False
        JIT = torch.zeros_like(TARGET)
        JIT[:, 2:] = TARGET[:, :-2]
        JIT &= ~TARGET
        ELSE = ~TARGET & ~JIT; ELSE[:, :64] = False
        masks[cls] = dict(T=TARGET, J=JIT, R=TARGET & False, E=ELSE, n=int(TARGET.sum()))
        print(f"{cls}: n={masks[cls]['n']}", flush=True)

    hooks = [H[L].attn.c_proj.register_forward_pre_hook(mk_hook(L))
             for L in need_layers]

    def ce_all(head):
        CUR['head'] = (head if head is None or isinstance(head, set) else {head})
        acc = {c: {k: [0.0, 0] for k in 'TJRE'} for c in masks}
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            for c in masks:
                for kk in 'TJRE':
                    mm = masks[c][kk][i:i + 8].to(DEV)
                    acc[c][kk][0] += float(ce[mm].sum()); acc[c][kk][1] += int(mm.sum())
        return {c: {kk: acc[c][kk][0] / max(acc[c][kk][1], 1) for kk in 'TJRE'}
                for c in masks}

    base = ce_all(None)
    results = {}
    for name, hs in (('8.2', {(8, 2)}), ('11.0', {(11, 0)}),
                     ('both', {(8, 2), (11, 0)}), ('10.6', {(10, 6)})):
        r = ce_all(hs if len(hs) > 1 else next(iter(hs)))
        d = {c: {kk: round(r[c][kk] - base[c][kk], 4) for kk in 'TJRE'} for c in r}
        results[name] = d
        print(f"{name}: " + " ".join(f"{c} {d[c]['T']:+.4f}" for c in d), flush=True)
        json.dump({'partial': True, 'results': results}, open(OUT, 'w'), indent=1)
    for hk in hooks:
        hk.remove()

    nlsum = lambda d: sum(d[c]['T'] * masks[c]['n'] for c in
                          ('nl_single', 'nl_para', 'nl_mixed')) / \
        max(sum(masks[c]['n'] for c in ('nl_single', 'nl_para', 'nl_mixed')), 1)
    j82, j110, jboth = nlsum(results['8.2']), nlsum(results['11.0']), nlsum(results['both'])
    pa = jboth >= 1.3 * max(j82, j110)
    subs = {c: results['8.2'][c]['T'] for c in ('nl_single', 'nl_para', 'nl_mixed')}
    hi, lo = max(subs.values()), max(min(subs.values()), 1e-4)
    pb = hi / lo >= 2.0
    qo = results['10.6']['q_open']['T']; qc = results['10.6']['q_close']['T']
    pc = qo >= 2.0 * max(qc, 1e-4)
    out = {'n': {c: masks[c]['n'] for c in masks}, 'results': results,
           'newline_joint': {'8.2': round(j82, 4), '11.0': round(j110, 4),
                             'both': round(jboth, 4)},
           'subtypes_82': {c: round(v, 4) for c, v in subs.items()},
           'quote_sides_106': {'open': round(qo, 4), 'close': round(qc, 4)},
           'pred_a_pair_redundant': bool(pa), 'pred_b_82_structured': bool(pb),
           'pred_c_106_opener': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"joint {jboth:.4f} vs solos {j82:.4f}/{j110:.4f} | subs {subs} | "
          f"open {qo:.4f} close {qc:.4f}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
