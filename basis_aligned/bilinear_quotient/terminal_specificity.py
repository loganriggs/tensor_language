# terminal_specificity: is 10.5 QUESTION-specific or a general sentence-terminal head?
# Mean-ablate 10.5's c_proj slice alone and measure damage at three terminal classes:
# "?" in WH-opened sentences (anchor), "!" as next token, "." as next token (sentence-
# final periods). 960 rows.
#
# Registered predictions:
#   pred_a ANCHOR: "?" damage within 30% of the §1284 value (0.65).
#   pred_b QUESTION-SPECIFIC: "!" and "." damage each <= 20% of "?" damage.
#   pred_c ELSEWHERE CLEAN: elsewhere damage <= 5% of "?" damage.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'terminal_specificity_results.json'
NMEAN = 24; NR = 960; L = 10
H = m.transformer.h


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
    qm = set(); ex = set(); pe = set(); sent_end = set(); wh = set()
    WH = ['Who', 'What', 'When', 'Where', 'Why', 'How', 'Which', 'Is', 'Are', 'Do', 'Does',
          'Did', 'Can', 'Could', 'Will', 'Would', 'Should']
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        if '?' in d:
            qm.add(tok)
        if '!' in d:
            ex.add(tok)
        if d.strip() == '.':
            pe.add(tok)
        if any(c in d for c in '.!?'):
            sent_end.add(tok)
        if d.strip() in WH:
            wh.add(tok)
    qm = torch.tensor(sorted(qm)); ex_t = torch.tensor(sorted(ex)); pe_t = torch.tensor(sorted(pe))
    se_t = torch.tensor(sorted(sent_end)); wh_t = torch.tensor(sorted(wh))

    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]
    toks = EVR[:, :-1]; tgt_all = EVR[:, 1:]
    is_end = torch.isin(toks, se_t); is_wh = torch.isin(toks, wh_t)
    B2, T2 = toks.shape
    state = torch.zeros(B2, dtype=torch.bool)
    QSTATE = torch.zeros_like(toks, dtype=torch.bool)
    rec = torch.full((B2,), 99, dtype=torch.long)
    for p in range(T2):
        op = is_wh[:, p] & (rec <= 2)
        state = torch.where(is_end[:, p], torch.zeros_like(state), state | op)
        QSTATE[:, p] = state
        rec = torch.where(is_end[:, p], torch.zeros_like(rec), rec + 1)
    QT = torch.isin(tgt_all, qm) & QSTATE
    ET = torch.isin(tgt_all, ex_t)
    PT2 = torch.isin(tgt_all, pe_t)
    for M in (QT, ET, PT2):
        M[:, :64] = False
    ELSE = ~QT & ~ET & ~PT2; ELSE[:, :64] = False
    print(f"q {int(QT.sum())} | ! {int(ET.sum())} | . {int(PT2.sum())}", flush=True)

    caps = []
    hk = H[L].attn.c_proj.register_forward_pre_hook(
        lambda mod, args: caps.append(args[0].detach().float().mean((0, 1))))
    for i in range(0, NMEAN, 4):
        fwd(MEANR[i:i + 4, :-1].to(DEV).contiguous())
    hk.remove()
    ymean = torch.stack(caps).mean(0)

    SEL = {'h': None, 'all': False}

    def hook(mod, args):
        if SEL['h'] is None and not SEL['all']:
            return args
        y = args[0].clone()
        if SEL['all']:
            y[:, :, :] = ymean.to(y.dtype)
        else:
            hh = SEL['h']
            y[:, :, hh * 128:(hh + 1) * 128] = ymean[hh * 128:(hh + 1) * 128].to(y.dtype)
        return (y,)

    hk = H[L].attn.c_proj.register_forward_pre_hook(hook)

    NAMES = ('q', 'ex', 'pe', 'els'); SETS = (QT, ET, PT2, ELSE)

    def ce_sets(h, allh=False):
        SEL['h'], SEL['all'] = h, allh
        tots = {k: 0.0 for k in NAMES}; ns = {k: 0 for k in NAMES}
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV); idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            lse = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1), reduction='none').view(tg.shape)
            for name, mask in zip(NAMES, SETS):
                mm = mask[i:i + 8].to(DEV)
                tots[name] += float(lse[mm].sum()); ns[name] += int(mm.sum())
        return {k: tots[k] / max(ns[k], 1) for k in tots}

    base = ce_sets(None)
    r = ce_sets(5)
    hk.remove()
    d = {k: r[k] - base[k] for k in NAMES}
    pa = abs(d['q'] - 0.65) <= 0.3 * 0.65
    pb = d['ex'] <= 0.2 * max(d['q'], 1e-4) and d['pe'] <= 0.2 * max(d['q'], 1e-4)
    pc = abs(d['els']) <= 0.05 * max(d['q'], 1e-4)
    out = {'n_rows': NR, 'counts': {k: int(M.sum()) for k, M in zip(NAMES, SETS)},
           'base': {k: round(v, 4) for k, v in base.items()},
           'dmg_10_5': {k: round(v, 4) for k, v in d.items()},
           'pred_a_anchor': bool(pa), 'pred_b_question_specific': bool(pb),
           'pred_c_else_clean': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"dmg: q {d['q']:.4f} | ! {d['ex']:.4f} | . {d['pe']:.4f} | els {d['els']:.4f}")
    print(f"pred_a anchor {pa} | pred_b q-specific {pb} | pred_c els {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
