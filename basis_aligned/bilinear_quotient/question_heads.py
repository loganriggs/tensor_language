# question_heads: per-head decomposition of §1282's winner. attn10 carries the
# question-state consumption (conc 32.6 at "?"-prediction targets in WH-opened sentences).
# Which of its 9 heads owns it? Per-head c_proj-slice mean ablation (the §1272/§1276
# instrument), 480 rows (§1282's n=39 targets was flagged small; ~2.5x rows here).
#
# Registered predictions (dominant-owner pattern, as at 13.8 and 8.7):
#   pred_a ONE OWNER: some single head's target damage >= 60% of the whole-layer share
#          reference (whole-layer dmg re-measured here on the same rows).
#   pred_b CONCENTRATED: that head's concentration >= 10 with jitter <= 1.5.
#   pred_c REST QUIET: every other head <= 20% of the whole-layer damage.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'question_heads_results.json'
NMEAN = 24; NR = 480; L = 10
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
    qm = set(); sent_end = set(); wh = set()
    WH = ['Who', 'What', 'When', 'Where', 'Why', 'How', 'Which', 'Is', 'Are', 'Do', 'Does',
          'Did', 'Can', 'Could', 'Will', 'Would', 'Should']
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        if '?' in d:
            qm.add(tok)
        if any(c in d for c in '.!?'):
            sent_end.add(tok)
        if d.strip() in WH:
            wh.add(tok)
    qm = torch.tensor(sorted(qm)); sent_end = torch.tensor(sorted(sent_end)); wh = torch.tensor(sorted(wh))

    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]
    toks = EVR[:, :-1]; tgt_all = EVR[:, 1:]
    is_end = torch.isin(toks, sent_end); is_wh = torch.isin(toks, wh)
    B2, T2 = toks.shape
    state = torch.zeros(B2, dtype=torch.bool)
    QSTATE = torch.zeros_like(toks, dtype=torch.bool)
    recent_end = torch.full((B2,), 99, dtype=torch.long)
    for p in range(T2):
        opener = is_wh[:, p] & (recent_end <= 2)
        state = torch.where(is_end[:, p], torch.zeros_like(state), state | opener)
        QSTATE[:, p] = state
        recent_end = torch.where(is_end[:, p], torch.zeros_like(recent_end), recent_end + 1)
    TARGET = torch.isin(tgt_all, qm) & QSTATE
    TARGET[:, :64] = False
    ntar = int(TARGET.sum())
    jit = torch.roll(TARGET, shifts=3, dims=1); jit[:, :64] = False
    JITTER = jit & ~TARGET
    ELSE = ~TARGET & ~JITTER; ELSE[:, :64] = False
    print(f"targets {ntar}", flush=True)

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

    def ce_sets(h, allh=False):
        SEL['h'], SEL['all'] = h, allh
        tots = {'t': 0.0, 'j': 0.0, 'e': 0.0}; ns = {'t': 0, 'j': 0, 'e': 0}
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV); idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            lse = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1), reduction='none').view(tg.shape)
            for name, mask in (('t', TARGET), ('j', JITTER), ('e', ELSE)):
                mm = mask[i:i + 8].to(DEV)
                tots[name] += float(lse[mm].sum()); ns[name] += int(mm.sum())
        return {k: tots[k] / max(ns[k], 1) for k in tots}

    base = ce_sets(None)
    lay = ce_sets(None, allh=True)
    dlay = lay['t'] - base['t']
    print(f"base_t {base['t']:.4f} | whole-layer dmg_t {dlay:.4f}", flush=True)
    heads = {}
    for hh in range(9):
        r = ce_sets(hh)
        dt = r['t'] - base['t']; dj = r['j'] - base['j']; de = r['e'] - base['e']
        heads[hh] = {'dmg_t': round(dt, 4), 'dmg_j': round(dj, 4), 'dmg_e': round(de, 5),
                     'share': round(dt / max(dlay, 1e-4), 3),
                     'conc': round(dt / max(de, 1e-4), 2),
                     'conc_jit': round(dj / max(de, 1e-4), 2)}
        print(f"10.{hh}: dmg_t {dt:.4f} share {heads[hh]['share']:.3f} conc {heads[hh]['conc']:.1f} "
              f"(jit {heads[hh]['conc_jit']:.1f})", flush=True)
    hk.remove()
    win = max(heads, key=lambda h: heads[h]['dmg_t'])
    w = heads[win]
    pa = w['share'] >= 0.60
    pb = w['conc'] >= 10 and w['conc_jit'] <= 1.5
    pc = all(heads[h]['share'] <= 0.20 for h in heads if h != win)
    out = {'n_targets': ntar, 'base_t': round(base['t'], 4), 'layer_dmg_t': round(dlay, 4),
           'heads': heads, 'winner': f'10.{win}',
           'pred_a_one_owner': bool(pa), 'pred_b_concentrated': bool(pb), 'pred_c_rest_quiet': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"winner 10.{win} share {w['share']} conc {w['conc']}")
    print(f"pred_a one-owner {pa} | pred_b concentrated {pb} | pred_c rest-quiet {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
