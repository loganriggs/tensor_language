# comparative_heads: per-head decomposition of §1303's winner. attn8 carries the
# comparative->"than" expectation (conc 44.7 at 5x data). Which of its 9 heads owns it?
# L8 is crowded (fetchers 8.3/8.4, successor 8.7). Per-head c_proj-slice mean ablation,
# 960 rows.
#
# Registered predictions:
#   pred_a ONE OWNER: some single head's target damage >= 60% of the whole-layer damage.
#   pred_b A FOURTH SPECIALIST: the winner is NOT 8.3, 8.4, or 8.7.
#   pred_c REST QUIET: every other head <= 20% of whole-layer damage.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'comparative_heads_results.json'
NMEAN = 24; NR = 960; L = 8
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
    COMP = ['bigger', 'smaller', 'better', 'worse', 'larger', 'greater', 'higher',
            'lower', 'faster', 'slower', 'older', 'younger', 'stronger', 'weaker',
            'easier', 'harder', 'longer', 'shorter', 'cheaper', 'richer', 'more', 'less',
            'fewer', 'rather']
    than = set(); comp = set()
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        if d.strip().lower() == 'than':
            than.add(tok)
        if d.strip().lower() in COMP:
            comp.add(tok)
    qm = torch.tensor(sorted(than)); wh = torch.tensor(sorted(comp))
    print(f"than-ids {len(qm)} | comparative-ids {len(wh)}", flush=True)


    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]
    tgt_all = EVR[:, 1:]
    toks = EVR[:, :-1]
    is_comp = torch.isin(toks, wh)
    ctx = torch.zeros_like(is_comp)
    for w in range(2, 21):
        sh = torch.zeros_like(is_comp)
        sh[:, w:] = is_comp[:, :-w]
        ctx |= sh
    TARGET = torch.isin(tgt_all, qm) & ctx
    TARGET[:, :64] = False
    ntar = int(TARGET.sum())
    print(f"target positions: {ntar}", flush=True)
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
    pb = win not in (3, 4, 7)
    pc = all(heads[h]['share'] <= 0.20 for h in heads if h != win)
    out = {'n_targets': ntar, 'base_t': round(base['t'], 4), 'layer_dmg_t': round(dlay, 4),
           'heads': heads, 'winner': f'10.{win}',
           'pred_a_one_owner': bool(pa), 'pred_b_fourth_specialist': bool(pb), 'pred_c_rest_quiet': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"winner 10.{win} share {w['share']} conc {w['conc']}")
    print(f"pred_a one-owner {pa} | pred_b fourth {pb} | pred_c rest-quiet {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
