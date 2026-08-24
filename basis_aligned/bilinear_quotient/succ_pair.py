# succ_pair: the §1279 open edge — 14.4's causal share of succession. Mean-ablate at INC
# targets (the §1276 target class): 8.7 alone (anchor), 14.4 alone, both. Process note for
# the record: succ_general's exit=134 was a post-results teardown abort (HF prefetch thread
# died during interpreter finalization; JSON intact).
#
# Registered predictions:
#   pred_a 14.4 IS BACKUP, NOT CO-PRIMARY: its solo INC damage <= 0.3x of 8.7's (0.273).
#   pred_b REDUNDANT COVERAGE: joint ablation >= 1.3x 8.7-alone (the backup's value appears
#          when the primary dies — §1207/§1279 pattern).
#   pred_c ANCHOR: 8.7-alone replicates §1276 (0.273 ± 0.05).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'succ_pair_results.json'
NMEAN = 24; NR = 480
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

    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]
    tgt_all = EVR[:, 1:]
    TGT = torch.zeros_like(tgt_all, dtype=torch.bool)
    for d in range(2, 10):
        is_d = torch.isin(tgt_all, dig[d])
        prev = torch.isin(EVR[:, :-1], dig[d - 1])
        ctx = torch.zeros_like(prev)
        for w in range(1, 129):
            sh = torch.zeros_like(prev)
            sh[:, w:] = prev[:, :-w]
            ctx |= sh
        TGT |= (is_d & ctx)
    TGT[:, :64] = False
    ELSE = ~TGT; ELSE[:, :64] = False
    print(f"INC targets {int(TGT.sum())}", flush=True)

    ymeans = {}
    hooks = []
    caps = {8: [], 14: []}
    for L in (8, 14):
        def mk(L):
            def h(mod, args):
                caps[L].append(args[0].detach().float().mean((0, 1)))
            return h
        hooks.append(H[L].attn.c_proj.register_forward_pre_hook(mk(L)))
    for i in range(0, NMEAN, 4):
        fwd(MEANR[i:i + 4, :-1].to(DEV).contiguous())
    for h in hooks:
        h.remove()
    for L in (8, 14):
        ymeans[L] = torch.stack(caps[L]).mean(0)

    SEL = {'set': set()}
    HEADIDX = {8: 7, 14: 4}
    hooks = []
    for L in (8, 14):
        def mk(L):
            def h(mod, args):
                if L in SEL['set']:
                    y = args[0].clone()
                    hh = HEADIDX[L]
                    y[:, :, hh * 128:(hh + 1) * 128] = ymeans[L][hh * 128:(hh + 1) * 128].to(y.dtype)
                    return (y,)
                return args
            return h
        hooks.append(H[L].attn.c_proj.register_forward_pre_hook(mk(L)))

    def ce_sets(sel):
        SEL['set'] = sel
        tots = {'t': 0.0, 'e': 0.0}; ns = {'t': 0, 'e': 0}
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV); idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            lse = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1), reduction='none').view(tg.shape)
            for name, mask in (('t', TGT), ('e', ELSE)):
                mm = mask[i:i + 8].to(DEV)
                tots[name] += float(lse[mm].sum()); ns[name] += int(mm.sum())
        return {k: tots[k] / max(ns[k], 1) for k in tots}

    base = ce_sets(set())
    r87 = ce_sets({8})
    r144 = ce_sets({14})
    rboth = ce_sets({8, 14})
    for h in hooks:
        h.remove()
    d87 = r87['t'] - base['t']; d144 = r144['t'] - base['t']; db = rboth['t'] - base['t']
    out = {'n_rows': NR, 'base_t': round(base['t'], 4),
           'dmg': {'h87': round(d87, 4), 'h144': round(d144, 4), 'both': round(db, 4)},
           'else_dmg': {'h87': round(r87['e'] - base['e'], 5), 'h144': round(r144['e'] - base['e'], 5)},
           'pred_a_backup': bool(d144 <= 0.3 * d87),
           'pred_b_redundant': bool(db >= 1.3 * d87),
           'pred_c_anchor': bool(abs(d87 - 0.2725) <= 0.05),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"dmg 8.7 {d87:.4f} | 14.4 {d144:.4f} | both {db:.4f}")
    print(f"pred_a backup {out['pred_a_backup']} | pred_b redundant {out['pred_b_redundant']} | pred_c anchor {out['pred_c_anchor']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
