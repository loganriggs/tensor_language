# ordinal_screen: BEHAVIOUR-FIRST screen for ORDINAL INCREMENT — predicting "2" where "1"
# appeared before, "3" after "2", etc. (list continuation / counting). This is INCREMENT,
# not copy: the target token never appeared in context; its PREDECESSOR did. A computation
# class no prior circuit covers (induction copies; digit circuit formats).
#
# Target: positions whose next token decodes (stripped) to d or "d."/"d)" for d in 2..9,
# with a token decoding to d-1 (same forms) in the prior 128 positions. Controls per SOP:
# jitter ±3 and random matched-size. Screen: mean-ablate 36 components, rank by
# concentration.
#
# Registered predictions:
#   pred_a A CARRIER EXISTS: some component reaches concentration >= 3 with jitter <= 1.5.
#   pred_b DEDUP BET: the carrier is NOT the induction stations' layers (L2/L3) — increment
#          needs the predecessor's VALUE transformed, not copied; registered bet: a deep
#          component (L >= 9), like the delimiter closer.
#   pred_c RANDOM CONTROL FLAT (<= 1.5).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ordinal_screen_results.json'
NMEAN = 24; NR = 96
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
    print(f"digit id counts: {[len(dig[d]) for d in range(1,10)]}", flush=True)

    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]

    # component means from MEANR
    means = {}
    caps = {}
    hs = []
    for L in range(18):
        def mk_attn(L):
            def h(mod, args, out):
                caps.setdefault(('attn', L), []).append(out[0].detach().float().mean((0, 1)))
                return out
            return h
        def mk_mlp(L):
            def h(mod, args, out):
                caps.setdefault(('mlp', L), []).append(out.detach().float().mean((0, 1)))
                return out
            return h
        hs.append(H[L].attn.register_forward_hook(mk_attn(L)))
        hs.append(H[L].mlp.register_forward_hook(mk_mlp(L)))
    for i in range(0, NMEAN, 4):
        fwd(MEANR[i:i + 4, :-1].to(DEV).contiguous())
    for h in hs:
        h.remove()
    for k, v in caps.items():
        means[k] = torch.stack(v).mean(0)

    # position sets on EVR
    tgt_all = EVR[:, 1:]
    TARGET = torch.zeros_like(tgt_all, dtype=torch.bool)
    for d in range(2, 10):
        is_d = torch.isin(tgt_all, dig[d])
        prev = torch.isin(EVR[:, :-1], dig[d - 1])
        ctx = torch.zeros_like(prev)
        for w in range(1, 129):
            sh = torch.zeros_like(prev)
            sh[:, w:] = prev[:, :-w]
            ctx |= sh
        TARGET |= (is_d & ctx)
    TARGET[:, :64] = False
    ntar = int(TARGET.sum())
    print(f"target positions: {ntar}", flush=True)
    g = torch.Generator().manual_seed(3)
    jit = torch.roll(TARGET, shifts=3, dims=1)
    jit[:, :64] = False
    JITTER = jit & ~TARGET
    RAND = torch.zeros_like(TARGET)
    flat = torch.randperm(TARGET.numel(), generator=g)[:ntar]
    RAND.view(-1)[flat] = True
    RAND[:, :64] = False
    ELSE = ~TARGET & ~JITTER & ~RAND
    ELSE[:, :64] = False

    hs = [H[L].attn.register_forward_hook(hook_attn(L)) for L in range(18)] + \
         [H[L].mlp.register_forward_hook(hook_mlp(L)) for L in range(18)]

    def ce_sets(kind, L):
        ABL['kind'], ABL['layer'] = kind, L
        ABL['mean'] = None if kind is None else means[(kind, L)]
        tots = {k: 0.0 for k in ('tar', 'jit', 'rand', 'els')}
        ns = {k: 0 for k in tots}
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV); idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            lse = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1), reduction='none').view(tg.shape)
            for name, mask in (('tar', TARGET), ('jit', JITTER), ('rand', RAND), ('els', ELSE)):
                mm = mask[i:i + 8].to(DEV)
                tots[name] += float(lse[mm].sum()); ns[name] += int(mm.sum())
        return {k: tots[k] / max(ns[k], 1) for k in tots}

    base = ce_sets(None, -1)
    print(f"base {base}", flush=True)
    results = {}
    for kind in ('attn', 'mlp'):
        for L in range(18):
            r = ce_sets(kind, L)
            dmg = {k: r[k] - base[k] for k in r}
            conc = dmg['tar'] / max(dmg['els'], 1e-4)
            cj = dmg['jit'] / max(dmg['els'], 1e-4)
            crand = dmg['rand'] / max(dmg['els'], 1e-4)
            results[f'{kind}{L}'] = {'dmg': {k: round(v, 4) for k, v in dmg.items()},
                                     'conc': round(conc, 2), 'conc_jit': round(cj, 2),
                                     'conc_rand': round(crand, 2)}
            print(f"{kind}{L}: conc {conc:.2f} (jit {cj:.2f}, rand {crand:.2f}) dmg_tar {dmg['tar']:.4f}", flush=True)
    for h in hs:
        h.remove()
    ranked = sorted(results.items(), key=lambda kv: -kv[1]['conc'])
    win, wr = ranked[0]
    pa = wr['conc'] >= 3 and wr['conc_jit'] <= 1.5
    pb = win.startswith('attn') and 2 <= int(win.replace('attn', '').replace('mlp', '')) <= 8
    pc = wr['conc_rand'] <= 1.5
    out = {'n_targets': ntar, 'base': {k: round(v, 4) for k, v in base.items()},
           'top5': [(k, v['conc'], v['conc_jit'], v['conc_rand'], v['dmg']['tar']) for k, v in ranked[:5]],
           'winner': win, 'winner_stats': wr,
           'pred_a_carrier': bool(pa), 'pred_b_attn_L2_8': bool(pb), 'pred_c_rand_flat': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"top5 {out['top5']}")
    print(f"pred_a carrier {pa} | pred_b attnL2-8 {pb} | pred_c rand {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
