# pool_screen_a10_a17: §1302-STANDARD SCREENS FOR THE LAST TWO POOL CANDIDATES.
# (1) sentence_end (next token contains .!?), registry leader a10 (+0.065 atlas) —
#     dedup notes: open_quote at the SAME layer was demoted §1351 (conc 2.7, jitter
#     dirty); the newline circuit is FRONT-attention; expectation registered crowd-ward.
# (2) open_bracket (next token contains '('), registry leader a17 (+0.070 atlas, n=59) —
#     shares a17 with the capitalized committee.
# Method: full 18-layer attention ladders (mean-ablate each layer's y at c_proj),
# target/jitter/random/else masks, concentration = top damage / mean of other 17.
# NR=960. Assumptions registered per standing directive.
#
# Registered predictions:
#   pred_a sentence_end FAILS at least one §1302 bar (top conc < 3x OR jitter dirty —
#          damage > 1.5x-scaled controls), echoing open_quote's distributed class.
#   pred_b open_bracket's top layer is a16 or a17 (late, near the capitalized band).
#   pred_c open_bracket controls clean: jitter damage <= max(0.010, 0.33x target damage).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'pool_screen_a10_a17_results.json'
NMEAN = 24; NR = 960
H = m.transformer.h
CUR = {'layer': None, 'mean': {}}


def mk_hook(L):
    def hook(mod, args):
        if CUR['layer'] != L:
            return None
        y = args[0].clone()
        y[:] = CUR['mean'][L].to(y.dtype).reshape(1, 1, D)
        return (y,)
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
    se = set(); ob = set()
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        if any(c in d for c in '.!?'):
            se.add(tok)
        if '(' in d:
            ob.add(tok)
    se_ids = torch.tensor(sorted(se)); ob_ids = torch.tensor(sorted(ob))

    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]

    caps = {L: [] for L in range(18)}
    hks = [H[L].attn.c_proj.register_forward_pre_hook(
        (lambda LL: lambda mod, args: caps[LL].append(
            args[0].detach().float().mean((0, 1)).cpu()))(L)) for L in range(18)]
    CUR['layer'] = None
    for i in range(0, NMEAN, 4):
        fwd(MEANR[i:i + 4, :-1].to(DEV).contiguous())
    for hk in hks:
        hk.remove()
    for L in range(18):
        CUR['mean'][L] = torch.stack(caps[L]).mean(0).to(DEV)

    tgt_all = EVR[:, 1:]
    fams = {}
    for name, ids in (('sentence_end', se_ids), ('open_bracket', ob_ids)):
        TARGET = torch.isin(tgt_all, ids)
        TARGET[:, :64] = False
        JIT = torch.zeros_like(TARGET)
        JIT[:, 2:] = TARGET[:, :-2]
        JIT &= ~TARGET
        g = torch.Generator().manual_seed(97)
        sc = torch.rand(TARGET.shape, generator=g)
        sc[TARGET | JIT] = -1.0; sc[:, :64] = -1.0
        k = int(TARGET.sum())
        flat = sc.flatten()
        RAND = torch.zeros_like(flat, dtype=torch.bool)
        RAND[flat.topk(min(k, int((flat > 0).sum()))).indices] = True
        RAND = RAND.view(TARGET.shape)
        ELSE = ~TARGET & ~JIT & ~RAND; ELSE[:, :64] = False
        fams[name] = dict(T=TARGET, J=JIT, R=RAND, E=ELSE, n=k)
        print(f"{name}: n={k}", flush=True)

    hooks = [H[L].attn.c_proj.register_forward_pre_hook(mk_hook(L)) for L in range(18)]

    def ce_all(layer):
        CUR['layer'] = layer
        acc = {f: {k: [0.0, 0] for k in 'TJRE'} for f in fams}
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            for f in fams:
                for kk in 'TJRE':
                    mm = fams[f][kk][i:i + 8].to(DEV)
                    acc[f][kk][0] += float(ce[mm].sum()); acc[f][kk][1] += int(mm.sum())
        return {f: {kk: acc[f][kk][0] / max(acc[f][kk][1], 1) for kk in 'TJRE'}
                for f in fams}

    base = ce_all(None)
    ladders = {f: {} for f in fams}
    for L in range(18):
        r = ce_all(L)
        for f in fams:
            ladders[f][L] = {kk: round(r[f][kk] - base[f][kk], 4) for kk in 'TJRE'}
        print(f"a{L}: " + " | ".join(
            f"{f} T {ladders[f][L]['T']:+.3f} J {ladders[f][L]['J']:+.3f}"
            for f in fams), flush=True)
        json.dump({'partial': True, 'ladders': {f: {str(x): ladders[f][x]
                  for x in ladders[f]} for f in fams}}, open(OUT, 'w'), indent=1)
    for hk in hooks:
        hk.remove()

    verdicts = {}
    for f in fams:
        dmg = {L: ladders[f][L]['T'] for L in range(18)}
        top = max(dmg, key=lambda L: dmg[L])
        others = [dmg[L] for L in range(18) if L != top]
        conc = dmg[top] / max(sum(others) / len(others), 1e-4)
        jit = ladders[f][top]['J']
        clean = jit <= max(0.010, 0.33 * max(dmg[top], 1e-4))
        verdicts[f] = {'top': int(top), 'top_dmg': round(dmg[top], 4),
                       'conc': round(conc, 2), 'jitter_at_top': round(jit, 4),
                       'controls_clean': bool(clean)}
        print(f"{f}: top a{top} dmg {dmg[top]:+.4f} conc {conc:.1f}x "
              f"jit {jit:+.4f} clean {clean}", flush=True)

    se_v = verdicts['sentence_end']; ob_v = verdicts['open_bracket']
    pa = (se_v['conc'] < 3.0) or (not se_v['controls_clean'])
    pb = ob_v['top'] in (16, 17)
    pc = ob_v['controls_clean']
    out = {'n': {f: fams[f]['n'] for f in fams},
           'ladders': {f: {str(L): ladders[f][L] for L in range(18)} for f in fams},
           'verdicts': verdicts,
           'pred_a_sentence_end_fails_a_bar': bool(pa),
           'pred_b_open_bracket_late': bool(pb),
           'pred_c_open_bracket_clean': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
