# circuit_examples: extract REAL target examples for the circuit viewer (user request).
# For each circuit: build its target mask on fresh FineWeb rows, ablate its specialist
# component (per-head y-mean at the c_proj input, same estimator as the template runs),
# and record per-target CE damage. Output per circuit: top-5 targets by damage
# (ILLUSTRATIVE — selection criterion disclosed) + 5 seed-7 RANDOM targets, each with
# 14 tokens of context, the target continuation, and its dCE in nats.
# Not a registered experiment — a display extraction; no predictions, no verdicts.
import json, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'circuit_examples.json'
NMEAN = 24; NR = 480
H = m.transformer.h
CUR = {'spec': None, 'means': None}   # spec: {layer: [heads] or 'ALL'}


def mk_hook(L):
    def hook(mod, args):
        spec = CUR['spec']
        if spec is None or L not in spec:
            return None
        y = args[0].clone()
        mean = CUR['means'][L]            # (9,128)
        if spec[L] == 'ALL':
            y[:] = mean.reshape(-1).to(y.dtype)
        else:
            for h in spec[L]:
                y[..., h * 128:(h + 1) * 128] = mean[h].to(y.dtype)
        return (y,)
    return hook


@torch.no_grad()
def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


def stem(s):
    s = s.strip().lower()
    for suf in ('ing', 'es', 'ed', 's', 'd'):
        if s.endswith(suf) and len(s) - len(suf) > 3:
            return s[:-len(suf)]
    return s


@torch.no_grad()
def main():
    cl.use_state(PT + 'census_state_diverse.pt')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')

    # token sets
    COMP = ['bigger', 'smaller', 'better', 'worse', 'larger', 'greater', 'higher',
            'lower', 'faster', 'slower', 'older', 'younger', 'stronger', 'weaker',
            'easier', 'harder', 'longer', 'shorter', 'cheaper', 'richer', 'more', 'less',
            'fewer', 'rather']
    WH = ['Who', 'What', 'When', 'Where', 'Why', 'How', 'Which', 'Is', 'Are', 'Do',
          'Does', 'Did', 'Can', 'Could', 'Will', 'Would', 'Should']
    than_t, comp_t, qm_t, se_t, wh_t, close_t, open_t = (set() for _ in range(7))
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        ds = d.strip()
        if ds.lower() == 'than':
            than_t.add(tok)
        if ds.lower() in COMP:
            comp_t.add(tok)
        if '?' in d:
            qm_t.add(tok)
        if any(c in d for c in '.!?'):
            se_t.add(tok)
        if ds in WH:
            wh_t.add(tok)
        if ')' in d:
            close_t.add(tok)
        if '(' in d:
            open_t.add(tok)
    tt = lambda s: torch.tensor(sorted(s))
    than_i, comp_i, qm_i, se_i, wh_i, close_i, open_i = map(
        tt, (than_t, comp_t, qm_t, se_t, wh_t, close_t, open_t))

    ROWS = cl.fineweb_rows(NMEAN + NR, skip=40)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]
    toks = EVR[:, :-1]; tgt = EVR[:, 1:]

    # ---- masks -------------------------------------------------------------
    masks = {}
    is_comp = torch.isin(toks, comp_i)
    ctx = torch.zeros_like(is_comp)
    for w in range(2, 21):
        sh = torch.zeros_like(is_comp); sh[:, w:] = is_comp[:, :-w]; ctx |= sh
    masks['comparative'] = torch.isin(tgt, than_i) & ctx

    is_end = torch.isin(toks, se_i); is_wh = torch.isin(toks, wh_i)
    B2, T2 = toks.shape
    state = torch.zeros(B2, dtype=torch.bool)
    QS = torch.zeros_like(toks, dtype=torch.bool)
    rec = torch.full((B2,), 99, dtype=torch.long)
    for p in range(T2):
        op = is_wh[:, p] & (rec <= 2)
        state = torch.where(is_end[:, p], torch.zeros_like(state), state | op)
        QS[:, p] = state
        rec = torch.where(is_end[:, p], torch.zeros_like(rec), rec + 1)
    masks['question'] = torch.isin(tgt, qm_i) & QS

    is_open = torch.isin(toks, open_i); is_close = torch.isin(toks, close_i)
    depth = torch.zeros_like(toks)
    d_run = torch.zeros(B2, dtype=torch.long)
    for p in range(T2):
        d_run = (d_run + is_open[:, p].long() - is_close[:, p].long()).clamp_min(0)
        depth[:, p] = d_run
    masks['close_bracket'] = torch.isin(tgt, close_i) & (depth > 0)

    # stem-matcher: variant-supported positions (§1308 construction, exact-match excluded)
    sm = torch.zeros_like(toks, dtype=torch.bool)
    stem_cache = {}
    def st(tok):
        if tok not in stem_cache:
            stem_cache[tok] = stem(enc.decode([tok]))
        return stem_cache[tok]
    for r in range(B2):
        row = toks[r].tolist()
        seen = {}
        for p, tk in enumerate(row):
            s_ = st(tk)
            if p >= 64 and len(s_) > 3:
                if s_ in seen and seen[s_][1] != tk:
                    sm[r, p] = True
            if len(s_) > 3:
                seen[s_] = (p, tk)
    masks['stem_matcher'] = sm
    for k in masks:
        masks[k][:, :64] = False
        print(k, int(masks[k].sum()), flush=True)

    # ---- per-head y means from MEANR ---------------------------------------
    LAYERS_NEEDED = (8, 10, 1, 13)
    caps = {L: [] for L in LAYERS_NEEDED}
    hs = []
    for L in LAYERS_NEEDED:
        def mk(L):
            def h(mod, args):
                caps[L].append(args[0].detach().float().reshape(-1, 9, 128).mean(0))
                return None
            return h
        hs.append(H[L].attn.c_proj.register_forward_pre_hook(mk(L)))
    CUR['spec'] = None
    for i in range(0, NMEAN, 4):
        fwd(MEANR[i:i + 4, :-1].to(DEV).contiguous())
    for h in hs:
        h.remove()
    CUR['means'] = {L: torch.stack(caps[L]).mean(0) for L in LAYERS_NEEDED}

    hooks = [H[L].attn.c_proj.register_forward_pre_hook(mk_hook(L))
             for L in LAYERS_NEEDED]

    CIRCUITS = {'comparative': {8: [1]}, 'question': {10: [5]},
                'stem_matcher': {1: [1, 8]}, 'close_bracket': {13: 'ALL'}}

    def ce_map(spec):
        CUR['spec'] = spec
        rowsce = []
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg2 = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg2.reshape(-1),
                                 reduction='none').view(tg2.shape)
            rowsce.append(ce.cpu())
        return torch.cat(rowsce)

    base = ce_map(None)
    out = {}
    g = torch.Generator().manual_seed(7)
    for name, spec in CIRCUITS.items():
        abl = ce_map(spec)
        dce = (abl - base)
        M = masks[name]
        pos = torch.nonzero(M)
        if pos.numel() == 0:
            out[name] = {'n': 0}
            continue
        vals = dce[M]
        order = vals.argsort(descending=True)
        def render(r, p):
            pre = enc.decode(toks[r, max(0, p - 14):p + 1].tolist())
            nxt = enc.decode([int(tgt[r, p])])
            return {'context': pre, 'target': nxt,
                    'dce': round(float(dce[r, p]), 3),
                    'base_ce': round(float(base[r, p]), 3)}
        illus = [render(int(pos[i][0]), int(pos[i][1])) for i in order[:5].tolist()]
        ridx = torch.randperm(pos.shape[0], generator=g)[:5]
        rand = [render(int(pos[i][0]), int(pos[i][1])) for i in ridx.tolist()]
        out[name] = {'n': int(M.sum()),
                     'mean_dce_target': round(float(vals.mean()), 4),
                     'illustrative': illus, 'random': rand}
        print(f"{name}: n {out[name]['n']} mean dce {out[name]['mean_dce_target']}",
              flush=True)
    for h in hooks:
        h.remove()
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"wrote {OUT}")


if __name__ == '__main__':
    main()
