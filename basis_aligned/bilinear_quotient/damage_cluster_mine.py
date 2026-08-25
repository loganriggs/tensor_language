# damage_cluster_mine: REGISTRY GENERATOR — refill the empty candidate pool (§1409)
# from the DAMAGE side. Per-position dCE for each of the 18 attention-layer ablations
# (y -> layer mean at c_proj) on fresh rows; assign each position with max-dCE > 0.1 to
# its argmax layer; per layer, dump the top-30 target tokens by damage mass + a coarse
# class histogram (word/fragment/digit/punct/capitalized/newline/other). New-candidate
# rule: a class with >= 20% of a layer's damage mass whose (layer, class) pair is not in
# the named inventory. NR=960, skip=3200 (fresh rows). Assumptions registered.
#
# Registered predictions:
#   pred_a >= 2 layers show a >= 20% class NOT in the named inventory (pool refills).
#   pred_b a13's cluster is bracket/quote-dominated (close+quote+pipe classes >= 40%
#          of its mass) — rediscovery sanity.
#   pred_c a0 and a4 clusters are generic: no single class >= 15% of mass beyond
#          word/fragment (matching §1409's dirty screens).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'damage_cluster_mine_results.json'
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

    ROWS = cl.fineweb_rows(NMEAN + NR, skip=3200)[:, :T + 1].contiguous()
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

    hooks = [H[L].attn.c_proj.register_forward_pre_hook(mk_hook(L)) for L in range(18)]

    def ce_map(layer):
        CUR['layer'] = layer
        out = []
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            out.append(ce.cpu())
        return torch.cat(out)

    base = ce_map(None)
    DC = torch.zeros(18, *base.shape)
    for L in range(18):
        DC[L] = ce_map(L) - base
        print(f"a{L} mapped (mean dCE {float(DC[L][:, 64:].mean()):.4f})", flush=True)
    for hk in hooks:
        hk.remove()

    DC[:, :, :64] = 0.0
    mx, am = DC.max(0)
    OWNED = mx > 0.1

    def clas(t):
        d = enc.decode([t])
        ds = d.strip()
        if ')' in d or ']' in d or '}' in d:
            return 'close_brk'
        if '"' in d or "'" in d:
            return 'quote'
        if '|' in d:
            return 'pipe'
        if '\n' in d:
            return 'newline'
        if ds.isdigit():
            return 'digit'
        if len(d) >= 2 and d[0] == ' ' and d[1].isupper() and d[1:].isalpha():
            return 'capitalized'
        if ds.isalpha():
            return 'word' if d.startswith(' ') else 'fragment'
        if ds and all(not c.isalnum() for c in ds):
            return 'punct'
        return 'other'

    tgt = EVR[:, 1:]
    layers = {}
    for L in range(18):
        sel = OWNED & (am == L)
        n = int(sel.sum())
        if n == 0:
            layers[str(L)] = {'n': 0}
            continue
        dmg = mx[sel]
        toks = tgt[sel]
        mass = float(dmg.sum())
        # class histogram by damage mass
        hist = {}
        tok_mass = {}
        for t, d_ in zip(toks.tolist(), dmg.tolist()):
            c = clas(t)
            hist[c] = hist.get(c, 0.0) + d_
            tok_mass[t] = tok_mass.get(t, 0.0) + d_
        hist = {k: round(v / mass, 3) for k, v in sorted(hist.items(), key=lambda kv: -kv[1])}
        top_toks = sorted(tok_mass.items(), key=lambda kv: -kv[1])[:30]
        layers[str(L)] = {'n': n, 'mass': round(mass, 1), 'class_hist': hist,
                          'top_targets': [[repr(enc.decode([t])), round(v, 1)]
                                          for t, v in top_toks]}
        print(f"a{L}: n={n} mass={mass:.0f} hist={hist}", flush=True)
        json.dump({'partial': True, 'layers': layers}, open(OUT, 'w'), indent=1)

    KNOWN = {('13', 'close_brk'), ('13', 'quote'), ('13', 'pipe'),
             ('8', 'digit'), ('10', 'punct'), ('8', 'punct'),
             ('15', 'capitalized'), ('16', 'capitalized'), ('17', 'capitalized')}
    novel = []
    for L in range(18):
        info = layers[str(L)]
        if info.get('n', 0) < 30:
            continue
        for c, frac in info['class_hist'].items():
            if frac >= 0.20 and c not in ('word', 'fragment') \
                    and (str(L), c) not in KNOWN:
                novel.append({'layer': int(L), 'class': c, 'frac': frac,
                              'n': info['n']})
    a13h = layers.get('13', {}).get('class_hist', {})
    pb_mass = a13h.get('close_brk', 0) + a13h.get('quote', 0) + a13h.get('pipe', 0)
    pc_ok = True
    for L in ('0', '4'):
        hh = layers.get(L, {}).get('class_hist', {})
        for c, frac in hh.items():
            if c not in ('word', 'fragment') and frac >= 0.15:
                pc_ok = False
    pa = len({x['layer'] for x in novel}) >= 2
    out = {'layers': layers, 'novel_candidates': novel,
           'a13_bracket_quote_pipe_mass': round(pb_mass, 3),
           'pred_a_pool_refills': bool(pa), 'pred_b_a13_sanity': bool(pb_mass >= 0.40),
           'pred_c_a0_a4_generic': bool(pc_ok), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"novel: {novel}")
    print(f"pred_a {pa} | pred_b {pb_mass >= 0.40} ({pb_mass:.2f}) | pred_c {pc_ok}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
