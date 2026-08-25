# damage_cluster_mine_heads: GENERATOR, HEAD GRAIN (§1413: pool empty; §1410 validated
# the layer-grain mine). Per-position dCE for each of the 162 head OV ablations (y-slice
# -> head mean at c_proj) on fresh rows, NR=480; assign positions with max-dCE > 0.15 to
# their argmax head; report class histograms + top targets for heads with n >= 50.
# Novelty vs the named inventory. skip=4400 (fresh rows).
#
# Registered predictions:
#   pred_a >= 3 novel (head, class) pairs at >= 25% mass with n >= 100 (pool refills).
#   pred_b 13.8's cluster is close+quote+pipe >= 50% of its mass (head-grain sanity).
#   pred_c >= 5 of the 12 committee members show capitalized >= 25% of their mass
#          (roster check from the damage side).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'damage_cluster_mine_heads_results.json'
NMEAN = 24; NR = 480
H = m.transformer.h
CUR = {'head': None, 'mean': {}}


def mk_hook(L):
    def hook(mod, args):
        if CUR['head'] is None or CUR['head'][0] != L:
            return None
        h = CUR['head'][1]
        y = args[0].clone()
        y[..., h * 128:(h + 1) * 128] = CUR['mean'][L][h].to(y.dtype)
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

    ROWS = cl.fineweb_rows(NMEAN + NR, skip=4400)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]

    caps = {L: [] for L in range(18)}
    hks = [H[L].attn.c_proj.register_forward_pre_hook(
        (lambda LL: lambda mod, args: caps[LL].append(
            args[0].detach().float().reshape(-1, 9, 128).mean(0)))(L)) for L in range(18)]
    CUR['head'] = None
    for i in range(0, NMEAN, 4):
        fwd(MEANR[i:i + 4, :-1].to(DEV).contiguous())
    for hk in hks:
        hk.remove()
    for L in range(18):
        CUR['mean'][L] = torch.stack(caps[L]).mean(0).to(DEV)

    hooks = [H[L].attn.c_proj.register_forward_pre_hook(mk_hook(L)) for L in range(18)]

    def ce_map(head):
        CUR['head'] = head
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
    DC = torch.zeros(162, *base.shape)
    for L in range(18):
        for h in range(9):
            i2 = L * 9 + h
            DC[i2] = ce_map((L, h)) - base
        print(f"a{L} heads mapped", flush=True)
    for hk in hooks:
        hk.remove()

    DC[:, :, :64] = 0.0
    mx, am = DC.max(0)
    OWNED = mx > 0.15

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
    for idx2 in range(162):
        L = f'{idx2 // 9}.{idx2 % 9}'
        sel = OWNED & (am == idx2)
        n = int(sel.sum())
        if n < 50:
            layers[str(L)] = {'n': n}
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
        print(f"{L}: n={n} mass={mass:.0f} hist={hist}", flush=True)
    json.dump({'partial': True, 'layers': layers}, open(OUT, 'w'), indent=1)

    COMMITTEE = {'13.0', '13.5', '14.4', '14.6', '14.7', '15.3', '16.0', '16.4',
                 '16.5', '17.0', '17.1', '17.2'}
    KNOWN = {('13.8', 'close_brk'), ('13.8', 'quote'), ('13.8', 'pipe'),
             ('8.3', 'digit'), ('8.7', 'digit'), ('10.5', 'punct'), ('8.1', 'word'),
             ('0.3', 'word'), ('0.3', 'fragment'), ('5.7', 'word')}
    KNOWN |= {(m, 'capitalized') for m in COMMITTEE}
    novel = []
    for name, info in layers.items():
        if info.get('n', 0) < 100:
            continue
        for c, frac in info['class_hist'].items():
            if frac >= 0.25 and c not in ('word', 'fragment') \
                    and (name, c) not in KNOWN:
                novel.append({'head': name, 'class': c, 'frac': frac, 'n': info['n']})
    h138 = layers.get('13.8', {}).get('class_hist', {})
    pb_mass = h138.get('close_brk', 0) + h138.get('quote', 0) + h138.get('pipe', 0)
    cap_hits = sum(1 for m in COMMITTEE
                   if layers.get(m, {}).get('class_hist', {}).get('capitalized', 0) >= 0.25)
    pa = len(novel) >= 3
    out = {'layers': {k: v for k, v in layers.items() if v.get('n', 0) >= 50},
           'novel_candidates': novel,
           'h138_mass': round(pb_mass, 3), 'committee_cap_hits': cap_hits,
           'pred_a_pool_refills': bool(pa), 'pred_b_138_sanity': bool(pb_mass >= 0.50),
           'pred_c_committee_5': bool(cap_hits >= 5), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"novel: {novel}")
    print(f"pred_a {pa} | pred_b {pb_mass >= 0.50} ({pb_mass:.2f}) | pred_c {cap_hits >= 5} ({cap_hits})")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
