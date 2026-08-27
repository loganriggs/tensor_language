# digits_head_dispute: IS THE REGISTRY'S digits HEAD SET WRONG, OR IS THE SLICE?
#
# S1617 found that for `digits` the |lambda| eigen-slice resolves at head grain to
# a consistent NEIGHBOUR of every certified head, at every layer and both sites:
#     certified 7.3  -> slice says 7.8
#     certified 6.5  -> slice says 6.7
#     certified 12.6 -> slice says 12.7
#     certified 11.5 -> slice says 11.6
# Four for four is not noise. Either the slice is systematically off for this
# class, or the REGISTRY's digits assignment is wrong. comma and `and` produce
# EXACT hits at other layers, so it is not an indexing artifact.
#
# This settles it causally rather than by attribution. For each of the four
# layers, ablate the CERTIFIED head and the SLICE head separately with optimal
# constants (opt_ablation_consts_all.pt, the same mechanism circuit_verify_high.py
# uses) and compare digits-class CE rise and selectivity.
#
# PRIOR: the registry heads came from GREEDY CAUSAL search (S1513/S1515), which
# beat the weights-only top-5 on 4/4 classes. Causal search should outrank an
# attribution method, so the certified heads are expected to win. If they do not,
# a registry entry needs correcting.
#
# Local curated_rows.pt 3 x 333 (digits ~4357 positions), positions >=64,
# target-side mask, optimal-constant removal, one head at a time.
#
# Registered predictions:
#   pred_a the CERTIFIED head produces a larger digits-class CE rise than the
#          SLICE head in >= 3 of the 4 layers.
#   pred_b the CERTIFIED head has higher selectivity (class rise / global rise) in
#          >= 3 of the 4 layers.
#   pred_c CONTROL: at comma@L11, where S1617's slice HIT (certified 11.7), the
#          certified head beats the slice-disputed neighbour 11.6 on class rise --
#          confirming the comparison discriminates where the slice and registry AGREE.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'digits_head_dispute_results.json'
NR = 960
SITE = 11
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
EDIT = {'set': set(), 'V': None, 'mu': None}   # mu: {name: [2]}
FIN = {'on': False, 'V': None, 'mu': None}


def rx(pat):
    v = torch.zeros(50257, dtype=torch.bool)
    for t in range(50257):
        if re.match(pat, ENC.decode([t])):
            v[t] = True

CONSTS = torch.load(PT + 'opt_ablation_consts_all.pt', map_location='cpu')
HSET = {'set': []}
CHUNKS, ROWS_PER_CHUNK = 3, 333

# (layer, certified head, slice head) from S1617
DISPUTES = [(7, 3, 8), (6, 5, 7), (12, 6, 7), (11, 5, 6)]
DIGITS = r'^ ?[0-9]+$'
CONTROL = (11, 7, 6, r'^,$|^ ,$')      # comma: certified 11.7 vs neighbour 11.6


def mk_hook(L):
    def hook(mod, args):
        hs = [hh for (LL, hh) in HSET['set'] if LL == L]
        if not hs:
            return None
        x = args[0].clone()
        for hh in hs:
            x[:, :, hh * 128:(hh + 1) * 128] = \
                CONSTS[f'head{L}.{hh}'].to(DEV).float().to(x.dtype)
        return (x,)
    return hook


@torch.no_grad()
def fwd_plain(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def measure(chunks, mask_v):
    """Returns (class_ce, global_ce) under the current HSET."""
    cs = gs = 0.0; cn = gn = 0
    for ch in chunks:
        for i in range(0, ch.shape[0], 8):
            bb = ch[i:i + 8]
            idx = bb[:, :-1].to(DEV).contiguous(); tg = bb[:, 1:].to(DEV)
            lg = fwd_plain(idx).float()
            ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1),
                                 reduction='none').view_as(tg)
            valid = torch.ones_like(tg, dtype=torch.bool); valid[:, :64] = False
            cm = mask_v.to(DEV)[tg] & valid
            gm = valid & ~cm
            cs += float(ce[cm].sum()); cn += int(cm.sum())
            gs += float(ce[gm].sum()); gn += int(gm.sum())
    return cs / max(cn, 1), gs / max(gn, 1), cn


@torch.no_grad()
def main():
    import os
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    raw = torch.load(PT + 'curated_rows.pt', map_location='cpu')['rows']
    allr = raw[:CHUNKS * ROWS_PER_CHUNK, :T + 1].contiguous()
    chunks = [allr[c * ROWS_PER_CHUNK:(c + 1) * ROWS_PER_CHUNK] for c in range(CHUNKS)]
    hooks = [H[L].attn.c_proj.register_forward_pre_hook(mk_hook(L)) for L in range(18)]

    def run(mask_v, L, h):
        HSET['set'] = [(L, h)]
        c, g, n = measure(chunks, mask_v)
        HSET['set'] = []
        return c, g, n

    md = rx(DIGITS)
    HSET['set'] = []
    base_c, base_g, n_dig = measure(chunks, md)
    print(f"digits baseline: class {base_c:.4f} global {base_g:.4f} n={n_dig}", flush=True)

    rows = []
    for (L, cert_h, slice_h) in DISPUTES:
        cc, cg, _ = run(md, L, cert_h)
        sc, sg, _ = run(md, L, slice_h)
        cert = {'head': f'{L}.{cert_h}', 'class_rise': round(cc - base_c, 4),
                'global_rise': round(cg - base_g, 5),
                'sel': round((cc - base_c) / max(cg - base_g, 1e-6), 2)}
        slic = {'head': f'{L}.{slice_h}', 'class_rise': round(sc - base_c, 4),
                'global_rise': round(sg - base_g, 5),
                'sel': round((sc - base_c) / max(sg - base_g, 1e-6), 2)}
        rows.append({'layer': L, 'certified': cert, 'slice': slic,
                     'cert_wins_rise': cert['class_rise'] > slic['class_rise'],
                     'cert_wins_sel': cert['sel'] > slic['sel']})
        print(f"  L{L:<2d} certified {cert['head']:6s} rise={cert['class_rise']:+.4f} "
              f"sel={cert['sel']:8.2f}   |   slice {slic['head']:6s} "
              f"rise={slic['class_rise']:+.4f} sel={slic['sel']:8.2f}   "
              f"{'CERT' if cert['class_rise'] > slic['class_rise'] else 'SLICE'} wins rise", flush=True)

    Lc, hc, hn, cpat = CONTROL
    mc = rx(cpat)
    HSET['set'] = []
    cb_c, cb_g, cn2 = measure(chunks, mc)
    a_c, a_g, _ = run(mc, Lc, hc)
    b_c, b_g, _ = run(mc, Lc, hn)
    ctrl = {'layer': Lc, 'certified': f'{Lc}.{hc}', 'neighbour': f'{Lc}.{hn}',
            'cert_rise': round(a_c - cb_c, 4), 'neigh_rise': round(b_c - cb_c, 4),
            'cert_wins': (a_c - cb_c) > (b_c - cb_c), 'n': cn2}
    print(f"  CONTROL comma L{Lc}: certified {Lc}.{hc} rise={ctrl['cert_rise']:+.4f} vs "
          f"neighbour {Lc}.{hn} rise={ctrl['neigh_rise']:+.4f} -> "
          f"{'CERT' if ctrl['cert_wins'] else 'NEIGH'} wins", flush=True)
    for hk in hooks:
        hk.remove()

    nrise = sum(1 for r in rows if r['cert_wins_rise'])
    nsel = sum(1 for r in rows if r['cert_wins_sel'])
    pa = nrise >= 3
    pb = nsel >= 3
    pc = bool(ctrl['cert_wins'])

    out = {'config': {'disputes': DISPUTES, 'chunks': CHUNKS,
                      'rows_per_chunk': ROWS_PER_CHUNK, 'row_source': 'curated_rows.pt',
                      'rows_are_fresh': False, 'removal': 'optimal constants, one head at a time'},
           'digits_baseline': {'class_ce': round(base_c, 4), 'global_ce': round(base_g, 4),
                               'n': n_dig},
           'layers': rows, 'control_comma': ctrl,
           'cert_wins_rise': nrise, 'cert_wins_sel': nsel,
           'predictions': {'pred_a_cert_rise_ge3of4': bool(pa),
                           'pred_b_cert_sel_ge3of4': bool(pb),
                           'pred_c_control_cert_wins': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f"\n  certified wins rise {nrise}/4, sel {nsel}/4", flush=True)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)", flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)   # LESSONS 14


if __name__ == '__main__':
    main()
