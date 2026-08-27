# writer_slice_rule: DOES THE SLICE RULE DECIDE THE WRITER SET?
# S1604 found, unregistered, that at pronouns@mlp17 the top-2 POSITIVE writers of
# the pos_r8 payload are mlp12/mlp13 in 6/6 samples -- while S1598, same class and
# same site, reported writers mlp16/x0/mlp15/mlp9/mlp14/attn9 into the |lambda|-
# top-8 slice. Same (class, site), disjoint answers, because the slice rules
# differ. If that is general it qualifies EVERY writer-graph number in this
# program (S1597, S1598, and the theseus registry entries), so it is worth
# registering rather than leaving as an observation.
#
# Fixed cell: pronouns @ mlp17. Four slice rules at rank 8, one uniform harness,
# 3 disjoint 96-row samples each, no refitting:
#   abs   |lambda|-top-8          (S1598 rule; at this site 6 of 8 are negative)
#   pos   8 most POSITIVE         (S1604 payload rule)
#   neg   8 most NEGATIVE         (the gate)
#   rand  random orthonormal 8    (control: are these just high-norm components?)
# Overlap is Jaccard on the top-6 |signed contribution| writer sets.
#
# Registered predictions:
#   pred_a abs-vs-pos top-6 writer sets share <= 2 of 6 (the rules genuinely
#          disagree at a fixed cell; S1604 saw zero overlap on top-2).
#   pred_b Jaccard(abs, neg) > Jaccard(abs, pos) -- the |lambda| slice at this
#          suppression-dominated site is gate-dominated, so it should resemble
#          the explicit gate slice more than the payload slice.
#   pred_c the random control shares <= 2 of 6 with EVERY real rule, i.e. the
#          real writer sets are not merely the highest-norm components.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'writer_slice_rule_results.json'
NR = 960
SITE = 17
RANK = 8
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
    return v


def mk_cproj_hook(L):
    def hook(mod, args, output):
        nm = f'attn{L}'
        if nm not in EDIT['set']:
            return None
        o = output.float()
        pv = o @ EDIT['V']                       # [B,T,2]
        o = o - (pv - EDIT['mu'][nm]) @ EDIT['V'].T
        return o.to(output.dtype)
    return hook


def mk_mlp_hook(L):
    def hook(mod, args, output):
        o = None
        nm = f'mlp{L}'
        if nm in EDIT['set']:
            o = output.float()
            pv = o @ EDIT['V']
            o = o - (pv - EDIT['mu'][nm]) @ EDIT['V'].T
        return None if o is None else o.to(output.dtype)
    return hook


@torch.no_grad()
def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    if FIN['on']:
        xf = x.float()
        pv = xf @ FIN['V']
        x = (xf - (pv - FIN['mu']) @ FIN['V'].T).to(x.dtype)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def capture_fwd(idx, V2, lam2, acc, pm):
    """Exact manual forward through layer SITE, accumulating projections of
    every component output onto V2 (global + class sums), head-grain scores,
    mean_s, and the reconstruction check. pm: [B,T] class mask."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    vmask = torch.ones(B, T, dtype=torch.bool, device=DEV)
    vmask[:, :64] = False
    vf = vmask.reshape(-1); pf = pm.reshape(-1)
    tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))

    def add(nm, o):
        pv = (o.float().reshape(-1, D) @ V2)      # [N,2]
        acc['sum'][nm] += pv[vf].sum(0)
        acc['csum'][nm] += pv[pf].sum(0)

    add('x0', x0)
    for L, blk in enumerate(H):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        qp = at.c_q(xin).view(B, T, 9, 128).float()
        kp = at.c_k(xin).view(B, T, 9, 128).float()
        q2p = at.c_q2(xin).view(B, T, 9, 128).float()
        k2p = at.c_k2(xin).view(B, T, 9, 128).float()
        cos, sin = at.rotary(qp)
        q = are(F.rms_norm(qp, (128,)), cos, sin)
        k = are(F.rms_norm(kp, (128,)), cos, sin)
        q2 = are(F.rms_norm(q2p, (128,)), cos, sin)
        k2 = are(F.rms_norm(k2p, (128,)), cos, sin)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k) / 128.0) \
            * (torch.einsum('bqhd,bkhd->bhqk', q2, k2) / 128.0)
        pat = pat.masked_fill(~tril, 0.0)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
        ao = at.c_proj(y.reshape(B, T, D))
        add(f'attn{L}', ao)
        # head grain: y_h @ Wp_h.T projected on V2
        Wp = at.c_proj.weight.float()             # [D, D]
        for hh in range(9):
            M = Wp[:, hh * 128:(hh + 1) * 128].T @ V2      # [128,2]
            pv = (y[:, :, hh].float().reshape(-1, 128) @ M)
            acc['hsum'][L][hh] += pv[vf].sum(0)
            acc['hcsum'][L][hh] += pv[pf].sum(0)
        x = xm + ao
        mo = blk.mlp(F.rms_norm(x, (D,)))
        add(f'mlp{L}', mo)
        x = x + mo
    P = x
    acc['n'] += int(vf.sum()); acc['cn'] += int(pf.sum())
    acc['P_proj'].append((P.float().reshape(-1, D) @ V2)[vf].sum(0))


def _stack_order():
    order = ['x0']
    for L in range(18):
        order += [f'attn{L}', f'mlp{L}']
    return order


def _layer_of(name):
    if name == 'x0':
        return -1
    return int(name[4:]) if name.startswith('attn') else int(name[3:])


@torch.no_grad()
def depth_curve(rows, V2, lam2, mask_v):
    """Per-component signed class contribution, summed in stack order."""
    comps = ['x0'] + [f'attn{L}' for L in range(18)] + [f'mlp{L}' for L in range(18)]
    acc = {'sum': {c: torch.zeros(RANK, device=DEV) for c in comps},
           'csum': {c: torch.zeros(RANK, device=DEV) for c in comps},
           'hsum': {L: {h: torch.zeros(RANK, device=DEV) for h in range(9)} for L in range(18)},
           'hcsum': {L: {h: torch.zeros(RANK, device=DEV) for h in range(9)} for L in range(18)},
           'n': 0, 'cn': 0, 'P_proj': []}
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous(); tg = bb[:, 1:].to(DEV)
        pm = mask_v.to(DEV)[tg]
        pm[:, :64] = False
        capture_fwd(idx, V2, lam2, acc, pm)

    # exact depth-decay coefficients from the block lambdas (same as S1601)
    lam0 = [float(blk.lambdas[0]) for blk in H]
    lam1 = [float(blk.lambdas[1]) for blk in H]
    coef = {}
    for l in range(18):
        c = 1.0
        for k in range(l + 1, 18):
            c *= lam0[k]
        coef[f'attn{l}'] = c; coef[f'mlp{l}'] = c
    tx0 = 1.0
    for k in range(18):
        tx0 = lam0[k] * tx0 + lam1[k]
    coef['x0'] = tx0

    recon = sum(coef[c] * acc['sum'][c] for c in comps)
    Pv = torch.stack(acc['P_proj']).sum(0)
    rec_err = float((recon - Pv).abs().max() / Pv.abs().max())

    mu = {c: acc['sum'][c] / max(acc['n'], 1) for c in comps}
    cmu = {c: acc['csum'][c] / max(acc['cn'], 1) for c in comps}
    signed = {c: (coef[c] * (cmu[c] - mu[c])).sum().item() for c in comps}

    order = _stack_order()
    steps = [signed[c] for c in order]
    cum, run = [], 0.0
    for s in steps:
        run += s
        cum.append(run)
    peak_i = max(range(len(cum)), key=lambda i: cum[i])
    drop_i = min(range(len(steps)), key=lambda i: steps[i])
    peak, final = cum[peak_i], cum[-1]
    drawdown = (peak - final) / peak if peak > 0 else float('nan')
    return {'order': order, 'steps': steps, 'cum': cum,
            'peak_component': order[peak_i], 'peak_layer': _layer_of(order[peak_i]),
            'peak_value': peak, 'final_value': final, 'drawdown': drawdown,
            'largest_drop_component': order[drop_i],
            'largest_drop_value': steps[drop_i],
            'recon_rel_err': rec_err, 'class_n': acc['cn'], 'signed': signed}



SAMPLES = [15000, 20000, 25000]
TOPK = 6
PRON = r'^ (he|she|they|He|She|They)$'


def jaccard(a, b):
    A, B = set(a), set(b)
    return round(len(A & B) / max(len(A | B), 1), 3)


@torch.no_grad()
def main():
    import os
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    mask_v = rx(PRON)
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
    Lw = H[SITE].mlp.Left.weight.float(); Rw = H[SITE].mlp.Right.weight.float()
    Dw = H[SITE].mlp.Down.weight.float()
    Q = Lw.T @ ((u @ Dw)[:, None] * Rw)
    S = 0.5 * (Q + Q.T)
    lam, V = torch.linalg.eigh(S)

    gen = torch.Generator(device=DEV).manual_seed(1729)
    rand_basis, _ = torch.linalg.qr(torch.randn(D, RANK, device=DEV, generator=gen))
    rules = {
        'abs': V[:, lam.abs().argsort(descending=True)[:RANK]].contiguous(),
        'pos': V[:, lam.argsort(descending=True)[:RANK]].contiguous(),
        'neg': V[:, lam.argsort()[:RANK]].contiguous(),
        'rand': rand_basis.contiguous(),
    }
    eigs = {
        'abs': [round(float(x), 2) for x in lam[lam.abs().argsort(descending=True)[:RANK]]],
        'pos': [round(float(x), 2) for x in lam[lam.argsort(descending=True)[:RANK]]],
        'neg': [round(float(x), 2) for x in lam[lam.argsort()[:RANK]]],
        'rand': None,
    }
    print('abs eigs', eigs['abs'], flush=True)
    print('pos eigs', eigs['pos'], flush=True)
    print('neg eigs', eigs['neg'], flush=True)

    rows_cache = {s: cl.fineweb_rows(96, skip=s)[:, :T + 1].contiguous() for s in SAMPLES}
    out_rules = {}
    for name, V2 in rules.items():
        lam2 = (lam[lam.abs().argsort(descending=True)[:RANK]] if name == 'abs' else
                lam[lam.argsort(descending=True)[:RANK]] if name == 'pos' else
                lam[lam.argsort()[:RANK]] if name == 'neg' else
                torch.ones(RANK, device=DEV))
        per_sample_top = []
        for skip in SAMPLES:
            r = depth_curve(rows_cache[skip], V2, lam2.contiguous(), mask_v)
            signed = r.pop('signed')
            top = [c for c in sorted(signed, key=lambda c: -abs(signed[c]))[:TOPK]]
            per_sample_top.append(top)
            print(f"{name:5s} skip={skip:6d} n={r['class_n']:3d} top{TOPK}={top} "
                  f"recon={r['recon_rel_err']:.1e}", flush=True)
        # stability-weighted consensus: components appearing in the most samples
        counts = {}
        for t in per_sample_top:
            for i, c in enumerate(t):
                counts[c] = counts.get(c, 0) + (TOPK - i)
        consensus = sorted(counts, key=lambda c: -counts[c])[:TOPK]
        out_rules[name] = {'eigs': eigs[name], 'per_sample_top': per_sample_top,
                           'consensus_top': consensus}
        print(f"  -> {name} consensus top{TOPK}: {consensus}", flush=True)

    C = {n: out_rules[n]['consensus_top'] for n in rules}
    J = {f'{a}|{b}': jaccard(C[a], C[b])
         for a, b in [('abs', 'pos'), ('abs', 'neg'), ('pos', 'neg'),
                      ('rand', 'abs'), ('rand', 'pos'), ('rand', 'neg')]}
    shared_abs_pos = len(set(C['abs']) & set(C['pos']))
    rand_shared = {n: len(set(C['rand']) & set(C[n])) for n in ('abs', 'pos', 'neg')}

    pa = shared_abs_pos <= 2
    pb = J['abs|neg'] > J['abs|pos']
    pc = all(v <= 2 for v in rand_shared.values())

    out = {'config': {'class': 'pronouns', 'site': SITE, 'rank': RANK,
                      'samples': SAMPLES, 'topk': TOPK},
           'rules': out_rules, 'jaccard': J,
           'shared_abs_pos_of_6': shared_abs_pos, 'rand_shared': rand_shared,
           'S1598_reference': ['mlp16', 'x0', 'mlp15', 'mlp9', 'mlp14', 'attn9'],
           'predictions': {'pred_a_abs_pos_share_le2': bool(pa),
                           'pred_b_abs_closer_to_neg': bool(pb),
                           'pred_c_random_control_le2_all': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\njaccard {json.dumps(J)}", flush=True)
    print(f"abs&pos share {shared_abs_pos}/6 | rand shares {json.dumps(rand_shared)}", flush=True)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)", flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)   # LESSONS 14


if __name__ == '__main__':
    main()
