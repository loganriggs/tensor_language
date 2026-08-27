# readwrite_2x2: IS THE READ-WRITE RELATIONSHIP A PROPERTY OF THE SITE OR THE CLASS?
# S1601: at mlp11, question -- the module is the LARGEST POSITIVE writer of the
#        subspace its own quadratic form reads (+1028). "Reads and writes the
#        same channel."
# S1604: at mlp17, pronouns -- the module writes strongly NEGATIVE into its own
#        all-positive payload subspace (-1374..-5111, 0/6 positive). The
#        coincidence does NOT generalise.
# Those two cells differ in BOTH class and site, so neither is identified. This
# fills the 2x2: {question, pronouns} x {mlp11, mlp17}, one uniform slice rule
# (pos_r8 payload, S1598) at every cell, 3 disjoint samples per cell.
# Codex declined this on the board at 04:06 ("your pronoun run can supply a
# suppression-dominated case"), so lane 1 owns it; no duplication.
# 96 rows/sample, positions >=64, target-side class masks, no refitting.
#
# Registered predictions (SITE, not class, decides the sign):
#   pred_a question@mlp17 self-write is NEGATIVE in 3/3 samples.
#   pred_b pronouns@mlp11 self-write is POSITIVE in 3/3 samples.
#   pred_c |self-write @mlp17| / |self-write @mlp11| > 2 for BOTH classes
#          (suppression at the late site dominates amplification at the mid site).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'readwrite_2x2_results.json'
NR = 960
RANK = 8          # pos_r8 payload slice at every cell (S1598 rule)
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



CLASSES = {'question': r'^\?$| \?$',
           'pronouns': r'^ (he|she|they|He|She|They)$'}
SITES = [11, 17]
SAMPLES = [15000, 20000, 25000]     # disjoint; skip=80 excluded (S1604 outlier)


@torch.no_grad()
def payload_slice(site, mask_v):
    """The RANK most POSITIVE eigendirections of the class-projected form at `site`."""
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
    Lw = H[site].mlp.Left.weight.float(); Rw = H[site].mlp.Right.weight.float()
    Dw = H[site].mlp.Down.weight.float()
    Q = Lw.T @ ((u @ Dw)[:, None] * Rw)
    S = 0.5 * (Q + Q.T)
    lam, V = torch.linalg.eigh(S)
    o = lam.argsort(descending=True)[:RANK]
    V2 = V[:, o].contiguous(); lam2 = lam[o].contiguous()
    assert bool((lam2 > 0).all()), f'payload slice at mlp{site} must be all-positive'
    return V2, lam2


@torch.no_grad()
def main():
    import os
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows_cache = {s: cl.fineweb_rows(96, skip=s)[:, :T + 1].contiguous() for s in SAMPLES}

    cells = {}
    for cname, pat in CLASSES.items():
        mask_v = rx(pat)
        for site in SITES:
            V2, lam2 = payload_slice(site, mask_v)
            key = f'{cname}@mlp{site}'
            self_writes, peaks, dds = [], [], []
            for skip in SAMPLES:
                r = depth_curve(rows_cache[skip], V2, lam2, mask_v)
                signed = r.pop('signed')
                sw = signed[f'mlp{site}']
                self_writes.append(sw); peaks.append(r['peak_component'])
                dds.append(r['drawdown'])
                print(f"{key:20s} skip={skip:6d} n={r['class_n']:3d} "
                      f"self_write={sw:+10.1f} peak={r['peak_component']:7s} "
                      f"dd={r['drawdown']:+.4f} recon={r['recon_rel_err']:.1e}", flush=True)
            cells[key] = {'class': cname, 'site': site,
                          'eigs': [round(float(x_), 3) for x_ in lam2],
                          'self_write': [round(x, 1) for x in self_writes],
                          'mean_self_write': round(sum(self_writes) / len(self_writes), 1),
                          'n_positive': sum(1 for x in self_writes if x > 0),
                          'n_negative': sum(1 for x in self_writes if x < 0),
                          'peaks': peaks, 'drawdowns': [round(x, 4) for x in dds]}
            print(f"  -> {key}: mean self_write {cells[key]['mean_self_write']:+.1f} "
                  f"({cells[key]['n_positive']}+/{cells[key]['n_negative']}-)", flush=True)

    q17, p11 = cells['question@mlp17'], cells['pronouns@mlp11']
    pa = q17['n_negative'] == 3
    pb = p11['n_positive'] == 3
    ratios = {}
    for c in CLASSES:
        a17 = abs(cells[f'{c}@mlp17']['mean_self_write'])
        a11 = abs(cells[f'{c}@mlp11']['mean_self_write'])
        ratios[c] = round(a17 / max(a11, 1e-9), 3)
    pc = all(v > 2.0 for v in ratios.values())

    sign_by_site = {f'mlp{s}': {c: ('+' if cells[f'{c}@mlp{s}']['mean_self_write'] > 0 else '-')
                                for c in CLASSES} for s in SITES}
    out = {'config': {'classes': list(CLASSES), 'sites': SITES, 'rank': RANK,
                      'samples': SAMPLES, 'rows_per_sample': 96,
                      'slice_rule': 'pos_rRANK payload (S1598), uniform at every cell'},
           'cells': cells, 'sign_by_site': sign_by_site,
           'abs_ratio_mlp17_over_mlp11': ratios,
           'reference': {'S1601_question_at_mlp11_lambda_top2': 1028.5,
                         'S1604_pronouns_at_mlp17_pos_r8': '-1374 to -5111'},
           'predictions': {'pred_a_q17_negative_3of3': bool(pa),
                           'pred_b_p11_positive_3of3': bool(pb),
                           'pred_c_abs_ratio_gt2_both': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nsign by site: {json.dumps(sign_by_site)}", flush=True)
    print(f"|mlp17|/|mlp11| ratios: {json.dumps(ratios)}", flush=True)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)", flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)   # LESSONS 14


if __name__ == '__main__':
    main()
