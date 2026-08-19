"""Is the pronoun-vs-sentence-ender axis a cross-layer BUS variable? (L16 -> L17 edge)

Two independent measurements found what looks like the same variable at two sites:
§22: layer 16's #2 causal direction (40% of that layer) -- curvature positive on
pronouns/copulas, negative on sentence-enders, fires on enders.
§8.2: layer 17's #2 output direction -- writes toward determiners, subtracts a squared
projection onto sentence-final punctuation, adds one onto copulas/pronouns.

"The same axis appears twice" is a correlational observation. The causal-abstraction
claim worth testing is a GRAPH EDGE: layer 16's write along its axis is READ by layer
17's computation of its axis, i.e. the signal rides the residual bus between them. Three
tests, each with a control:

  T1 correlation: corr(c16, c17) across positions, against a grid of other direction
     pairs from the same two layers (is this pair special?).
  T2 the edge, interventionally: steer c16 by +/-2 sigma at every position and measure
     the shift in c17, in units of c17's natural sigma. Control: steer a
     Shapley-irrelevant layer-16 direction by the same amount. If the bus is real,
     steering the axis moves c17 specifically.
  T3 the output semantics: the same +/-2 sigma steering must move next-token
     determiner probabilities in the direction §8.2's reading predicts (copula/pronoun
     side up -> determiners more likely; ender side up -> less). Control token set
     matched in frequency. This is the "which text is affected" test at the output.
"""

import json
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import tiktoken
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_identifiable import form_for_direction
from tier2_model import rope_tables, apply_rot

NH, HD, D = 9, 128, 1152
enc = tiktoken.get_encoding('gpt2')
OUT = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
       'bilin18_syntax_bus_results.json')

C16_HOOK = None            # fn(c16_values) -> replacement values


@torch.no_grad()
def run(idx, want_logits=False):
    """Forward; optionally steer c16; returns (c16, c17, mean logprob shifts)."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    c16 = c17 = None
    for li in range(len(m.transformer.h)):
        blk = m.transformer.h[li]; x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        hcur = F.rms_norm(x, (D,))

        def qk(l):
            z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,))
            return apply_rot(z, cosb, sinb)

        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k1_, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k1_) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        xhat = F.rms_norm(x, (D,)); mlp = blk.mlp
        mo = mlp.Down(mlp.Left(xhat) * mlp.Right(xhat)) + mlp.Down_bias
        if li == 16:
            c = mo.float() @ RUN['d16']
            c16 = c.clone()
            if C16_HOOK is not None:
                cnew = C16_HOOK(c)
                mo = mo + ((cnew - c)[..., None] * RUN['d16']).to(mo.dtype)
        if li == 17:
            xf = xhat.float()
            c17 = torch.einsum('bti,ij,btj->bt', xf, RUN['M17'], xf)
        x = x + mo
    if not want_logits:
        return c16, c17, None
    logits = 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)
    lp = F.log_softmax(logits.float(), -1)
    return c16, c17, lp


RUN = {}


def main():
    global C16_HOOK
    t0 = time.time()
    # layer-16 axis: #2 causal direction from the battery
    rec16 = json.load(open('bilin18_layer16_battery_results.json'))
    phi16 = torch.tensor(rec16['b1']['phi'])
    accs = []
    for i in range(0, 60, 6):
        acc = []
        fwd(FW[i:i + 6, :513].to(DEV), collect=16, acc=acc)
        accs.append(acc[0])
    Y16 = torch.cat(accs)
    _, _, Vh16 = torch.linalg.svd((Y16 - Y16.mean(0)).float(), full_matrices=False)
    Q16 = orth(Vh16[:32].T)
    lead2 = int(phi16.argsort(descending=True)[1])
    d16 = Q16[:, lead2].float()
    ctrl_i = int(phi16.argsort()[2])        # a causally irrelevant direction
    d16c = Q16[:, ctrl_i].float()
    # layer-17 axis: #2 output PC, form from §8
    from bilin18_layer17 import out_pcs
    P17, mu17, ev17 = out_pcs(m, FW[:32, :513], 17, 8)
    M17 = form_for_direction(m.transformer.h[17].mlp, P17[1].float()).float()
    RUN.update({'d16': d16, 'M17': M17})
    out = {'d16_dir': lead2, 'ctrl_dir': ctrl_i}

    rows = FW[300:324, :257].to(DEV)
    c16, c17, _ = run(rows)
    s16, s17 = float(c16.std()), float(c17.std())

    # T1: is this pair special?
    print('== T1: correlation across positions ==')
    r_pair = float(torch.corrcoef(torch.stack([c16.flatten(), c17.flatten()]))[0, 1])
    base_rs = []
    for j in range(0, 12):
        dj = Q16[:, j].float()
        RUN['d16'] = dj
        cj, c17j, _ = run(rows)
        rj = float(torch.corrcoef(torch.stack([cj.flatten(), c17j.flatten()]))[0, 1])
        base_rs.append((j, rj))
    RUN['d16'] = d16
    base_abs = sorted(abs(r) for j, r in base_rs if j != lead2)
    out['t1'] = {'r_pair': r_pair, 'other_pairs_median': base_abs[len(base_abs)//2],
                 'other_pairs_max': base_abs[-1]}
    print(f'  corr(c16 axis, c17 axis) = {r_pair:+.3f}; other layer-16 directions vs '
          f'the same c17: median |r| {base_abs[len(base_abs)//2]:.3f}, max '
          f'{base_abs[-1]:.3f}')

    # T2: the edge, interventionally
    print('\n== T2: steer c16 by +/-2 sigma, measure c17 (in sigma_17 units) ==')
    res = {}
    for tag, dvec in (('axis', d16), ('control dir', d16c)):
        RUN['d16'] = dvec
        cnat, _, _ = run(rows)
        sd = float(cnat.std())
        moves = {}
        for sgn in (+1, -1):
            C16_HOOK = (lambda c, s=sgn, w=sd: c + s * 2 * w)
            globals()['C16_HOOK'] = C16_HOOK
            _, c17p, _ = run(rows)
            moves[sgn] = float((c17p - c17).mean()) / s17
            globals()['C16_HOOK'] = None
        res[tag] = moves
        print(f'  {tag:12s}: +2s -> Δc17 {moves[+1]:+.3f}σ | -2s -> Δc17 '
              f'{moves[-1]:+.3f}σ')
    RUN['d16'] = d16
    out['t2'] = res
    gain = abs(res['axis'][+1] - res['axis'][-1])
    gainc = abs(res['control dir'][+1] - res['control dir'][-1])
    out['t2']['specificity'] = gain / max(gainc, 1e-9)
    print(f'  edge specificity (axis swing / control swing): '
          f'{out["t2"]["specificity"]:.1f}x')

    # T3: output semantics
    print('\n== T3: does steering the axis move determiner probabilities? ==')
    dets = [enc.encode(w)[0] for w in (' the', ' a', ' an', ' their', ' our',
                                       ' your', ' his', ' her', ' its', ' my')]
    ctrl = [enc.encode(w)[0] for w in (' said', ' time', ' people', ' way',
                                       ' good', ' day', ' man', ' work',
                                       ' world', ' life')]
    _, _, lp0 = run(rows, want_logits=True)
    sem = {}
    for sgn in (+1, -1):
        globals()['C16_HOOK'] = (lambda c, s=sgn: c + s * 2 * s16)
        _, _, lp1 = run(rows, want_logits=True)
        globals()['C16_HOOK'] = None
        d_det = float((lp1[..., dets] - lp0[..., dets]).mean())
        d_ctl = float((lp1[..., ctrl] - lp0[..., ctrl]).mean())
        sem[sgn] = {'d_logprob_determiners': d_det, 'd_logprob_control': d_ctl}
        print(f'  {"+" if sgn>0 else "-"}2σ: Δ logprob determiners {d_det:+.4f} | '
              f'control tokens {d_ctl:+.4f}')
    out['t3'] = sem
    swing_det = sem[+1]['d_logprob_determiners'] - sem[-1]['d_logprob_determiners']
    swing_ctl = sem[+1]['d_logprob_control'] - sem[-1]['d_logprob_control']
    out['t3_swing'] = {'determiners': swing_det, 'control': swing_ctl}
    print(f'  full swing (+2σ vs −2σ): determiners {swing_det:+.4f}, control '
          f'{swing_ctl:+.4f}')

    out['runtime_s'] = time.time() - t0
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
