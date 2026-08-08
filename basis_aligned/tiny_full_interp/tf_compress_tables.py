"""Render the RESULTS.md tables straight from the measurement JSONs, so no
number in the write-up is transcribed by hand."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
S0 = f'{HERE}/tf_vanilla_d1_w128_b8192_s0_compress.json'
S1 = f'{HERE}/tf_vanilla_d1_w128_b8192_s1_compress.json'


def mb(x):
    return f'{x/1e6:.3f}'


def bits_at_kl(curve, kl):
    """Log-linear interpolation of a (bits, KL) curve to the bits the baseline
    would need to reach `kl`.  Used only to state "saves X% at matched KL"."""
    import math
    c = sorted(curve)
    for i in range(len(c) - 1):
        (b0, k0), (b1, k1) = c[i], c[i + 1]
        if (k0 - kl) * (k1 - kl) <= 0 and k0 > 0 and k1 > 0:
            f = (math.log(kl) - math.log(k0)) / (math.log(k1) - math.log(k0))
            return b0 + f * (b1 - b0)
    return None


def main():
    d = json.load(open(S0))
    fp32 = d['model']['fp32_bits']
    print(f"model: {d['model']['n_params']} params, fp32 {mb(fp32)} Mbit, "
          f"held CE {d['model']['ce']:.4f}, KL floor "
          f"{d['positive_control']['kl_identity']:.2e}")

    print('\n### T1 the model as its own description (uniform b-bit weights)\n')
    print('| bits/weight | description length | x smaller than fp32 | KL |')
    print('|---|---|---|---|')
    for r in d['A_self_quantisation']['rows']:
        if not r['scheme'].startswith('uniform_'):
            continue
        print(f"| {r['emb_bits']} | {mb(r['bits'])} Mbit | "
              f"{fp32/r['bits']:.1f}x | {r['kl']:.5f} |")

    print('\n### T2 the Pareto frontier (all families, everything charged)\n')
    fr = json.load(open(S0.replace('.json', '_frontier.json')))['frontier']
    print('| description length | x smaller | KL | scheme |')
    print('|---|---|---|---|')
    for p in fr:
        print(f"| {mb(p['bits'])} Mbit | {fp32/p['bits']:.1f}x | "
              f"{p['kl']:.5f} | `{p['scheme']}` |")

    print('\n### T3 coarsening the token axis: merging tokens is a bad code\n')
    b = d['B_split']['curves']
    print('| groups k | read-role KL | write-role KL | read, random grouping | '
          'write, random grouping |')
    print('|---|---|---|---|---|')
    for i, r in enumerate(b['read']):
        print(f"| {r['k']} | {r['kl']:.3f} | {b['write'][i]['kl']:.3f} | "
              f"{b['read_random'][i]['kl']:.3f} | "
              f"{b['write_random'][i]['kl']:.3f} |")

    print('\n### T4 per-role PRECISION (the currency that works)\n')
    j = d['J_roles']
    print('| bits/weight | read role coarsened only | write role only | both '
          '(tied) |')
    print('|---|---|---|---|')
    for i, r in enumerate(j['read_precision']):
        print(f"| {r['b']} | {r['kl']:.5f} | {j['write_precision'][i]['kl']:.5f}"
              f" | {j['both_precision'][i]['kl']:.5f} |")

    print('\n### T5 embedding schemes at ~matched bits\n')
    rows = sorted(d['C_embedding']['rows'] + d['G_codes']['rows'],
                  key=lambda r: r['bits_embedding'])
    print('| scheme | embedding bits | KL (body exact fp32) |')
    print('|---|---|---|')
    for r in rows:
        print(f"| `{r['scheme']}` | {mb(r['bits_embedding'])} Mbit | "
              f"{r['kl']:.5f} |")

    print('\n### T6 the body: structure vs precision\n')
    print('| scheme | body bits | KL (embedding exact fp32) |')
    print('|---|---|---|')
    for r in d['E_body']['rows']:
        print(f"| `{r['scheme']}` | {mb(r['bits_body'])} Mbit | {r['kl']:.5f} |")

    print('\n### T7 anchor rows + compressed tail (ported from ../qk_mdl)\n')
    print('| scheme | embedding bits | KL |')
    print('|---|---|---|')
    for r in d['D_anchor']['rows']:
        print(f"| `{r['scheme']}` | {mb(r['bits_embedding'])} Mbit | "
              f"{r['kl']:.5f} |")

    print('\n### T8 distillation vs post-hoc quantisation at the same bits\n')
    print('| budget (emb/body bits) | distilled bits | distilled KL | '
          'post-hoc bits | post-hoc KL |')
    print('|---|---|---|---|---|')
    for r in d['I_distilled']['rows']:
        c = r['control_posthoc']
        se = f" ± {r['kl_se']:.5f}" if 'kl_se' in r else ''
        print(f"| {r['emb_bits']}/{r['body_bits']} | {mb(r['bits'])} Mbit | "
              f"{r['kl']:.5f}{se} | {mb(c['bits'])} Mbit | {c['kl']:.5f} |")

    print('\n### T9 does the token SPELLING pay for its row?\n')
    k = d['K_features']
    print(f"feature regression R^2 on the embedding = {k['regression_r2']:.4f} "
          f"({k['n_features']} features)\n")
    print('| residual bits | feature-conditional bits | its KL | plain bits '
          'at the SAME KL | bits saved |')
    print('|---|---|---|---|---|')
    base = [(r['control_plain_q']['bits_embedding'],
             r['control_plain_q']['kl']) for r in k['rows']]
    for r in k['rows']:
        eq = bits_at_kl(base, r['kl'])
        sv = '—' if eq is None else f"{100*(1-r['bits_embedding']/eq):+.1f}%"
        print(f"| {r['b']} | {mb(r['bits_embedding'])} Mbit | {r['kl']:.5f} | "
              f"{'—' if eq is None else mb(eq)+' Mbit'} | {sv} |")

    if 'L_corpus_stats' in d:
        L = d['L_corpus_stats']
        print('\n### T9b does the CORPUS CO-OCCURRENCE statistic pay for it?\n')
        print(f"PPMI-SVD regression R^2 on the embedding = "
              f"{L['regression_r2']:.4f} ({L['n_features']} features)\n")
        print('| residual bits | conditional bits (statistic free) | its KL | '
              'plain bits at the SAME KL | bits saved |')
        print('|---|---|---|---|---|')
        base = [(r['control_plain_q']['bits_embedding'],
                 r['control_plain_q']['kl']) for r in L['rows']]
        for r in L['rows']:
            eq = bits_at_kl(base, r['kl'])
            sv = ('—' if eq is None
                  else f"{100*(1-r['bits_embedding']/eq):+.1f}%")
            print(f"| {r['b']} | {mb(r['bits_embedding'])} Mbit | "
                  f"{r['kl']:.5f} | {'—' if eq is None else mb(eq)+' Mbit'} | "
                  f"{sv} |")

    print('\n### T10 the weights-free artifacts, priced\n')
    print('| artifact | description length | x the fp32 model | KL |')
    print('|---|---|---|---|')
    for r in d['H_weightsfree']['rows']:
        if r.get('bits') is None:
            continue
        print(f"| `{r['scheme']}` | {mb(r['bits'])} Mbit | "
              f"{r['bits']/fp32:.1f}x | {r['kl']:.4f} |")

    if os.path.exists(S1):
        d1 = json.load(open(S1))
        print('\n### T11 CONFIRMATION on seed 1 (same cell, independent run)\n')
        print('| quantity | seed 0 | seed 1 |')
        print('|---|---|---|')
        print(f"| held CE | {d['model']['ce']:.4f} | {d1['model']['ce']:.4f} |")
        for nm in ('uniform_4bit', 'uniform_6bit', 'uniform_8bit'):
            a = [r for r in d['A_self_quantisation']['rows']
                 if r['scheme'] == nm][0]
            b_ = [r for r in d1['A_self_quantisation']['rows']
                  if r['scheme'] == nm][0]
            print(f"| KL, {nm} | {a['kl']:.5f} | {b_['kl']:.5f} |")
        for sec, key in (('F_combined', 'embT640+body8'),
                         ('F_combined', 'embT768+body8')):
            a = [r for r in d[sec]['rows'] if r['scheme'] == key]
            b_ = [r for r in d1.get(sec, {}).get('rows', [])
                  if r['scheme'] == key]
            if a and b_:
                print(f"| {key}: bits / KL | {mb(a[0]['bits'])} Mbit / "
                      f"{a[0]['kl']:.5f} | {mb(b_[0]['bits'])} Mbit / "
                      f"{b_[0]['kl']:.5f} |")
        if 'K_features' in d1:
            print(f"| feature regression R^2 | "
                  f"{d['K_features']['regression_r2']:.4f} | "
                  f"{d1['K_features']['regression_r2']:.4f} |")
        if 'J_roles' in d1:
            for i, r in enumerate(d['J_roles']['read_precision']):
                if r['b'] != 3:
                    continue
                r1 = d1['J_roles']['read_precision'][i]
                w = d['J_roles']['write_precision'][i]
                w1 = d1['J_roles']['write_precision'][i]
                print(f"| read-only / write-only KL at 3 bits | "
                      f"{r['kl']:.5f} / {w['kl']:.5f} | "
                      f"{r1['kl']:.5f} / {w1['kl']:.5f} |")
        if 'B_split' in d1:
            a = [r for r in d['B_split']['curves']['read'] if r['k'] == 512][0]
            b_ = [r for r in d1['B_split']['curves']['read'] if r['k'] == 512][0]
            print(f"| read clustering k=512 KL | {a['kl']:.3f} | "
                  f"{b_['kl']:.3f} |")


if __name__ == '__main__':
    main()
