# mdl_bill: THE PER-FUNCTION BILL (§1365 queue; the registry's open MDL-bridge item).
# For each closed kit, itemize: heads kept + parameter counts (raw counts — gauge-
# invariant per balanced_gauge_spec WP1), gate/probe description costs, and capability-
# nats recovered. SHARED HEADS PRICED ONCE across kits (10.5 serves question + the
# comparative refine stage; 17.2/17.3 serve exclaim + ellipsis).
#
# Numbers are pulled from the committed results jsons of the closing runs — this is an
# ACCOUNTING run (CPU, no forwards). Per-head attention parameters in bilin18: six
# 128-slice maps (c_q, c_k, c_q2, c_k2, c_v rows + c_proj cols) = 6 * D * 128.
# Kit-level extras: gates computed from the token stream = ~0 bits; the exclaim probe =
# its ridge vector (1152 or 2304 floats); route = per-layer lambda scalars (~free,
# counted as 18 floats); live patterns are weights-functions shared by every kit
# (window-foldable, §1161-66) and are listed once as a shared line, not per kit.
#
# Registered predictions:
#   pred_a every kit lands under 5M attention parameters.
#   pred_b shared-head dedup saves >= 15% of the naive per-kit sum.
#   pred_c the closer kit is the cheapest per capability-nat (one head, two surfaces).
import json

D = 1152; HEAD = 6 * D * 128            # 884,736 params per head
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mdl_bill_results.json'
OUTMD = PT + 'MDL_BILL.md'


def load(name):
    return json.load(open(PT + name))


def main():
    # ---- kit definitions: heads kept + the closing numbers (from committed jsons)
    q = load('question_kit_slim_results.json')
    q_heads = [tuple(int(x) for x in h.split('.')) for h in q['ranked_heads'][:16]] + [(10, 5)]
    q_rec = q['recovery']['kit16']['target']            # 0.641
    q_gap = q['ce']['ymean']['target'] - q['ce']['full']['target']

    c = load('comparative_refine_heads_results.json')
    c_heads = [(0, h) for h in range(9)] + [(1, h) for h in range(9)] + [(2, h) for h in range(9)]
    c_heads += [(8, 1), (10, 5), (12, 8), (11, 7), (11, 6)]
    c_rec = c['recovery']['kit_top4']['target']          # 0.778
    c_gap = c['ce']['ymean']['target'] - c['ce']['full']['target']

    k = load('closer_kit_results.json')
    kq = load('closer_quote_depth_results.json')
    cl_heads = [(L, h) for L in range(6) for h in range(9)] + [(13, 8)]
    cl_rec_b = k['recovery']['kit_bracket']['bracket']   # 0.657
    cl_rec_q = kq['recovery']['kit05']['quote']          # 0.630
    cl_gap_b = k['ce']['ymean']['bracket'] - k['ce']['full']['bracket']
    cl_gap_q = k['ce']['ymean']['quote'] - k['ce']['full']['quote']

    e = load('exclaim_probe_gate2_results.json')
    e_heads = [(L, h) for L in range(6) for h in range(9)] + [(17, 2), (17, 3)]
    e_rec = e['recovery']['gated']['target'] - e['recovery']['route']['target']  # kit increment
    e_rec_total = e['recovery']['gated']['target']       # 0.666
    e_gap = e['ce']['ymean']['target'] - e['ce']['full']['target']

    KITS = {
        'question': {'heads': set(q_heads), 'extra_params': 0,
                     'nats': q_rec * q_gap, 'rec': q_rec, 'gap': q_gap,
                     'desc': '16 gated heads + 10.5, clause gate'},
        'comparative': {'heads': set(c_heads), 'extra_params': 0,
                        'nats': c_rec * c_gap, 'rec': c_rec, 'gap': c_gap,
                        'desc': 'a02 gated + 8.1 + refine {10.5,12.8,11.7,11.6}'},
        'closer': {'heads': set(cl_heads), 'extra_params': 0,
                   'nats': cl_rec_b * cl_gap_b + cl_rec_q * cl_gap_q,
                   'rec': (cl_rec_b + cl_rec_q) / 2,
                   'gap': cl_gap_b + cl_gap_q,
                   'desc': 'a05 union-gated + 13.8 (brackets + quotes, one kit)'},
        'exclaim': {'heads': set(e_heads), 'extra_params': 2 * D,
                    'nats': e_rec_total * e_gap, 'rec': e_rec_total, 'gap': e_gap,
                    'desc': 'a05 probe-gated + pair (deploy-legal, recall-capped)'},
    }

    all_heads = set()
    naive_params = 0
    rows = []
    for name, kit in KITS.items():
        n_h = len(kit['heads'])
        params = n_h * HEAD + kit['extra_params'] + 18
        naive_params += params
        all_heads |= kit['heads']
        per_nat = params / max(kit['nats'], 1e-6)
        rows.append({'kit': name, 'heads': n_h, 'params': params,
                     'nats': round(kit['nats'], 3), 'recovery': round(kit['rec'], 3),
                     'params_per_nat': round(per_nat), 'desc': kit['desc']})
        print(f"{name}: {n_h} heads | {params/1e6:.2f}M params | "
              f"{kit['nats']:.2f} nats | {per_nat/1e6:.2f}M/nat", flush=True)

    dedup_params = len(all_heads) * HEAD + sum(k2['extra_params'] for k2 in KITS.values()) + 18
    saving = 1 - dedup_params / naive_params
    total_nats = sum(k2['nats'] for k2 in KITS.values())
    print(f"\nnaive sum {naive_params/1e6:.2f}M | dedup (shared heads once) "
          f"{dedup_params/1e6:.2f}M | saving {saving:.1%}", flush=True)
    print(f"unique heads {len(all_heads)} of 162 "
          f"({len(all_heads)/162:.0%} of attention heads touched)", flush=True)

    pa = all(r['params'] < 5_000_000 for r in rows)
    per_nat_rank = sorted(rows, key=lambda r: r['params_per_nat'])
    pb = saving >= 0.15
    pc = per_nat_rank[0]['kit'] == 'closer'
    out = {'per_head_params': HEAD, 'kits': rows,
           'naive_sum_params': naive_params, 'dedup_params': dedup_params,
           'dedup_saving': round(saving, 4), 'unique_heads': len(all_heads),
           'total_capability_nats': round(total_nats, 3),
           'shared_line_items': 'live patterns (162 window-foldable weights fns, '
                                'priced once model-wide, §1161-66); v1-route scalars; '
                                'generic mid-pool (+0.226 elsewhere-invariant, §1348)',
           'pred_a_under_5M_each': bool(pa), 'pred_b_dedup_15pct': bool(pb),
           'pred_c_closer_cheapest': bool(pc)}
    json.dump(out, open(OUT, 'w'), indent=1)

    md = ['# The per-function MDL bill (§1366)\n',
          f'Per-head cost: {HEAD:,} params (6 slices x 1152 x 128). Gates: ~0 bits '
          '(token-computable). Probe: 2,304 floats. Route: 18 scalars.\n',
          '| Kit | Heads | Params | Capability nats | Recovery | Params/nat |',
          '|---|---|---|---|---|---|']
    for r in rows:
        md.append(f"| {r['kit']} ({r['desc']}) | {r['heads']} | {r['params']/1e6:.2f}M | "
                  f"{r['nats']:.2f} | {r['recovery']:.3f} | {r['params_per_nat']/1e6:.2f}M |")
    md.append(f"\nNaive sum: {naive_params/1e6:.2f}M -> shared-heads-once: "
              f"{dedup_params/1e6:.2f}M (saving {saving:.1%}). Unique heads: "
              f"{len(all_heads)}/162.\n")
    open(OUTMD, 'w').write('\n'.join(md))
    print(f"pred_a under-5M {pa} | pred_b dedup {pb} ({saving:.1%}) | pred_c closer {pc}")
    print(f"wrote {OUT} and {OUTMD}")


if __name__ == '__main__':
    main()
