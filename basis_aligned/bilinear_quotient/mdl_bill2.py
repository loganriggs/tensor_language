# mdl_bill2: THE BILL, RE-PRICED WITH THE MEASURED COMMONS (§1368). Accounting run.
#
# Registered predictions:
#   pred_a the three-kit total (commons + specialists + probe-free gates) <= 28M params.
#   pred_b params-per-nat improves >= 1.8x vs the §1366 dedup bill (54.86M for these
#          kits' nats).
#   pred_c the average per-capability MARGINAL cost <= 1.5M params.
import json

D = 1152; HEAD = 6 * D * 128
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'


def main():
    s = json.load(open(PT + 'shared_band_kit_results.json'))
    n_commons = s['n_commons']
    commons_params = n_commons * HEAD + 18
    SPEC = {'question': [(10, 5)],
            'comparative': [(8, 1), (10, 5), (12, 8), (11, 7), (11, 6)],
            'closer': [(13, 8)]}
    uniq_spec = set()
    for v in SPEC.values():
        uniq_spec |= set(v)
    spec_params = len(uniq_spec) * HEAD

    ce = s['ce']
    def gap(fam):
        return ce['ymean'][fam] - ce['full'][fam]
    nats = (s['scores']['question'] * gap('question')
            + s['scores']['comparative'] * gap('comparative')
            + s['scores']['closer_b'] * gap('closer_b')
            + s['scores']['closer_q'] * gap('closer_q'))
    total = commons_params + spec_params
    per_nat = total / nats

    old = json.load(open(PT + 'mdl_bill_results.json'))
    old_three = old['dedup_params'] - HEAD * 2 - 2 * D  # remove exclaim-only pair+probe (approx)
    old_nats = sum(k['nats'] for k in old['kits'] if k['kit'] != 'exclaim')
    old_per_nat = old_three / max(old_nats, 1e-6)
    improve = old_per_nat / per_nat

    marginals = {k: len(set(v) - (uniq_spec - set(v))) for k, v in SPEC.items()}
    # marginal = heads not shared with any other kit, plus shared heads amortized
    marg_params = {}
    for k, v in SPEC.items():
        own = [h for h in v if sum(h in set(v2) for v2 in SPEC.values()) == 1]
        shared = [h for h in v if h not in own]
        marg_params[k] = len(own) * HEAD + len(shared) * HEAD / 2
    avg_marg = sum(marg_params.values()) / len(marg_params)

    pa = total <= 28_000_000
    pb = improve >= 1.8
    pc = avg_marg <= 1_500_000
    out = {'n_commons': n_commons, 'commons_params': commons_params,
           'unique_specialists': sorted(f'{a}.{b}' for a, b in uniq_spec),
           'spec_params': spec_params, 'total_params': total,
           'capability_nats': round(nats, 3), 'params_per_nat': round(per_nat),
           'improvement_vs_1366': round(improve, 2),
           'marginal_params': {k: round(v) for k, v in marg_params.items()},
           'avg_marginal': round(avg_marg),
           'pred_a_under_28M': bool(pa), 'pred_b_improves_18x': bool(pb),
           'pred_c_marginal_15M': bool(pc)}
    json.dump(out, open(PT + 'mdl_bill2_results.json', 'w'), indent=1)
    md = [
        '# The bill, re-priced on the commons (§1369)\n',
        f'- Commons: {n_commons} heads = {commons_params/1e6:.1f}M (serves every kit, twice-vetted)',
        f'- Specialists: {len(uniq_spec)} unique heads = {spec_params/1e6:.1f}M '
        f'({", ".join(sorted(f"{a}.{b}" for a, b in uniq_spec))})',
        f'- Gates: token-computed, ~0 bits. Route: 18 scalars.',
        f'- **Total: {total/1e6:.1f}M params ({total/546e6:.1%} of the model) for '
        f'{nats:.1f} capability-nats across four families (66-87% recovery each).**',
        f'- {per_nat/1e6:.2f}M params/nat — {improve:.1f}x better than the §1366 bill.',
        f'- Per-capability marginals (own heads + half-share of moonlighters): '
        + ", ".join(f"{k} {v/1e6:.2f}M" for k, v in marg_params.items()),
    ]
    open(PT + 'MDL_BILL.md', 'a').write('\n'.join(md) + '\n')
    for line in md:
        print(line)
    print(f"pred_a {pa} | pred_b {pb} ({improve:.2f}x) | pred_c {pc} ({avg_marg/1e6:.2f}M)")


if __name__ == '__main__':
    main()
