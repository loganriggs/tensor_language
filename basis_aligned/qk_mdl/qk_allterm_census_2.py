"""SYNTHESIS pass over qk_allterm_census.json: (i) terms-to-95% profile across layers 0-17;
(ii) provenance flow map (which group-pairs dominate at which depth: energy share + single-term-kept
sufficiency per depth band); (iii) additive-vs-interaction anatomy per layer (diagonal-only vs
cross-only kept); (iv) per-layer table (floor, terms-to-95, named sufficient terms, dead groups)
+ concrete named-term examples for notable layers. Appends a 'synthesis' key to the JSON."""
import json
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
res = json.load(open(f'{QK}/qk_allterm_census.json'))
LAY = {int(k): v for k, v in res['layers'].items()}
NL = max(LAY) + 1

print(f"{'L':>2} {'floor':>8} {'SE':>7} {'prior':>8} {'match':>5} {'gate':>9} {'n95':>4} "
      f"{'sufficient terms':<38} {'diag':>8} {'cross':>8} {'dead/absent'}")
table = []
for L in range(NL):
    r = LAY[L]
    cf = r['configs']
    row = {'layer': L, 'floor': r['floor_dCE'], 'floor_SE': r['floor_SE'],
           'prior': r['floor_prior_ref'], 'match': r['floor_matches_prior'],
           'gate': r['gate']['recon_rel_err_global'], 'n95': r['terms_to_95pct'],
           'sufficient': r['sufficient_terms'],
           'diag_dCE': cf['diagonal']['dCE'], 'cross_dCE': cf['cross']['dCE'],
           'absent': r['groups_absent'], 'dead_causal': r.get('groups_dead_causal', []),
           'top1': r['energy_rank'][0],
           'top1_share': r['energy_shares'][r['energy_rank'][0]],
           'group_msq': r['group_msq_share_of_xpre']}
    table.append(row)
    print(f"{L:>2} {row['floor']:>8.4f} {row['floor_SE']:>7.5f} {row['prior']:>8.4f} "
          f"{str(row['match']):>5} {row['gate']:>9.2e} {str(row['n95']):>4} "
          f"{','.join(row['sufficient'] or ['-']):<38} {row['diag_dCE']:>8.4f} {row['cross_dCE']:>8.4f} "
          f"dead={row['dead_causal']} absent={row['absent']}")

# (ii) provenance flow: per-layer energy share aggregated into pair families
FAM = {}
for L in range(NL):
    sh = LAY[L]['energy_shares']
    for p, v in sh.items():
        FAM.setdefault(p, [0.0]*NL)[L] = v
print("\nPROVENANCE FLOW (centered energy share of each group-pair term, by layer):")
hdr = 'pair  ' + ' '.join(f'{L:>5}' for L in range(NL))
print(hdr)
for p in sorted(FAM, key=lambda p: -max(FAM[p])):
    if max(FAM[p]) < 0.01: continue
    print(f'{p:<6}' + ' '.join(f'{v:5.2f}' for v in FAM[p]))

# group involvement (sum of shares of terms involving each group)
GN = ['E', 'Ae', 'Ar', 'Me', 'Mr']
print("\nGROUP INVOLVEMENT (sum of energy shares of terms touching the group):")
print('grp   ' + ' '.join(f'{L:>5}' for L in range(NL)))
for g in GN:
    vals = []
    for L in range(NL):
        s = sum(v for p, v in LAY[L]['energy_shares'].items()
                if p.split('x')[0] == g or p.split('x')[1] == g)
        vals.append(s)
    print(f'{g:<6}' + ' '.join(f'{v:5.2f}' for v in vals))

# (iii) additive vs interaction: fraction of floor recovered by diagonal-only vs cross-only
print("\nANATOMY (fraction of floor REMAINING when keeping only diagonal / only cross terms):")
for L in range(NL):
    r = table[L]
    f = r['floor'] if r['floor'] else 1.0
    print(f"  L{L:2d} floor {r['floor']:+.4f}  diag-only leaves {r['diag_dCE']/f:5.1%}  "
          f"cross-only leaves {r['cross_dCE']/f:5.1%}")

res['synthesis'] = {
    'per_layer_table': table,
    'n95_profile': [t['n95'] for t in table],
    'flow_energy_share_by_layer': FAM,
    'notes': 'diag/cross fractions in per_layer_table; group involvement derivable from energy_shares'}
json.dump(res, open(f'{QK}/qk_allterm_census.json', 'w'), indent=1)
print("\nSYNTHESIS APPENDED")
