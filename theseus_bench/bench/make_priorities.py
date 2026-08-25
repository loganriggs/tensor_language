"""Priority board: WHERE TO START LOOKING (user directive 2026-08-25).
Rank components by UNEXPLAINED GLOBAL CE = delta_opt x (1 - best_fidelity):
a barely-important head at 100% understood ranks below a big MLP at 50%.
Consumes the optimal-ablation sweep (delta_opt; live, partial-tolerant) and
registry/fidelity_seed.json (best-known fidelity, hand-curated until the
verifier writes results). Regenerate on every tick / frontier move."""
import json, os, time

BQ = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
TB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sweep = json.load(open(BQ + 'optimal_ablation_all_results.json'))
seed = {k: v for k, v in json.load(open(os.path.join(TB, 'registry',
        'fidelity_seed.json'))).items() if not k.startswith('_')}

rows = []
for name, v in sweep['results'].items():
    fid = seed.get(name, {}).get('fidelity', 0.0)
    ref = seed.get(name, {}).get('ref', 'baseline zoo only')
    if name.startswith('head'):
        layer = 'attn' + name[4:].split('.')[0]
        lf = seed.get(layer, {}).get('fidelity', 0.0)
        if lf > fid:
            fid, ref = lf, f'inherited from {layer} stand-in'
    # inert-component rule (S1444): delta below noise floor -> excluded
    if v['delta_mean'] < 0.002:
        continue
    unexplained = v['delta_opt'] * (1.0 - fid)
    rows.append((unexplained, name, v['delta_opt'], fid, ref))
rows.sort(reverse=True)
n_done = len(sweep['results'])
stamp = time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())

out = [f"# Priority board — where to start looking",
       "",
       f"Ranked by **unexplained global CE** = Δ_opt × (1 − best fidelity).",
       f"A low-importance head at 100% understood ranks below a big MLP at 50%.",
       f"Anchors from the optimal-ablation sweep ({n_done}/198 components so far;",
       f"attention layers land last). Generated {stamp}; regenerate with",
       f"`python bench/make_priorities.py` after any frontier move or sweep progress.",
       "",
       "## Top targets",
       ""]
for i, (u, name, d, fid, ref) in enumerate(rows[:10]):
    out.append(f"{i+1}. **{name}** — unexplained {u:.3f} nats "
               f"(Δ_opt {d:.3f}, fidelity {fid:.2f}) — {ref}")
out += ["", "## Full table", "",
        "| component | Δ_opt | best fidelity | unexplained CE | current best |",
        "|---|---|---|---|---|"]
for u, name, d, fid, ref in rows:
    out.append(f"| {name} | {d:.4f} | {fid:.2f} | {u:.4f} | {ref} |")
open(os.path.join(TB, 'registry', 'priorities.md'), 'w').write('\n'.join(out) + '\n')
print(f"wrote registry/priorities.md ({len(rows)} components); top 3:")
for u, name, d, fid, ref in rows[:3]:
    print(f"  {name}: unexplained {u:.3f} (delta {d:.3f}, fid {fid})")
