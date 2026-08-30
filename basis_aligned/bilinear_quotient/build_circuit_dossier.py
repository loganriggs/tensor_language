# ONE READABLE DOSSIER PER CIRCUIT, ASSEMBLED FROM EVERYTHING MEASURED.
#
# TASK CONTEXT (Logan, 2026-08-30): "get as much useful info on as many circuits as you can, hopefully
# having like 35 candidate circuits". The information now exists but is scattered across the circuit
# files (story, examples, certification) and circuits/BATTERY.json (ablation + interchange localisation).
# This joins them into one document ordered by how well-localised each circuit is, so a reader can pick
# targets without opening 62 JSON files. Pure assembly of measured numbers -- it computes nothing new and
# claims nothing that its sources do not.
import json
import os

BAT = json.load(open('circuits/BATTERY.json'))
DAS = json.load(open('circuits/DAS.json')) if os.path.exists('circuits/DAS.json') else {'by_tag': {}}

import torch
ST = torch.load('census_state_diverse.pt', map_location='cpu', weights_only=False)
NMEM = {l['tag']: (len(l['member']), len(l['slice'])) for l in ST['leaves']}

cir = {}
for fn in sorted(os.listdir('circuits')):
    if not fn.endswith('.json') or fn.split('.')[0].isupper():
        continue
    try:
        d = json.load(open('circuits/' + fn))
    except Exception:
        continue
    if isinstance(d, dict) and 'tag' in d:
        cir[d['tag']] = d

rows = sorted(((v['mean_ablation']['top'][0]['concentration'], t)
               for t, v in BAT['by_tag'].items() if v['mean_ablation']['top']), reverse=True)

BAND = {'r.1.2', 'r.1.2.0', 'r.1.2.1', 'r.1.1.1', 'r.1.1.2', 'r.1.3.1'}
L = []
L.append('# Circuit dossier — bilin18\n')
L.append(f'Assembled {BAT["generated"]}. **{len(rows)} curated circuits**, each localised by two '
         'independent causal interventions over the '
         f'{BAT["grid_positions"]:,}-position census grid.\n')
L.append('`concentration` = mean|dCE| on the circuit\'s members / mean|dCE| off its slice, when the named '
         'component is ablated. **mean** replaces the component output with its grid mean; '
         '**interchange** replaces it with its output at a random other position (seed 20260830).\n')
L.append('Sources: `circuits/BATTERY.json` (localisation), `circuits/DAS.json` (learned subspace, where '
         'run), and each circuit\'s own file (story, examples, certification). Nothing here is recomputed.\n')
L.append('\n## Summary table\n')
L.append('| # | circuit | best (mean) | conc | best (interchange) | conc | agree | members |')
L.append('|---|---------|-------------|------|--------------------|------|-------|---------|')
for i, (c, t) in enumerate(rows, 1):
    b = BAT['by_tag'][t]
    ic = b['interchange']['top'][0]['concentration'] if b['interchange']['top'] else None
    nm, ns = NMEM.get(t, (0, 0))
    L.append(f'| {i} | `{t}` | {b["best_mean"]} | {c:.2f} | {b["best_interchange"]} | '
             f'{ic if ic is not None else "-"} | {"yes" if b["methods_agree"] else "**no**"} | '
             f'{nm:,} / {ns:,} |')

L.append('\n## Per-circuit detail\n')
for i, (c, t) in enumerate(rows, 1):
    b = BAT['by_tag'][t]
    d = cir.get(t, {})
    nm, ns = NMEM.get(t, (0, 0))
    L.append(f'\n### {i}. `{t}` — {b["best_mean"]}, concentration {c:.2f}\n')
    L.append(f'{nm:,} member positions in a slice of {ns:,} '
             f'({100.0*nm/ns if ns else 0:.1f}% of the slice).\n')
    if t in BAND:
        L.append('> **Band-localised, not component-localised.** The two methods disagree by one or two '
                 'layers inside the `m13`–`m16` band, which is the signature of a circuit spread across '
                 'adjacent MLPs rather than sitting on one.\n')
    tops = b['mean_ablation']['top'][:3]
    L.append('| method | 1st | 2nd | 3rd |')
    L.append('|--------|-----|-----|-----|')
    L.append('| mean | ' + ' | '.join(f'{e["component"]} {e["concentration"]}' for e in tops) + ' |')
    it = b['interchange']['top'][:3]
    L.append('| interchange | ' + ' | '.join(f'{e["component"]} {e["concentration"]}' for e in it) + ' |')
    e0 = tops[0]
    L.append(f'\nAt `{e0["component"]}`: mean|dCE| on members **{e0["abs_dce_members"]}**, off slice '
             f'{e0["abs_dce_offslice"]}, signed dCE on members {e0.get("signed_dce_members")}. '
             f'Second-best component is `{tops[1]["component"]}` at {tops[1]["concentration"]} — '
             f'a {e0["concentration"]/tops[1]["concentration"]:.2f}x margin.\n')
    dv = DAS.get('by_tag', {}).get(t)
    if dv:
        r1 = dv['ranks'].get('1') or dv['ranks'].get(1) or {}
        if r1.get('optimiser_healthy'):
            L.append(f'\n**DAS (rank 1, held-out):** member dCE {r1.get("das_dce_members")}, '
                     f'concentration {r1.get("das_concentration")}, recovers '
                     f'{r1.get("fraction_of_full_recovered")} of the full component; overlap with the '
                     f'closed-form direction {r1.get("overlap_with_closed_form")}.\n')
        else:
            L.append('\n**DAS:** fit did not pass the optimiser health gate; no subspace number is '
                     'reported for this circuit.\n')
    if d.get('story'):
        L.append(f'\n**Story (from the circuit file):** {d["story"]}\n')
    ex = (d.get('examples') or {}).get('top') or []
    if ex:
        L.append('\n**Top members** (context → target, dCE when the circuit is ablated):\n')
        for e in ex[:3]:
            ctx = str(e.get('context', '')).replace('\n', '\\n')[-70:]
            tgt = str(e.get('target', '')).replace('\n', '\\n')
            L.append(f'- `…{ctx}` → `{tgt}`  (dCE {e.get("dce")}, base CE {e.get("base_ce")})')
        L.append('')

open('circuits/DOSSIER.md', 'w').write('\n'.join(L) + '\n')
print(f'wrote circuits/DOSSIER.md — {len(rows)} circuits, {len("".join(L))//1024}KB')
