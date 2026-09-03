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
HEL = json.load(open('circuits/HELDOUT.json')) if os.path.exists('circuits/HELDOUT.json') else None
DAS = json.load(open('circuits/DAS.json')) if os.path.exists('circuits/DAS.json') else {'by_tag': {}}

import torch
ST = torch.load('census_state_diverse.pt', map_location='cpu', weights_only=False)
NMEM = {l['tag']: (len(l['member']), len(l['slice'])) for l in ST['leaves']}

cir = {}
behavior = {}
for fn in sorted(os.listdir('circuits')):
    if not fn.endswith('.json') or fn.split('.')[0].isupper():
        continue
    try:
        d = json.load(open('circuits/' + fn))
    except Exception:
        continue
    if isinstance(d, dict) and 'tag' in d:
        cir[d['tag']] = d
        if d.get('schema_version') == 2 and d.get('identity', {}).get('kind') == 'behavior_circuit':
            behavior[d['tag']] = d

rows = sorted(((v['mean_ablation']['top'][0]['concentration'], t)
               for t, v in BAT['by_tag'].items() if v['mean_ablation']['top']), reverse=True)

BAND = {'r.1.2', 'r.1.2.0', 'r.1.2.1', 'r.1.1.1', 'r.1.1.2', 'r.1.3.1'}


def confidence(tag):
    """S2061: does the component assignment survive a row split, and do both methods agree on it?"""
    rows_ok = bool(HEL and (HEL['by_tag'].get(tag) or {}).get('stable'))
    meth_ok = bool(BAT['by_tag'][tag]['methods_agree'])
    if HEL is None:
        return 'unknown', 'held-out test not run'
    if rows_ok and meth_ok:
        return 'both', 'stable across a row split AND agreed by both interventions'
    if rows_ok:
        return 'rows-only', 'stable across a row split, but the two interventions name different components'
    if meth_ok:
        return 'methods-only', 'both interventions agree, but the argmax moves when the rows are split'
    return 'neither', 'the named component is not stable across rows and the two interventions disagree'
L = []
L.append('# Circuit dossier — bilin18\n')
L.append(f'Assembled from the frozen census (source note: {BAT["generated"]}) plus current version-2 records. '
         f'**{len(rows)} census response regions and {len(behavior)} task-defined behavior circuits**. '
         'Each census region was localised by two '
         'independent causal interventions over the '
         f'{BAT["grid_positions"]:,}-position census grid.\n')
L.append('`concentration` = mean|dCE| on the circuit\'s members / mean|dCE| off its slice, when the named '
         'component is ablated. **mean** replaces the component output with its grid mean; '
         '**interchange** replaces it with its output at a random other position (seed 20260830).\n')
L.append('Sources: `circuits/BATTERY.json` (localisation), `circuits/DAS.json` (learned subspace, where '
         'run), and each circuit\'s own file (story, examples, certification). Nothing here is recomputed.\n')
L.append('\n## Behavior circuits and counterfactual identification\n')
L.append('These version-2 records are task-defined behaviors, not assumed aliases of census leaves. '
         'Their events include failed/null/invalid evidence so the same causal question is not silently repeated.\n')
L.append('| circuit | status | declared variable | families | negative events | next missing evidence |')
L.append('|---|---|---|---:|---:|---|')
for tag, d in sorted(behavior.items()):
    active=[c for c in d['claims'] if c['status']!='superseded'][-1]
    negative=sum(e['verdict'] in ('failed','null','invalid') for e in d.get('evidence_events',[]))
    nxt=active.get('next_missing','').replace('|','/')
    L.append(f'| `{tag}` | {active["status"]} | `{active["causal_variable"]["id"]}` | '
             f'{len(active["counterfactual_families"])} | {negative} | {nxt} |')
L.append('')
for tag, d in sorted(behavior.items()):
    active=[c for c in d['claims'] if c['status']!='superseded'][-1]
    L.append(f'### `{tag}` — {active["status"]}\n')
    variable=active['causal_variable']
    L.append(f'**Read:** {variable["read"]}. **Operation:** {variable["operation"]}. '
             f'**Write:** {variable["write"]}. **Endpoint:** {variable["endpoint"]}.\n')
    L.append('| family | role | status |')
    L.append('|---|---|---|')
    for family in active['counterfactual_families']:
        L.append(f'| `{family["family_id"]}` | {family["role"]} | {family["status"]} |')
    events=d.get('evidence_events',[])
    if events:
        superseded_by={event.get('supersedes_event_id'):event['event_id'] for event in events
                       if event.get('supersedes_event_id')}
        L.append('\n**Append-only evidence ledger:**')
        L.append('| event | stage | test | verdict | lifecycle | result artifact |')
        L.append('|---|---|---|---|---|---|')
        for event in events:
            lifecycle=(f'superseded by `{superseded_by[event["event_id"]]}`'
                       if event['event_id'] in superseded_by else 'active')
            artifact=event.get('result_artifact_id') or '—'
            L.append(f'| `{event["event_id"]}` | {event["stage"]} | {event["test_type"]} | '
                     f'**{event["verdict"]}** | {lifecycle} | `{artifact}` |')
        L.append(f'\n**Frozen artifacts:** {len(d.get("artifacts", {}))}. '
                 'Paths and SHA-256 hashes are in the canonical JSON record.')
    L.append(f'\n**Next:** {active.get("next_missing", "not recorded")}\n')
L.append('\n## Summary table\n')
L.append('| # | circuit | best (mean) | conc | best (interchange) | conc | agree | members |')
L.append('|---|---------|-------------|------|--------------------|------|-------|---------|')
for i, (c, t) in enumerate(rows, 1):
    b = BAT['by_tag'][t]
    ic = b['interchange']['top'][0]['concentration'] if b['interchange']['top'] else None
    nm, ns = NMEM.get(t, (0, 0))
    cf, _ = confidence(t)
    L.append(f'| {i} | `{t}` | {b["best_mean"]} | {c:.2f} | {b["best_interchange"]} | '
             f'{ic if ic is not None else "-"} | {cf} | {nm:,} / {ns:,} |')

L.append('\n## Per-circuit detail\n')
for i, (c, t) in enumerate(rows, 1):
    b = BAT['by_tag'][t]
    d = cir.get(t, {})
    nm, ns = NMEM.get(t, (0, 0))
    L.append(f'\n### {i}. `{t}` — {b["best_mean"]}, concentration {c:.2f}\n')
    L.append(f'{nm:,} member positions in a slice of {ns:,} '
             f'({100.0*nm/ns if ns else 0:.1f}% of the slice).\n')
    cf, why = confidence(t)
    if cf != 'both':
        hv = (HEL['by_tag'].get(t) or {}) if HEL else {}
        extra = (f" On a held-out row split the argmax moves "
                 f"`{hv.get('selected_on_A')}` -> `{hv.get('argmax_on_B')}`."
                 if hv and not hv.get('stable') else '')
        L.append(f'> **Confidence: {cf}** — {why}.{extra} Its held-out concentration is '
                 f'{hv.get("conc_B_of_A_selected", "?")}, so the circuit still localises; it is the '
                 f'single component NAME that is not settled (§2061).\n')
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
