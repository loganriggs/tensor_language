"""Post-result grouped uncertainty and internal split provenance; no new gates."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent


def main():
    out=ROOT/'CORRELATIVE_FOLDED_EFFECTS_V1_RESULT.json'
    assert not out.exists()
    source=ROOT/'CORRELATIVE_FOLDED_PROGRAM_V1_RESULT.json';r=json.loads(source.read_text())
    artifact=torch.load(ROOT/'CORRELATIVE_REPLAYABLE_INTERFACE_V1_ARTIFACT.pt',map_location='cpu',weights_only=True)
    fit_groups={row['group_id'] for row in artifact['fit_rows']}
    fit_ids={row['row_id'] for row in artifact['fit_rows']}
    first=r['reports']['A1'];scale=float(np.median(np.array(first['donor_oriented_native_margin'])+np.array(first['base_oriented_margin']['native'])))
    rng=np.random.default_rng(9111274);panels={}
    for name,p in r['reports'].items():
        rows=p['rows'];groups=sorted({row['group_id'] for row in rows})
        idx=[np.array([i for i,row in enumerate(rows) if row['group_id']==group]) for group in groups]
        draws=rng.integers(0,len(groups),size=(4000,len(groups)))
        base=np.array(p['base_oriented_margin']['native']);donor=np.array(p['donor_oriented_native_margin'])
        patched=np.array(p['base_oriented_margin']['donor_folded']);den=donor+base
        recovery=float(np.mean((base-patched)/den)) if name in ('A1','A2') and np.all(den>1e-6) else None
        ce_native=np.array(p['base_answer_ce']['native']);ce_mean=np.array(p['base_answer_ce']['mean_folded']);delta=ce_mean-ce_native
        sums=np.array([delta[i].sum() for i in idx]);counts=np.array([len(i) for i in idx])
        boot=sums[draws].sum(1)/counts[draws].sum(1)
        panels[name]={'rows':len(rows),'groups':len(groups),'group_sizes':counts.tolist(),
                      'internal_fit_group_overlap':sorted(fit_groups&set(groups)),
                      'internal_fit_row_overlap':len(fit_ids&{row['row_id'] for row in rows}),
                      'authored_split_labels':sorted({row['split'] for row in rows}),
                      'donor_raw_recovery':recovery,
                      'donor_absolute_margin_movement_over_A1_scale':float(np.mean(np.abs(patched-base))/scale),
                      'mean_replacement_ce_damage':float(delta.mean()),
                      'mean_replacement_ce_ci95':np.quantile(boot,[.025,.975]).tolist(),
                      'mean_replacement_ce_positive_rows':int((delta>0).sum()),
                      'mean_replacement_margin_damage':float(np.mean(base-np.array(p['base_oriented_margin']['mean_folded']))),
                      'folded_vs_original_ce_maxabs':max(abs(a-b) for arm in ('donor','mean') for a,b in zip(p['base_answer_ce'][arm+'_folded'],p['base_answer_ce'][arm+'_original']))}
    result={'schema':'correlative.folded_effects.v1','panels':panels,'A1_scale':scale,'bootstrap_draws':4000,'bootstrap_seed':9111274,
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'scope':'Exploratory post-result group bootstrap on reused authored panels. P still carries target variable; P removal damage alone is not unrelated collateral. C is disjoint readout. Internal split differs from original FIT namespace; no fresh OOD claim.'}
    with out.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
