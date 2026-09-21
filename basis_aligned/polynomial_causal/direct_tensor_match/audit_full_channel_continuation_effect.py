"""Document-cluster uncertainty and nonuniformity after the opened-panel screen."""
from pathlib import Path
import json,numpy as np
p=Path(__file__).resolve().parent;d=json.loads((p/'FULL_CHANNEL_CONTINUATION_EFFECT_V1.json').read_text());rng=np.random.default_rng(941);out=[]
for s in d['summary']:
 rows=[r for r in d['records'] if all(r[k]==s[k] for k in ('domain','kind','cohort'))];E=np.array([r['error_energy'] for r in rows]);T=np.array([r['reference_energy'] for r in rows]);N=np.array([r['sites'] for r in rows]);native=np.array([r['native_ce_sum'] for r in rows]);student=np.array([r['student_ce_sum'] for r in rows]);ids=rng.integers(0,len(rows),(5000,len(rows)));valid=(T[ids].sum(1)>0)&(N[ids].sum(1)>0);ids=ids[valid];err=np.sqrt(E[ids].sum(1)/T[ids].sum(1));ce=(student[ids].sum(1)-native[ids].sum(1))/N[ids].sum(1);positive=T>0
 out.append(dict(domain=s['domain'],kind=s['kind'],cohort=s['cohort'],effect_error_ci95=np.quantile(err,[.025,.975]).tolist(),removal_ce_difference_ci95=np.quantile(ce,[.025,.975]).tolist(),worst_document_effect_error=float(np.sqrt(E[positive]/T[positive]).max()),documents_above_10pct=int((E[positive]>.01*T[positive]).sum()),documents_with_effect=int(positive.sum())))
result=dict(records=out,bootstrap_draws=5000,seed=941,scope='Document-cluster uncertainty on opened diagnostic panel; not independent validation or a replacement for registered point-estimate outcomes. Zero-site rows retained during cluster resampling; zero-denominator draws excluded.')
(p/'FULL_CHANNEL_CONTINUATION_EFFECT_AUDIT_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
