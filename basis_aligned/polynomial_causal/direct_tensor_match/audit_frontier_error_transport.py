"""Recover paired error inner products from natural/hybrid/change error energies."""
from pathlib import Path
import json,math
P=Path(__file__).parent;d=json.loads((P/'FRONTIER_SCALAR_DIAGNOSTIC_V1.json').read_text());rows=[]
for cohort in ('all','continuation','spaced_word'):
 for candidate in ('graph','separate'):
  energies={family:sum(r['error_energy'] for r in d['records'] if r['cohort']==cohort and r['candidate']==candidate and r['family']==family) for family in ('natural','hybrid','change')};counts={family:sum(r['sites'] for r in d['records'] if r['cohort']==cohort and r['candidate']==candidate and r['family']==family) for family in energies};assert len(set(counts.values()))==1
  a,b,change=[energies[k] for k in ('natural','hybrid','change')];inner=(a+b-change)/2;common=(a+b)/2-change/4;cosine=inner/math.sqrt(a*b);assert common>=-1e-10 and abs(cosine)<=1+1e-10
  rows.append(dict(cohort=cohort,candidate=candidate,error_energies=energies,paired_error_inner_product=inner,error_cosine=cosine,common_level_error_energy=common,half_change_error_energy=change/4,sites=counts['natural']))
comparisons=[]
for cohort in ('all','continuation','spaced_word'):
 g,b=[next(r for r in rows if r['cohort']==cohort and r['candidate']==name) for name in ('graph','separate')]
 comparisons.append(dict(cohort=cohort,common_level_error_ratio=math.sqrt(g['common_level_error_energy']/b['common_level_error_energy']),change_error_ratio=math.sqrt(g['error_energies']['change']/b['error_energies']['change']),graph_error_cosine=g['error_cosine'],baseline_error_cosine=b['error_cosine']))
out=dict(records=rows,comparisons=comparisons,scope='Exact paired-vector identities from aggregate squared errors on identical recipient/donor sites. Common-level error means mean of natural and hybrid errors; it need not be a constant bias or an invariant semantic feature. No mean subtraction, row selection, fitting or new-data claim.')
(P/'FRONTIER_ERROR_TRANSPORT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(comparisons,indent=2))
