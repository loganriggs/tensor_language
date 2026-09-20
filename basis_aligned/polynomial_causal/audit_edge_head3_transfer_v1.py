from pathlib import Path
import json
p=Path(__file__).resolve().parent;a=p.parent/'bilinear_quotient/circuits/followups';r=json.loads((a/'v4_edge_head3_transfer_v1_result.json').read_text());records=[]
for x in r['records']:
 if x['arm']!='head3' or x['passed']:continue
 def same(y):return all(y[k]==x[k] for k in ['panel','family','subject_amplitude','attractor_amplitude'])
 carry=next(y for y in r['records'] if y['arm']=='carry' and same(y));exact=next(y for y in r['records'] if y['arm']=='exact' and same(y))
 # Reverse triangle bounds difference between head3 and fullcarry output vectors.
 lower=[abs(e-c) for e,c in zip(x['errors'],carry['errors'])]
 records.append(dict(panel=x['panel'],family=x['family'],subject=x['subject_amplitude'],attractor=x['attractor_amplitude'],head3_errors=x['errors'],carry_errors=carry['errors'],exact_errors=exact['errors'],head3_vs_fullcarry_difference_lower_bounds=lower,weak_norm=x['weak_norm']))
assert len(records)==3 and r['predictions']['pred_a_instrument']
out=p/'EDGE_HEAD3_TRANSFER_CPU_AUDIT.json';assert not out.exists();out.write_text(json.dumps(dict(records=records,scope='Cell-matched negative-result audit; reverse-triangle bounds not causal allocation among omitted heads'),indent=2)+'\n')
print([(x['panel'],x['subject'],max(x['head3_errors']),max(x['carry_errors']),max(x['exact_errors'])) for x in records])
