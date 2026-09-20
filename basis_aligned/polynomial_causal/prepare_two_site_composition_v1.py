"""Freeze paired source edits and composition budgets before joint outcomes."""
import hashlib
import json
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parent
source=P.parent/'bilinear_quotient/circuits/followups/source_ood_v2_result.json'
r=json.loads(source.read_text());assert r['predictions']['pred_a_instrument']
ctx={(c['panel'],c['role'],c['template']):c for c in r['contexts']}
records={(c['panel'],c['role'],c['family'],c['arm']):c for c in r['records']}
coordinates=[[1,0],[0,1],[-1,0],[0,-1],[.5,0],[0,.5],[-.5,0],[0,-.5],
             [1,1],[1,-1],[-1,1],[-1,-1],[.5,.5],[.5,-.5],[-.5,.5],[-.5,-.5]]
contexts=[];budgets=[];row_hashes={}
for panel in ['opposite','congruent']:
    path=P/f'SOURCE_OOD_V2_{panel.upper()}_ROWS.json'
    rows=json.loads(path.read_text());row_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    for template in dict.fromkeys(x['template'] for x in rows):
        entries=[x for x in rows if x['template']==template]
        assert all(x['subject_position']!=x['control_position'] for x in entries)
        amplitudes={role:ctx[panel,role,template]['amplitudes']['swap5'] for role in ['subject','attractor']}
        assert all(np.max(abs(np.array(a)[:,1]))==0 for a in amplitudes.values())
        contexts.append(dict(panel=panel,template=template,amplitudes=amplitudes))
        for family in dict.fromkeys(x['family'] for x in entries):
            ys=np.array(records[panel,'subject',family,'swap5']['target'])[:,0]
            ya=np.array(records[panel,'attractor',family,'swap5']['target'])[:,0]
            ns,na=np.linalg.norm(ys),np.linalg.norm(ya)
            assert min(ns,na)>1e-10
            budgets.append(dict(panel=panel,family=family,subject_number_norm=float(ns),attractor_number_norm=float(na),
                                individual_sum_ratio=float(np.linalg.norm(ys+ya)/(ns+na)),
                                individual_difference_ratio=float(np.linalg.norm(ys-ya)/(ns+na))))
out=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),row_sha256=row_hashes,
         implementation_sha256=hashlib.sha256((P/'two_axis_state_readout.py').read_bytes()).hexdigest(),
         coordinates=coordinates,contexts=contexts,budgets=budgets,
         scope='Opened texts and individual edits; new joint native outcomes not yet evaluated. Error budget at(s,t)=abs(s)*subject_norm+abs(t)*attractor_norm.',
         gates=dict(number_error=.1,modal_error=.05,native_replay=1e-4,exact_readout_replay=1e-10,jet_checks=1e-8),
         primary='Joint prediction plus both conditional increments on all64mixed cells. Eight axis settings are controls. Conditional attractor increment Y(s,t)-Y(s,0) uses abs(t)*attractor_norm; subject increment Y(s,t)-Y(0,t) uses abs(s)*subject_norm. Compare full quadratic, separable quadratic, and explicit-readout quadratic-state program.',
         amendment='Before joint native outcomes: preserve initial12setting binding; add four half-axis controls and conditional-increment gates because individual attractor effects are only0.35–10.9%ofsubject effects.')
(P/'TWO_SITE_COMPOSITION_V1R1_BINDING.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(contexts=len(contexts),base_cells=len(budgets),settings=len(coordinates),mixed_cells=64,
                     min_sum_ratio=min(x['individual_sum_ratio'] for x in budgets),min_difference_ratio=min(x['individual_difference_ratio'] for x in budgets)),indent=2))
