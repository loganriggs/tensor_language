"""Paired finite-panel uncertainty and carried/MLP scalar accounting; no refit."""
import hashlib,json
from pathlib import Path
import numpy as np
from paired_panel_bootstrap_v1 import PairedPanelBootstrap

def main():
    p=Path(__file__).resolve().parent;src=p/'GERUND_SHARED_READER_V1_RESULT.json';r=json.loads(src.read_text());out={}
    for j,name in enumerate(('A1','A2','P','C')):
        row=r['reports'][name];bs=PairedPanelBootstrap(16,9111371+j);arms={}
        for k,v in row['arms'].items():
            arms[k]={'raw_recovery_ci95':bs.mean(v['raw_recovery_per_row']),
                     'ce_change_ci95':bs.mean(v['ce_change_per_row']),
                     'absolute_ce_change_ci95':bs.mean(np.abs(v['ce_change_per_row']))}
        z=row['zero_removal'];a=np.asarray(z['base_ce_change_per_row']);b=np.asarray(z['donor_ce_change_per_row'])
        sm=np.asarray(row['scalars']['mlp_delta']);si=np.asarray(row['scalars']['input_delta']);sh=np.asarray(row['scalars']['final_delta'])
        out[name]={'arms':arms,'removal_mean_ce_ci95':bs.mean((a+b)/2),
            'removal_absolute_ce_ci95':bs.mean((np.abs(a)+np.abs(b))/2),
            'removal_base_harmed_count':int((a>0).sum()),'removal_donor_harmed_count':int((b>0).sum()),
            'scalar_accounting':{'mean_mlp_delta':float(sm.mean()),'mean_input_delta':float(si.mean()),'mean_final_delta':float(sh.mean()),
                                 'closure_max_abs':float(np.max(np.abs(sm+si-sh))),
                                 'mlp_to_total_l2_ratio':float(np.linalg.norm(sm)/max(np.linalg.norm(sh),1e-30)),
                                 'input_to_total_l2_ratio':float(np.linalg.norm(si)/max(np.linalg.norm(sh),1e-30))}}
    result={'schema':'gerund.shared_reader.audit.v1','source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'reports':out,
            'scope':'4000 paired resamples of16 authored lexical groups per panel, repeated verbs across frames not independent samples. Scalar ratios are accounting, not causal percentages; no changed thresholds.'}
    with (p/'GERUND_SHARED_READER_AUDIT_V1_RESULT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:{'scalar_final_recovery':v['arms']['scalar_final']['raw_recovery_ci95'],
                         'cross_verb_recovery':v['arms']['cross_verb']['raw_recovery_ci95'],
                         'removal_ce':v['removal_mean_ce_ci95'],'removal_abs_ce':v['removal_absolute_ce_ci95'],
                         'scalars':v['scalar_accounting']} for k,v in out.items()},indent=2))

if __name__=='__main__':main()
