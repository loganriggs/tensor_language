"""Paired uncertainty and exact accounting after the frozen scalar-network test."""
import hashlib,json
from pathlib import Path
import numpy as np
from paired_panel_bootstrap_v1 import PairedPanelBootstrap

def main():
    p=Path(__file__).resolve().parent;src=p/'GERUND_SCALAR_NETWORK_V1_RESULT.json';r=json.loads(src.read_text());reports={}
    for j,name in enumerate(('A1','A2','P','C')):
        row=r['reports'][name];bs=PairedPanelBootstrap(16,9111381+j);arms={}
        for key,arm in row['arms'].items():
            arms[key]={'recovery_ci95':bs.mean(arm['raw_recovery_per_row']),
                       'mean_ce_ci95':bs.mean(arm['ce_change_per_row']),
                       'mean_absolute_ce_ci95':bs.mean(np.abs(arm['ce_change_per_row'])),
                       'prediction_error_ci95':bs.relative_l2(arm['prediction_error_squared_per_row'],arm['live_effect_squared_per_row'])}
            if key.endswith('_swap'):
                direct=np.asarray(arm['direct_scalar_delta_per_row']);live=np.asarray(arm['final_scalar_delta_per_row'])
                arms[key]['scalar_feedback_mean']=float((live-direct).mean())
                arms[key]['scalar_feedback_ci95']=bs.mean(live-direct)
        removal={}
        for group in ('attention','mlp','all'):
            a=np.asarray(row['arms'][group+'_zero_base']['ce_change_per_row']);b=np.asarray(row['arms'][group+'_zero_donor']['ce_change_per_row'])
            removal[group]={'mean_ce':float(((a+b)/2).mean()),'mean_ce_ci95':bs.mean((a+b)/2),
                            'mean_absolute_ce':float(((np.abs(a)+np.abs(b))/2).mean()),
                            'absolute_ce_ci95':bs.mean((np.abs(a)+np.abs(b))/2)}
        terms=np.asarray(row['native_weighted_scalar_delta']);total=np.asarray(row['native_final_scalar_delta'])
        assert terms.shape==(18,2,16)
        source=terms.sum(0);means=terms.mean(2)
        reports[name]={'arms':arms,'removal':removal,'native_scalar_accounting':{
            'attention_mean':float(source[0].mean()),'mlp_mean':float(source[1].mean()),'total_mean':float(total.mean()),
            'closure_max_abs':float(np.abs(terms.sum((0,1))-total).max()),
            'mean_by_layer_and_kind':means.tolist(),
            'scope':'Signed sums of realized native writes, not independently removable causal shares.'}}
    # Minimal live-feedback counterexample: a scalar write is consumed by a square
    # written to a different coordinate. Prescribing the scalar does not fix that output.
    native=np.array([1.,1.]);predicted=np.array([2.,1.]);live=np.array([2.,4.])
    assert predicted[0]==live[0] and np.array_equal(live-predicted,[0.,3.])
    out={'schema':'gerund.scalar_network.audit.v1','source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'reports':reports,
         'toy_counterexample':{'native':native.tolist(),'frozen_prediction':predicted.tolist(),'live_state':live.tolist(),
                               'identical_scalar':True,'missing_complement_change':3.},
         'scope':'4000 paired resamples of16 authored groups per panel; no changed thresholds or independent population/OOD claim.'}
    with (p/'GERUND_SCALAR_NETWORK_AUDIT_V1_RESULT.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({k:{'source_means':v['native_scalar_accounting'],
                         'all_swap_error_ci95':v['arms']['all_swap']['prediction_error_ci95'],
                         'all_swap_recovery_ci95':v['arms']['all_swap']['recovery_ci95'],
                         'all_removal':v['removal']['all']} for k,v in reports.items()},indent=2))

if __name__=='__main__':main()
