"""Independent scoring and fixed-input checks for the signed-source controls."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];ART=ROOT/'basis_aligned/bilinear_quotient/circuits/followups'

def main():
    records={v:json.loads((ART/f'subject_attention_freeze_v{v}_result.json').read_text()) for v in [692,693,694]}
    maxima={};native_diffs=[];score_diffs=[]
    for v,r in records.items():
        maxima[str(v)]={}
        assert r['initial_interface_suffix_calls']==32 and r['reuse_counts']['post11_suffix']==32
        for name,cells in r['hybrid_reports'].items():
            errors=[]
            for family,c in cells.items():
                true=np.array(c['target']);pred=np.array(c['prediction'])
                error=float(np.linalg.norm(pred-true)/np.linalg.norm(true));errors.append(error)
                score_diffs.append(abs(error-c['target_error']))
                ref=records[692]['hybrid_reports'][name][family]
                native_diffs.append(float(np.max(np.abs(true-ref['target']))))
                native_diffs.append(float(np.max(np.abs(np.array(c['native_modal'])-ref['native_modal']))))
            maxima[str(v)][name]=max(errors)
    assert max(native_diffs)==0 and max(score_diffs)<1e-12
    result=dict(native_targets_max_difference=max(native_diffs),scoring_max_difference=max(score_diffs),
        number_error_maxima=maxima,group_worst=records[694]['group_worst'],group6_replay=records[694]['max_group6_replay'],
        scope='Signed failures persist after dense quadratic restoration and paired-block native composition; full six-block native control passes. No all-rank lower bound or compressed adoption.')
    Path(__file__).with_name('SIGNED_SOURCE_CPU_AUDIT_2026-09-20.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
