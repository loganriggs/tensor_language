"""Exact finite-sample least-squares bound for an operand-blind margin predictor.

The analytic identity is exact; the saved floating-point data calculation is
numerical. No population theorem, new threshold, fit selection, or model run.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def audit():
    source=Path(__file__).with_name('BILIN18_MLP1_OPERAND_DOMAIN_V1_RESULT.json')
    result=json.loads(source.read_text())
    assert result['predictions']['pred_a_instrument']
    panels={}
    for population,report in result['reports'].items():
        rows=report['rows']
        groups={'P':rows} if population=='controls' else {'temporal':rows[:24],'iswas':rows[24:]}
        for group,rs in groups.items():
            values=np.array([[r['arm_margin_change'][k] for k in ('left','right','symmetric','both')] for r in rs])
            left,right,symmetric,both=values.T
            midpoint=(left+right)/2
            irreducible=.5*float(np.sum((left-right)**2))
            excess=2*float(np.sum((symmetric-midpoint)**2))
            measured=float(np.sum((symmetric-left)**2+(symmetric-right)**2))
            reference=float(np.sum(left**2+right**2))
            closure=abs(measured-irreducible-excess)
            assert closure<=1e-10*max(measured,1.) and reference>0
            panels[group]={
                'rows':len(rs),'irreducible_squared_error':irreducible,
                'actual_symmetric_squared_error':measured,'nonlinear_midpoint_excess_squared_error':excess,
                'operand_blind_relative_joint_rms_floor':float(np.sqrt(irreducible/reference)),
                'actual_symmetric_relative_joint_rms':float(np.sqrt(measured/reference)),
                'irreducible_fraction_of_actual_squared_error':irreducible/measured if measured else None,
                'identity_closure_abs':closure,
                'both_minus_single_sum_margin_rms':float(np.sqrt(np.mean((both-left-right)**2))),
            }
    return {'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'panels':panels,'model_forwards':0,'new_data':False,'adoption':False,
        'scope':'Best possible identical prediction for the two operand labels on these saved margin-effect pairs. Decoder may vary freely by row. Not a lower bound for predictors retaining operand identity, full-distribution KL, or unseen data.'}


if __name__=='__main__':
    result=audit()
    with Path(__file__).with_name('OPERAND_IDENTITY_INFORMATION_V1_AUDIT.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
