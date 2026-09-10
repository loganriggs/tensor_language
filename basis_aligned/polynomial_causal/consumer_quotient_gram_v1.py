"""Joint sparse linear-reader metric, with explicit dependence checks.

Numerical row-space metric only. Does not certify dynamical/intervention closure.
"""
import numpy as np


def observed_squared_norm(margins,readers,tolerance=1e-10):
    if not 0<len(readers)<=256:raise ValueError('Require 1..256 sparse readers')
    gram=np.array([[sum(v*b.get(k,0.) for k,v in a.items()) for b in readers] for a in readers],dtype=float)
    values=np.asarray(margins,dtype=float)
    if values.ndim!=2 or values.shape[1]!=len(readers):raise ValueError('Expected rows x readers')
    if not np.isfinite(values).all() or not np.isfinite(gram).all():raise ValueError('Nonfinite reader data')
    inverse=np.linalg.pinv(gram,rcond=tolerance,hermitian=True)
    residual=values-values@inverse@gram
    if np.linalg.norm(residual)>tolerance*max(1.,np.linalg.norm(values)):
        raise ValueError('Reader values violate linear dependence at declared tolerance')
    result=float(np.sum((values@inverse)*values))
    if result < -tolerance:raise ValueError('Negative squared norm')
    return max(0.,result),gram


def controls():
    vector=np.array([3.,-1.,2.,-4.])
    families=[
        [{0:1.,1:-1.},{0:1.,2:-1.}],
        [{0:1.,1:-1.},{0:1.,2:-1.},{0:1.,1:-1.}],
        [{0:1.,1:-1.},{1:1.,2:-1.},{0:1.,2:-1.}],
        [{0:2.,1:-2.},{1:1.,2:-1.}]]
    errors=[];outputs=[]
    for readers in families:
        margins=np.array([[sum(vector[k]*v for k,v in r.items()) for r in readers]])
        score,gram=observed_squared_norm(margins,readers)
        outputs.append(score);errors.append(abs(score-26/3))
    readers=families[2];bad=np.array([[4.,-3.,2.]])
    rejected=False
    try:observed_squared_norm(bad,readers)
    except ValueError:rejected=True
    assert max(errors)<1e-12 and rejected
    naive=(4**2+1**2)/2
    assert abs(naive-outputs[0])>.1
    return {'passed':True,'max_exact_fixture_error':max(errors),
            'same_row_space_scores':outputs,'naive_independent_margin_score':naive,
            'inconsistent_cycle_rejected':rejected,
            'common_token_reader_cosine':.5,
            'scope':'Readout-coordinate geometry; common-token cosine is not evidence of shared hidden computation.',
            'model_forwards':0,'gpu_accessed':False}


if __name__=='__main__':
    import json
    print(json.dumps(controls(),indent=2))
