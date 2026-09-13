"""Can final scalar rounding alone account for the composition discrepancy?"""
import json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent

def main():
    result=json.loads((P/'FIXED_SELF_CONSUMERS_V1_RESULT.json').read_text())
    scores=np.array([c['scores'] for c in result['cells']]);f=scores.astype(np.float32)
    assert np.array_equal(f.astype(np.float64),scores)
    assert np.array_equal(scores[:,:,0],np.repeat(scores[:,0:1,0],4,axis=1))
    branch=np.array([1,-1,-1,1]);variant=np.array([1,1,-1,-1])
    residual=np.einsum('nvce,v,c->ne',scores,variant,branch)
    plus=np.nextafter(f,np.float32(np.inf)).astype(np.float64)-scores
    minus=scores-np.nextafter(f,np.float32(-np.inf)).astype(np.float64)
    # The pristine corner is structurally shared, so remove it before bounding.
    bound=.5*np.maximum(plus,minus)[:,:,1:].sum(axis=(1,2))
    effects=np.einsum('nvce,c->nve',scores,branch);allchange=effects[:,0]-effects[:,1]
    rows=[]
    for lo,hi,name in [(24*k,24*(k+1),'regional'+str(k)) for k in range(4)]+[(96+16*k,112+16*k,'FineWeb'+str(k)) for k in range(4)]:
        rr=residual[lo:hi];bb=bound[lo:hi];norm=np.linalg.norm(allchange[lo:hi],axis=0)
        rows.append(dict(group=name,observed_composition_relative_error=(np.linalg.norm(rr,axis=0)/norm).tolist(),
            entries_within_scalar_rounding_bound=(np.abs(rr)<=bb).sum(0).tolist(),rows=hi-lo,
            minimum_residual_beyond_scalar_rounding_relative_to_observed_change=(np.linalg.norm(np.maximum(np.abs(rr)-bb,0),axis=0)/norm).tolist(),
            max_absolute_composition_residual=np.max(np.abs(rr),axis=0).tolist()))
    out=dict(rows=rows,scope='Conservative nearest-FP32 final-scalar rounding interval,12nonsharedendpoints; nativecornercancels. Not a bound on intermediate body/RMS/softcap/logsoftmax errors. Relative denominators remain observed values. Compatibility with interval is not proof of a rounding explanation; original predicates unchanged.')
    (P/'FIXED_SELF_CONSUMER_ROUNDING_V1_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
