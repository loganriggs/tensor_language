"""Native and unpruned references for the shared-node pruning decision."""
import json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent

def main():
    old=json.loads((P/'DIAGONAL_MLP10_NATIVE_V1_RESULT.json').read_text())
    new=json.loads((P/'FIXED_SELF_NODE_REMOVAL_V1_RESULT.json').read_text())
    def array(r,key):return np.array([c[key] for c in r['cells']])
    assert np.array_equal(array(old,'native_scores'),array(new,'native_scores'))
    truth=array(new,'native_interaction');before=array(old,'generated_interaction');after=array(new,'generated_interaction')
    groups=[]
    for lo,hi,name in [(24*k,24*(k+1),'regional'+str(k)) for k in range(4)]+[(96+16*k,112+16*k,'FineWeb'+str(k)) for k in range(4)]:
        ref=truth[lo:hi];a=after[lo:hi];b=before[lo:hi]
        groups.append(dict(name=name,pruning_change_relative_to_native=(np.linalg.norm(a-b,axis=0)/np.linalg.norm(ref,axis=0)).tolist(),
            pruned_sign_reversals=((a*ref)<0).sum(0).tolist(),material_sign_reversals=(((a*ref)<0)&(np.abs(ref)>=1e-5)).sum(0).tolist(),
            max_absolute_pruning_change=np.max(np.abs(a-b),axis=0).tolist()))
    result=dict(groups=groups,unchanged_generated_scalar_endpoints=int((array(old,'generated_scores')==array(new,'generated_scores')).sum()),
        total_scalar_endpoints=int(array(old,'generated_scores').size),
        scope='Same160historicalprefixes/nativecorners. Changed joint effect from removing only fixed Q0 from generated branches; magnitude relative to native full interaction. Not wholewriter removal or freshOOD selectivecircuit evidence.')
    (P/'FIXED_SELF_NODE_REMOVAL_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
