"""Frozen native-corner and sign audit of five-bank full-branch evaluation."""
import json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent

def main():
    new=json.loads((P/'DIAGONAL_MLP10_NATIVE_V1_RESULT.json').read_text())
    old=json.loads((P/'COMPOSED_JOINT_NATIVE_FULL_V1_RESULT.json').read_text())
    native=np.array([c['native_scores'] for c in new['cells']])
    prior_native=np.array([c['native_scores'] for c in old['cells']]);assert np.array_equal(native,prior_native)
    def effect(scores):return scores[:,3]-scores[:,1]-scores[:,2]+scores[:,0]
    truth=effect(native);rows=[]
    for name,receipt in [('previous',old),('five_bank',new)]:
        predicted=effect(np.array([c['generated_scores'] for c in receipt['cells']]))
        flips=np.argwhere(predicted*truth<0)
        rows.append(dict(method=name,total_sign_reversals=len(flips),material_sign_reversals=int(((predicted*truth<0)&(np.abs(truth)>=1e-5)).sum()),
            flips=[dict(row=int(i),endpoint=int(j),reference=float(truth[i,j]),predicted=float(predicted[i,j])) for i,j in flips]))
    result=dict(native_corner_replay_exact=True,methods=rows,
        maximum_state_error=max(max(c['state_errors']) for c in new['cells']),
        scope='Same160historicalprefixes andexactnativecorners; materialthreshold1e-5 descriptive. Full goal properties and speed/adoption remainunproven.')
    (P/'DIAGONAL_MLP10_NATIVE_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
