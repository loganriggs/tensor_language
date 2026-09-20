"""Positive solver control and native conflicting-input witnesses."""
import json
from pathlib import Path
import numpy as np
from audit_shared_selective_direction import solve
P=Path(__file__).parent;A=P.parent/'bilinear_quotient/circuits/followups'
def main():
    G=np.zeros((4,4,5));G[:,0,2]=-1;G[:,1,0]=1;G[:,2,1]=1;G[:,3,3]=1
    planted=solve([dict(gradient=G.tolist())]);assert abs(planted['worst_linear_retention']-1)<1e-10
    G[1::2]*=-1;flipped=solve([dict(gradient=G.tolist())]);assert abs(flipped['worst_linear_retention']-planted['worst_linear_retention'])<1e-10
    old=json.loads((A/'five_source_modal_null_v1_result.json').read_text());witnesses={}
    for role in ['subject','attractor']:
        examples=[]
        for c in old['contexts']:
            if c['role']!=role or c['template'] not in ['near_greeted','outside_called']:continue
            for row,g in enumerate(c['gradient']):examples.append(dict(gradient=[g],panel=c['panel'],role=role,template=c['template'],row=row))
        retained=list(examples);calls=0
        for example in examples:
            trial=[x for x in retained if x is not example]
            if not trial:continue
            result=solve(trial);calls+=1
            if result['worst_linear_retention']<=1e-10:retained=trial
        result=solve(retained);restored=[]
        for index in range(len(retained)):
            reduced=retained[:index]+retained[index+1:];r=solve(reduced);restored.append(r['worst_linear_retention']);assert r['worst_linear_retention']>1e-10
        witnesses[role]=dict(examples=retained,solution=result,retention_after_each_single_deletion=restored,greedy_lp_calls=calls,scope='Deletion-minimal for this order and numerical tolerance, not proven minimum-cardinality.')
    out=dict(planted=planted,sign_flip=flipped,witnesses=witnesses)
    (P/'SHARED_SELECTIVE_DIRECTION_CONFLICT_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print({role:dict(examples=len(v['examples']),retention_after_deletion=v['retention_after_each_single_deletion']) for role,v in witnesses.items()})
if __name__=='__main__':main()
