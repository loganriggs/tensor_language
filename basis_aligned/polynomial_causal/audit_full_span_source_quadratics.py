"""Red-team rank-one success against six native settings spanning Sym(3)."""
import json
from pathlib import Path
import numpy as np
from test_shared_source_quadratics import PAIRS,SCALE,unpack
P=Path(__file__).parent;A=P.parent/'bilinear_quotient/circuits/followups'
def features(a):return np.array([a[i]*a[j]*(1 if i==j else 2**.5) for i,j in PAIRS])
def main():
    artifact=json.loads((P/'SOURCE_AMPLITUDE_QUADRATIC_V1.json').read_text());fit=json.loads((P/'SHARED_SOURCE_QUADRATICS_V1_RESULT.json').read_text());native=json.loads((A/'semantic_port_pairs_fresh_v1_result.json').read_text())
    keys=list(artifact['contexts']);coeff=np.array([artifact['contexts'][k] for k in keys]);lookup={k:i for i,k in enumerate(keys)};z=coeff[:,3:]*SCALE;h=unpack(z)
    basis=np.array(fit['dictionary']);ib=np.array(fit['input_basis']);proj=ib@ib.T
    hs=dict(full=h,shared2=unpack(z@basis.T@basis),input_rank2=proj[None]@h@proj[None])
    eigen,vectors=np.linalg.eigh(h);ix=np.argmax(abs(eigen),axis=1);v=vectors[np.arange(len(h)),:,ix];lam=eigen[np.arange(len(h)),ix];hs['context_rank1']=lam[:,None,None]*v[:,:,None]*v[:,None,:]
    settings={'middle_writes_4_7':[1,0,0],'mlp_8':[0,1,0],'mlp_10':[0,0,1],'pair23':[1,1,0],'pair24':[1,0,1],'pair34':[0,1,1]}
    original=np.array([features(a) for a in [[1,1,1],[-1,-1,-1],[1,-1,1],[2,2,2]]]);complete=np.array([features(a) for a in settings.values()]);assert np.linalg.matrix_rank(original)==2 and np.linalg.matrix_rank(complete)==6
    scores=[]
    for panel,roles in native['data'].items():
        for role,arms in roles.items():
            for arm,amplitude in settings.items():
                amp=np.array(amplitude,dtype=float)
                for family,c in arms[arm].items():
                    y=np.array(c['effects']);ids=[lookup[f'{panel}/{role}/{family}/{i}'] for i in range(len(y))];den=max(np.linalg.norm(y),1e-30);Bnorm=max(np.linalg.norm(arms['B'][family]['effects']),1e-30)
                    error={};budget={};absolute={}
                    for method,values in hs.items():
                        pred=-coeff[ids,:3]@amp-.5*np.einsum('i,bij,j->b',amp,values[ids],amp);num=np.linalg.norm(pred-y)
                        error[method]=float(num/den);budget[method]=float(num/Bnorm);absolute[method]=float(num/len(y)**.5)
                    scores.append(dict(panel=panel,role=role,family=family,arm=arm,target_norm=float(den),errors=error,B_budget_errors=budget,absolute_rms_errors=absolute))
    result=dict(original_quadratic_design_rank=2,complete_quadratic_design_rank=6,predictions=dict(pred_a_design_rank=True,pred_b_full_quadratic=all(c['errors']['full']<=.1 for c in scores),pred_c_context_rank1=all(c['errors']['context_rank1']<=.1 for c in scores)),max_own_effect_errors={k:max(c['errors'][k] for c in scores) for k in hs},max_B_budget_errors={k:max(c['B_budget_errors'][k] for c in scores) for k in hs},scores=scores,scope='Cross-audit of opened native intervention receipts; not new OOD and not a changed denominator for prior gates.')
    (P/'FULL_SPAN_SOURCE_QUADRATIC_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='scores'},indent=2))
if __name__=='__main__':main()
