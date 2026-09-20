"""Local constrained modal-leakage bound and bounded candidate, no outcome fitting."""
import json
from pathlib import Path
import numpy as np
P=Path(__file__).parent;A=P.parent/'bilinear_quotient/circuits/followups'
def main():
    r=json.loads((A/'semantic_source_multiobservable_v1_result.json').read_text());assert r['predictions']['pred_a_instrument'];out=[];residual=[]
    for context in r['contexts']:
        g=np.array(context['gradient']);h=np.array(context['hessian'])
        for i,(gi,hi) in enumerate(zip(g,h)):
            n=-gi[0];M=-gi[1:];u,s,vt=np.linalg.svd(M);assert s[-1]>s[0]*1e-12
            b=np.linalg.solve(M.T,n);direction=np.linalg.solve(M,b)/(b@b);lower=1/np.linalg.norm(b);residual.append(abs(n@direction-1))
            unit=-gi@np.ones(3)-.5*np.einsum('i,oij,j->o',np.ones(3),hi,np.ones(3))
            candidate=direction*unit[0];candidate/=max(1.,np.max(abs(candidate)))
            pred=-gi@candidate-.5*np.einsum('i,oij,j->o',candidate,hi,candidate)
            out.append(dict(panel=context['panel'],role=context['role'],template=context['template'],row=i,modal_gradient_condition=float(s[0]/s[-1]),minimum_l2_leakage_per_unit_number=float(lower),impossible_all_three_below_10_percent_locally=bool(lower>np.sqrt(3)*.1),amplitudes=candidate.tolist(),unit_prediction=unit.tolist(),candidate_prediction=pred.tolist(),predicted_signed_number_retention=float(pred[0]/unit[0]) if abs(unit[0])>1e-12 else None))
    assert max(residual)<1e-8
    result=dict(max_constraint_residual=max(residual),contexts=len(out),local_impossibility_count=sum(c['impossible_all_three_below_10_percent_locally'] for c in out),minimum_bound=min(c['minimum_l2_leakage_per_unit_number'] for c in out),median_bound=float(np.median([c['minimum_l2_leakage_per_unit_number'] for c in out])),maximum_bound=max(c['minimum_l2_leakage_per_unit_number'] for c in out),candidates=out,scope='Infinitesimal unbounded linear bound within three fixed source directions. Candidate is clipped by uniform scaling to max amplitude1 and evaluated with fullquadratic; not yet native-validated. Controls onlythree modal contrasts.')
    (P/'SOURCE_SELECTIVITY_BOUND_V1.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='candidates'})
if __name__=='__main__':main()
