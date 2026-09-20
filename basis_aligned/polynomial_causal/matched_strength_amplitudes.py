"""Choose B-direction strength from analytic quadratic number response only."""
import numpy as np

def choose(g,h,candidate):
    b=np.array([0.,0.,1.,1.,1.]);linear=-g[0]@b;quadratic=-.5*b@h[0]@b
    target=-g[0]@candidate-.5*candidate@h[0]@candidate
    scale=max(abs(linear),abs(quadratic),abs(target),1e-30)
    roots=np.roots([quadratic,linear,-target]) if abs(quadratic)>scale*1e-14 else np.array([target/linear]) if abs(linear)>scale*1e-14 else np.array([])
    feasible=[float(z.real) for z in roots if abs(z.imag)<1e-10 and -1e-12<=z.real<=1+1e-12]
    exact=bool(feasible)
    if exact:alpha=np.clip(min(feasible),0.,1.)
    else:
        points=[0.,1.]
        if quadratic and 0<=-linear/(2*quadratic)<=1:points.append(-linear/(2*quadratic))
        alpha=min(points,key=lambda t:abs(linear*t+quadratic*t*t-target))
    return alpha*b,dict(alpha=float(alpha),has_root=exact,predicted_target=float(target),predicted_match=float(linear*alpha+quadratic*alpha*alpha))

if __name__=='__main__':
    import json,hashlib
    from pathlib import Path
    P=Path(__file__).parent;source=P.parent/'bilinear_quotient/circuits/followups/five_source_modal_null_v1_result.json';old=json.loads(source.read_text());contexts=[];all_checks=[]
    for c in old['contexts']:
        choices=[choose(np.array(g),np.array(h),np.array(a)) for g,h,a in zip(c['gradient'],c['hessian'],c['candidate_amplitudes'])]
        contexts.append(dict(panel=c['panel'],role=c['role'],template=c['template'],amplitudes=[a.tolist() for a,_ in choices],checks=[v for _,v in choices]));all_checks += [v for _,v in choices]
    result=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),contexts=contexts,missing_roots=sum(not c['has_root'] for c in all_checks),max_predicted_match_absolute=max(abs(c['predicted_match']-c['predicted_target']) for c in all_checks),alpha_range=[min(c['alpha'] for c in all_checks),max(c['alpha'] for c in all_checks)],scope='Frozen derivative-only strength choice before new native evaluation; not native matched-strength evidence.')
    (P/'MATCHED_STRENGTH_AMPLITUDES_V1.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='contexts'})
