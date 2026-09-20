"""Bounded nonlinear source edit selection from analytic quadratic responses."""
import numpy as np
from scipy.optimize import minimize
from budgeted_modal_direction import choose as linear_choose

def choose(g,h,exact_null,reference=None):
    h=(h+h.transpose(0,2,1))/2;B=np.array([0.,0.,1.,1.,1.]) if reference is None else np.asarray(reference);assert B.shape==(g.shape[-1],);sign=np.sign((-g[0])@B) or 1.;scale=max(np.linalg.norm(g[0]),1e-30)
    def effect(a):return -g@a-.5*np.einsum('i,oij,j->o',a,h,a)
    def derivative(a):return -g-np.einsum('oij,j->oi',h,a)
    def cons(a):
        e=effect(a);return np.r_[.08*sign*e[0]-e[1:],.08*sign*e[0]+e[1:]]/scale
    def jac(a):
        d=derivative(a);return np.concatenate([.08*sign*d[0]-d[1:],.08*sign*d[0]+d[1:]])/scale
    starts=[np.zeros(len(B)),B,exact_null,linear_choose(g,.05,reference=B)[0],linear_choose(g,.1,reference=B)[0]]
    candidates=[np.zeros(len(B))];records=[]
    for a in starts:
        if cons(a).min()>=-1e-8:candidates.append(a)
        r=minimize(lambda a:-sign*effect(a)[0]/scale,a,jac=lambda a:-sign*derivative(a)[0]/scale,bounds=[(-1,1)]*len(B),constraints=[dict(type='ineq',fun=cons,jac=jac)],method='SLSQP',options=dict(maxiter=300,ftol=1e-11))
        feasible=cons(r.x).min()>=-1e-8 and abs(r.x).max()<=1+1e-10
        records.append(dict(success=bool(r.success),feasible=bool(feasible),iterations=int(r.nit),message=str(r.message)))
        if feasible:candidates.append(r.x)
    best=max(candidates,key=lambda a:sign*effect(a)[0]);assert cons(best).min()>=-1e-8
    return best,dict(min_constraint=float(cons(best).min()),predicted_effect=effect(best).tolist(),solves=records)

if __name__=='__main__':
    import json,hashlib
    from pathlib import Path
    P=Path(__file__).parent;A=P.parent/'bilinear_quotient/circuits/followups';contexts=[];summary={}
    for dataset,file in [('opened','five_source_modal_null_v1_result.json'),('ood_opened','source_ood_v1_result.json')]:
        source=A/file;r=json.loads(source.read_text());cells=[]
        for c in r['contexts']:
            g=np.array(c['gradient']);h=np.array(c['hessian']);choices=[choose(gi,hi,np.array(a)) for gi,hi,a in zip(g,h,c['candidate_amplitudes'])];a=np.array([x for x,_ in choices]);pred=-np.einsum('boi,bi->bo',g,a)-.5*np.einsum('bi,boij,bj->bo',a,h,a)
            B=np.broadcast_to([0,0,1,1,1],a.shape);base=-np.einsum('boi,bi->bo',g,B)-.5*np.einsum('bi,boij,bj->bo',B,h,B)
            contexts.append(dict(dataset=dataset,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),panel=c['panel'],role=c['role'],template=c['template'],amplitudes=a.tolist(),checks=[check for _,check in choices]))
            pattern='SEMANTIC_PORT_FRESH' if dataset=='opened' else 'SOURCE_OOD_V1';rows=json.loads((P/f"{pattern}_{c['panel'].upper()}_ROWS.json").read_text());rows=[x for x in rows if x['template']==c['template']]
            for family in dict.fromkeys(x['family'] for x in rows):
                ids=[i for i,x in enumerate(rows) if x['family']==family];y=pred[ids];yb=base[ids];ret=float(y[:,0]@yb[:,0]/max(yb[:,0]@yb[:,0],1e-30));modal=float(max(np.linalg.norm(y[:,1:],axis=0))/max(np.linalg.norm(y[:,0]),1e-30));cells.append(dict(panel=c['panel'],role=c['role'],family=family,retention=ret,modal_ratio=modal))
        summary[dataset]=dict(cells=cells,predicted_joint_passes=sum(c['retention']>=.8 and c['modal_ratio']<=.1 for c in cells),median_retention=float(np.median([c['retention'] for c in cells])),max_modal_ratio=max(c['modal_ratio'] for c in cells))
    out=dict(contexts=contexts,summary=summary,scope='Nonconvex local optimization of analytic response only. Feasible multistart candidates, not global optima. Both datasets opened; native validation pending.')
    (P/'QUADRATIC_BUDGETED_DIRECTIONS_V1.json').write_text(json.dumps(out,separators=(',',':'))+'\n');print({k:{key:v for key,v in value.items() if key!='cells'} for k,value in summary.items()})
