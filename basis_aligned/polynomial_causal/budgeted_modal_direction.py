"""Per-input bounded source edit with nonzero modal-gradient budget."""
import numpy as np
from scipy.optimize import linprog

def choose(g,kappa,reference=None):
    n=-g[0];modal=-g[1:];reference=np.array([0.,0.,1.,1.,1.]) if reference is None else np.asarray(reference);assert reference.shape==n.shape;sign=np.sign(n@reference) or 1.;target=sign*n;scale=max(np.linalg.norm(target),1e-30)
    A=np.concatenate([modal-kappa*target,-modal-kappa*target])/scale;c=-target/scale
    r=linprog(c,A_ub=A,b_ub=np.zeros(6),bounds=[(-1,1)]*len(n),method='highs');assert r.success,r.message
    dual=-np.sum(r.lower.marginals)+np.sum(r.upper.marginals);gap=abs(r.fun-dual);station=c-A.T@r.ineqlin.marginals-r.lower.marginals-r.upper.marginals
    assert gap<1e-9 and max(abs(station))<1e-9 and max(A@r.x)<1e-9
    return r.x,dict(linear_retention=float(target@r.x/max(abs(n@reference),1e-30)),gap=float(gap),stationarity=float(max(abs(station))))

if __name__=='__main__':
    import json,hashlib,argparse
    parser=argparse.ArgumentParser();parser.add_argument('--kappa',type=float,nargs='+',default=[.05,.1]);parser.add_argument('--output',default='BUDGETED_MODAL_DIRECTIONS_V1.json');args=parser.parse_args()
    from pathlib import Path
    P=Path(__file__).parent;A=P.parent/'bilinear_quotient/circuits/followups';contexts=[];summary={}
    for dataset,file in [('opened','five_source_modal_null_v1_result.json'),('ood_opened','source_ood_v1_result.json')]:
        source=A/file;r=json.loads(source.read_text());all_results={}
        for kappa in args.kappa:
            results=[]
            for c in r['contexts']:
                g=np.array(c['gradient']);h=np.array(c['hessian']);selected=[choose(gi,kappa) for gi in g];a=np.array([v for v,_ in selected]);pred=-np.einsum('boi,bi->bo',g,a)-.5*np.einsum('bi,boij,bj->bo',a,h,a)
                B=np.broadcast_to([0,0,1,1,1],a.shape);base=-np.einsum('boi,bi->bo',g,B)-.5*np.einsum('bi,boij,bj->bo',B,h,B)
                contexts.append(dict(dataset=dataset,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),panel=c['panel'],role=c['role'],template=c['template'],kappa=kappa,amplitudes=a.tolist(),checks=[v for _,v in selected]))
                rows=json.loads((P/(f"SEMANTIC_PORT_FRESH_{c['panel'].upper()}_ROWS.json" if dataset=='opened' else f"SOURCE_OOD_V1_{c['panel'].upper()}_ROWS.json")).read_text());rows=[r for r in rows if r['template']==c['template']]
                for family in dict.fromkeys(r['family'] for r in rows):
                    ids=[i for i,r in enumerate(rows) if r['family']==family];y=pred[ids];yb=base[ids];ret=float(y[:,0]@yb[:,0]/max(yb[:,0]@yb[:,0],1e-30));modal=float(max(np.linalg.norm(y[:,1:],axis=0))/max(np.linalg.norm(y[:,0]),1e-30));results.append(dict(panel=c['panel'],role=c['role'],family=family,predicted_retention=ret,predicted_modal_ratio=modal))
            all_results[str(kappa)]=dict(cells=results,predicted_joint_passes=sum(c['predicted_retention']>=.8 and c['predicted_modal_ratio']<=.1 for c in results),median_retention=float(np.median([c['predicted_retention'] for c in results])),max_modal_ratio=max(c['predicted_modal_ratio'] for c in results))
        summary[dataset]=all_results
    out=dict(contexts=contexts,summary=summary,scope='Derivative-only direction selection; finite quadratic estimates, not new native outcomes. Existing finite gates unchanged. Both datasets now opened; no fresh claim.')
    (P/args.output).write_text(json.dumps(out,indent=2)+'\n');print({d:{k:{key:v for key,v in value.items() if key!='cells'} for k,value in values.items()} for d,values in summary.items()})
