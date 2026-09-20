"""Oracle finite-response lower bound for a fixed six-atom dictionary."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
P=Path(__file__).parent;A=P.parent/'bilinear_quotient/circuits/followups'
def main():
    prev=json.loads((P/'SHARED_FIVE_SOURCE_SPARSE_RESIDUAL_V1_RESULT.json').read_text());F=np.array(prev['plane']);atoms=[np.outer(F[:,0],F[:,0]),(np.outer(F[:,0],F[:,1])+np.outer(F[:,1],F[:,0]))/2**.5,np.outer(F[:,1],F[:,1])]
    for i,j in prev['supports']['shared2_plus3']:
        z=np.zeros((5,5));z[i,j]=z[j,i]=1/(1 if i==j else 2**.5);atoms.append(z)
    atoms=np.array(atoms)
    native=json.loads((A/'five_source_full_span_v1_result.json').read_text());old=json.loads((A/'native_source_curvature_sum_v1_result.json').read_text())
    look={(r['panel'],r['role'],r['family'],tuple(r['source_set'])):r for r in native['records'] if r['mode']=='full'}
    oldlook={(r['panel'],r['role'],r['family'],r['arm']):r for r in old['records'] if r['mode']=='full'}
    cases=torch.load(A/'native_source_curvature_sum_v1.pt',map_location='cpu',weights_only=True)['cases'];bounds=[];normal=[];minimax=[]
    for c in cases:
        G=c['gradient'].numpy()[:,0]
        arms=[(np.broadcast_to(np.isin(np.arange(5),s).astype(float),G.shape),s,None) for s in native['plan']['source_sets']]
        arms += [(a.numpy(),None,name) for name,a in c['amplitudes'].items()]
        for family in dict.fromkeys(c['families']):
            ids=[i for i,f in enumerate(c['families']) if f==family];ys=[];designs=[];den=[]
            for a,s,name in arms:
                r=look[(c['panel'],c['role'],family,tuple(s))] if s is not None else oldlook[(c['panel'],c['role'],family,name)]
                target=np.array(r['target'])[:,0];ys.append(target+np.einsum('bp,bp->b',G[ids],a[ids]));den.append(max(np.linalg.norm(target),1e-30));designs.append(-.5*np.einsum('bi,kij,bj->bk',a[ids],atoms,a[ids]))
            Y=np.array(ys)/np.array(den)[:,None];D=np.array(designs)/np.array(den)[:,None,None];res=[]
            for b in range(len(ids)):
                coeff=np.linalg.lstsq(D[:,b],Y[:,b],rcond=None)[0];e=D[:,b]@coeff-Y[:,b];res.append(e);normal.append(float(abs(D[:,b].T@e).max()))
            U=np.stack([np.linalg.svd(D[:,b],full_matrices=False)[0][:,:6] for b in range(len(ids))],axis=1)
            start=np.einsum('abk,ab->bk',U,Y)
            def error(v):return np.einsum('abk,bk->ab',U,v[:-1].reshape(len(ids),6))-Y
            def constraint(v):return v[-1]-np.linalg.norm(error(v),axis=1)
            def jac(v):
                e=error(v);norm=np.maximum(np.linalg.norm(e,axis=1),1e-30)
                return np.column_stack(((-U*(e/norm[:,None])[...,None]).reshape(len(arms),-1),np.ones(len(arms))))
            x=np.r_[start.ravel(),np.linalg.norm(np.array(res).T,axis=1).max()+1e-8]
            objective_gradient=np.r_[np.zeros(len(x)-1),1.]
            fit=minimize(lambda v:v[-1],x,jac=lambda v:objective_gradient,constraints=[dict(type='ineq',fun=constraint,jac=jac)],method='SLSQP',options=dict(maxiter=500,ftol=1e-11))
            e=error(fit.x);norm=np.linalg.norm(e,axis=1);primal=float(norm.max())
            # Multipliers are not needed: fit nonnegative active-row weights for stationarity.
            active=norm>=primal-1e-6
            directions=e/np.maximum(norm[:,None],1e-30)
            gradients=(U*directions[...,None]).reshape(len(arms),-1)
            from scipy.optimize import nnls
            weights=np.zeros(len(arms));weights[active]=nnls(np.vstack([gradients[active].T,np.ones(active.sum())]),np.r_[np.zeros(gradients.shape[1]),1.])[0]
            dual=weights[:,None]*directions
            dual-=np.einsum('abk,bk->ab',U,np.einsum('abk,ab->bk',U,dual))
            dual/=max(1.,np.linalg.norm(dual,axis=1).sum())
            lower_dual=float(-np.sum(dual*Y));feas=float(abs(np.einsum('abk,ab->bk',U,dual)).max())
            minimax.append(dict(panel=c['panel'],role=c['role'],family=family,optimizer_success=bool(fit.success),primal=primal,dual=lower_dual,gap=primal-lower_dual,dual_stationarity=feas,feasible_under_ten_percent=primal<=.1,excluded_by_dual=lower_dual>.1))
            lower=float(np.linalg.norm(res)/len(arms)**.5)
            bounds.append(dict(panel=c['panel'],role=c['role'],family=family,worst_arm_error_lower_bound=lower,excludes_ten_percent=lower>.1))
    out=dict(minimax=minimax,minimax_max_primal=max(r["primal"] for r in minimax),minimax_max_gap=max(r["gap"] for r in minimax),minimax_feasible_cells=sum(r["feasible_under_ten_percent"] for r in minimax),minimax_excluded_cells=sum(r["excluded_by_dual"] for r in minimax),bounds=bounds,max_lower_bound=max(r['worst_arm_error_lower_bound'] for r in bounds),excluded_cells=sum(r['excludes_ten_percent'] for r in bounds),least_squares_normal_residual=max(normal),scope='Number-only oracle minimax fit, arbitrary six coefficients/context, fixed analytic linear term, all18 finite arms. Floating-point primal/dual witnesses checked independently. All native outcomes used; not a predictor, modal preservation, OOD or circuit evidence. No claim against other dictionaries or linear terms.')
    (P/'SHARED_DICTIONARY_MINIMAX_V1_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['bounds','minimax']},indent=2))
if __name__=='__main__':main()
