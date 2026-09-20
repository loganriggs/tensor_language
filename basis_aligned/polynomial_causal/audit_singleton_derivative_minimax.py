"""Per-input robust fit: singleton norm constraints reduce exactly to an LP."""
import numpy as np
from scipy.optimize import linprog
from audit_derivative_only_minimax import main

def singleton_fit(G,H,amplitudes,atoms,diagnostics=None,shared_budgets=None):
    coefficients=[];all_errors=[]
    for b in range(len(G)):
        a=amplitudes[:,b]
        linear=-np.einsum('op,ap->ao',G[b],a)
        quadratic=-.5*np.einsum('ap,opq,aq->ao',a,H[b],a)
        budgets=np.maximum(abs((linear+quadratic)[:,0]),1e-30) if shared_budgets is None else shared_budgets
        D=-.5*np.einsum('ap,kpq,aq->ak',a,atoms,a)/budgets[:,None]
        U,s,Vt=np.linalg.svd(D,full_matrices=False);assert s[-1]>s[0]*1e-12
        constraints=np.concatenate([np.column_stack([U,-np.ones(len(a))]),np.column_stack([-U,-np.ones(len(a))])])
        cs=[];errors=[]
        for o in range(G.shape[1]):
            y=quadratic[:,o]/budgets;rhs=np.r_[y,-y];objective=np.r_[np.zeros(len(atoms)),1.]
            result=linprog(objective,A_ub=constraints,b_ub=rhs,bounds=[(None,None)]*len(atoms)+[(0,None)],method='highs')
            assert result.success,result.message
            residual=U@result.x[:-1]-y;primal=float(abs(residual).max());dual=float(rhs@result.ineqlin.marginals)
            stationarity=float(abs(constraints[:,:-1].T@result.ineqlin.marginals).max())
            assert abs(primal-dual)<1e-7 and stationarity<1e-10
            cs.append(Vt.T@(result.x[:-1]/s));errors.append(primal)
            if diagnostics is not None:diagnostics.append(dict(output=o,optimizer_success=True,message='Singleton Chebyshev linear program',primal=primal,dual=dual,gap=primal-dual,dual_stationarity=stationarity))
        coefficients.append(cs);all_errors.append(errors)
    return np.array(coefficients),np.max(all_errors,axis=0).tolist()

def singleton_group_budget_fit(G,H,amplitudes,atoms,diagnostics=None):
    linear=-np.einsum('bop,abp->abo',G,amplitudes)
    quadratic=-.5*np.einsum('abp,bopq,abq->abo',amplitudes,H,amplitudes)
    budgets=np.maximum(np.linalg.norm((linear+quadratic)[...,0],axis=1),1e-30)
    return singleton_fit(G,H,amplitudes,atoms,diagnostics,budgets)

if __name__=='__main__':
    main(singleton_fit,'SINGLETON_DERIVATIVE_MINIMAX_V1_RESULT.json','Per-input LP fitter ignores group peers. Original groups used only for evaluation. Analytic scalar number budgets differ from previous group budgets.')
