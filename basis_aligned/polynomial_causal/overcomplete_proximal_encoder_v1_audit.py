"""Reuse existing accelerated proximal solver for fixed overcomplete readers."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from sklearn.decomposition import sparse_encode
from quadratic_token_dictionary_v1 import conditional
from two_support_oracle_v1 import encode

def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    data=np.load('/dev/shm/bilin18_overcomplete_reader_library_v1_sparse_s937.npz')
    x=data['test'];rows=[]
    for name,b in [('truth',data['truth']),('recovered937',data['dictionary'])]:
        xt,bt=torch.from_numpy(x),torch.from_numpy(b)
        start=time.perf_counter()
        code,report=conditional(torch.zeros(len(x),len(b)),bt@bt.T,xt@bt.T,.05,'codes',5000,1e-9)
        seconds=time.perf_counter()-start
        reference=sparse_encode(x,b,algorithm='lasso_lars',alpha=.05,n_jobs=1)
        z=code.numpy()
        objective=lambda c:float(.5*np.sum((c@b-x)**2)+.05*np.abs(c).sum())
        err=abs(objective(z)-objective(reference))/max(objective(reference),1)
        ids=np.argsort(np.abs(z),axis=1)[:,-2:];i,j=ids.T
        gram=b@b.T;rhs=x@b.T;row=np.arange(len(x))
        aa,bb,ab=gram[i,i],gram[j,j],gram[i,j];det=aa*bb-ab*ab
        assert det.min()>1e-12
        ci=(rhs[row,i]*bb-rhs[row,j]*ab)/det
        cj=(rhs[row,j]*aa-rhs[row,i]*ab)/det
        reconstruction=ci[:,None]*b[i]+cj[:,None]*b[j]
        capture=1-float(np.sum((reconstruction-x)**2)/np.sum(x*x))
        _,oracle=encode(b,x)
        rows.append(dict(name=name,proximal_report=report,seconds=seconds,reference_objective_error=err,
            lasso_top2_ls_capture=capture,oracle_capture=oracle['capture'],oracle_gap=oracle['capture']-capture,
            mean_lasso_active=float((np.abs(z)>1e-8).sum(1).mean())))
    predictions=dict(pred_a_reference=all(r['reference_objective_error']<=1e-9 for r in rows),
        pred_b_converged=all(r['proximal_report']['converged'] for r in rows),
        pred_c_capture=all(r['lasso_top2_ls_capture']>=.995 for r in rows),
        pred_d_oracle_gap=all(r['oracle_gap']<=.005 for r in rows))
    result=dict(predictions=predictions,arms=rows,scope='Fixed synthetic dictionaries; two-term execution after Lasso support proposal and exact LS; no native result')
    Path(__file__).with_name('OVERCOMPLETE_PROXIMAL_ENCODER_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
