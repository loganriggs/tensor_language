"""Exact two-atom sparse least squares for small dictionary controls only."""
import itertools
import numpy as np

def encode(dictionary,x):
    pairs=np.array(list(itertools.combinations(range(len(dictionary)),2)))
    gram=dictionary@dictionary.T;rhs=x@dictionary.T
    i,j=pairs.T
    aa,bb,ab=gram[i,i],gram[j,j],gram[i,j]
    determinant=aa*bb-ab**2
    assert determinant.min()>1e-12,'Dependent pairs need explicit separate handling'
    ci=(rhs[:,i]*bb-rhs[:,j]*ab)/determinant
    cj=(rhs[:,j]*aa-rhs[:,i]*ab)/determinant
    explained=rhs[:,i]*ci+rhs[:,j]*cj
    winner=explained.argmax(axis=1);rows=np.arange(len(x))
    codes=np.zeros((len(x),len(dictionary)))
    codes[rows,i[winner]]=ci[rows,winner]
    codes[rows,j[winner]]=cj[rows,winner]
    residual=x-codes@dictionary
    score=1-float(np.sum(residual**2)/np.sum(x**2))
    direct_error=float(abs(np.sum(residual**2)-(np.sum(x**2)-explained[rows,winner].sum()))/np.sum(x**2))
    normal=residual@dictionary.T
    normal_error=float(max(abs(normal[rows,i[winner]]).max(),abs(normal[rows,j[winner]]).max()))
    return codes,dict(capture=score,pair_count=len(pairs),direct_error=direct_error,normal_error=normal_error)

if __name__=='__main__':
    import json
    from pathlib import Path
    from sklearn.decomposition import sparse_encode
    saved=np.load('/dev/shm/bilin18_overcomplete_reader_library_v1_sparse_s937.npz')
    x=saved['test'];rows=[]
    for name,basis in [('truth',saved['truth']),('recovered937',saved['dictionary'])]:
        _,result=encode(basis,x)
        omp=sparse_encode(x,basis,algorithm='omp',n_nonzero_coefs=2,n_jobs=1)
        omp_capture=1-float(np.sum((x-omp@basis)**2)/np.sum(x**2))
        rows.append(dict(name=name,omp_capture=omp_capture,oracle_gain=result['capture']-omp_capture,**result))
    predictions=dict(pred_a_replay=all(max(r['direct_error'],r['normal_error'])<=1e-10 for r in rows),
        pred_b_truth=rows[0]['capture']>=1-1e-10,
        pred_c_greedy_miss=rows[0]['oracle_gain']>=.005,
        pred_d_recovered=rows[1]['capture']>=.999 and rows[1]['oracle_gain']>=.01)
    result=dict(predictions=predictions,arms=rows,scope='Fixed two-sparse synthetic support oracle, no native basis fitting')
    Path(__file__).with_name('TWO_SUPPORT_ORACLE_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
