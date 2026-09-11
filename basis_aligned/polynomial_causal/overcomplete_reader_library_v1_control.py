"""Planted overcomplete reader dictionary using established sklearn solver.
Predictions registered on AGENT_BOARD before execution; no native/data fitting.
"""
import json
import time
import warnings
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import DictionaryLearning,sparse_encode
import sklearn

def soft(x,t):return np.sign(x)*np.maximum(np.abs(x)-t,0)

def stationarity(x,z,b,alpha):
    residual=z@b-x
    gradient=residual@b.T
    code_error=np.where(np.abs(z)>1e-10,np.abs(gradient+alpha*np.sign(z)),np.maximum(np.abs(gradient)-alpha,0))
    gd=z.T@residual/len(x)
    lip=np.linalg.norm(z.T@z/len(x),2)
    proposal=b-gd/max(lip,1e-30)
    proposal/=np.maximum(np.linalg.norm(proposal,axis=1,keepdims=True),1)
    return dict(code_kkt_max=float(code_error.max()),dictionary_proximal_gradient_max=float(np.abs((b-proposal)*lip).max()))

def main():
    rng=np.random.default_rng(801)
    truth=rng.normal(size=(24,12));truth/=np.linalg.norm(truth,axis=1,keepdims=True)
    z=np.zeros((1536,24))
    for i in range(len(z)):
        ids=rng.choice(24,2,replace=False)
        z[i,ids]=rng.choice([-1.,1.],2)*rng.uniform(.5,1.5,2)
    sparse=z@truth;sparse/=np.linalg.norm(sparse,axis=1,keepdims=True)
    dense=rng.normal(size=(1536,12));dense/=np.linalg.norm(dense,axis=1,keepdims=True)
    rows=[]
    for kind,seed in [('sparse',0),('sparse',937),('dense',0)]:
        x=(sparse if kind=='sparse' else dense)
        train,test=x[:512],x[512:]
        initial=train[np.random.default_rng(seed).choice(len(train),24,replace=False)].copy()
        model=DictionaryLearning(n_components=24,alpha=.05,max_iter=1000,tol=1e-10,
            fit_algorithm='lars',dict_init=initial,code_init=np.zeros((512,24)),
            transform_algorithm='omp',transform_n_nonzero_coefs=2,random_state=seed,n_jobs=1)
        started=time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            code=model.fit_transform(train)
            fitted=model.components_
            predictions=sparse_encode(test,fitted,algorithm='omp',n_nonzero_coefs=2,n_jobs=1)
            train_code=sparse_encode(train,fitted,algorithm='omp',n_nonzero_coefs=2,n_jobs=1)
        normalized=fitted/np.linalg.norm(fitted,axis=1,keepdims=True)
        similarity=np.abs(normalized@truth.T);i,j=linear_sum_assignment(-similarity)
        row=dict(kind=kind,seed=seed,seconds=time.perf_counter()-started,iterations=model.n_iter_,
            stopped_before_limit=model.n_iter_<1000,objective=float(model.error_[-1]),
            norm_max=float(np.linalg.norm(fitted,axis=1).max()),
            matched_atom_cosine=float(similarity[i,j].mean()),
            train_capture=1-float(np.sum((train_code@fitted-train)**2)/np.sum(train**2)),
            heldout_capture=1-float(np.sum((predictions@fitted-test)**2)/np.sum(test**2)),
            stationarity=stationarity(train,code,fitted,.05),warnings=sorted(set(str(w.message) for w in caught)))
        cache=Path(f'/dev/shm/bilin18_overcomplete_reader_library_v1_{kind}_s{seed}.npz')
        assert not cache.exists()
        np.savez(cache,dictionary=fitted,code=code,train=train,test=test,truth=truth,objective_history=model.error_)
        row['cache']=str(cache)
        Path(__file__).with_name(f'OVERCOMPLETE_READER_LIBRARY_V1_{kind}_SEED_{seed}.json').write_text(json.dumps(row,indent=2)+'\n')
        rows.append(row);print(json.dumps(row),flush=True)
    sparse_rows=[r for r in rows if r['kind']=='sparse']
    predictions=dict(pred_a_instrument=all(np.isfinite(r['objective']) and r['norm_max']<=1+1e-8 for r in rows),
        pred_b_recovery=all(r['matched_atom_cosine']>=.99 for r in sparse_rows),
        pred_c_heldout=all(r['heldout_capture']>=.99 for r in sparse_rows),
        pred_d_stationarity=all(r['stationarity']['code_kkt_max']<=1e-4 and r['stationarity']['dictionary_proximal_gradient_max']<=1e-5 for r in rows))
    result=dict(predictions=predictions,arms=rows,sklearn_version=sklearn.__version__,
        scope='Planted overcomplete sparse reader control, library alternating LARS/dictionary updates; not K-SVD or a native model result.')
    Path(__file__).with_name('OVERCOMPLETE_READER_LIBRARY_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(predictions),flush=True)

if __name__=='__main__':main()
