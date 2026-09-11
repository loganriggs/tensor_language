"""Exact conditional code polish after the final library dictionary update."""
import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import sparse_encode
from overcomplete_reader_library_v1_control import stationarity

def objective(x,z,b):return float(.5*np.sum((z@b-x)**2)+.05*np.abs(z).sum())

def main():
    p=Path(__file__).parent
    prior=json.loads((p/'OVERCOMPLETE_READER_LIBRARY_V1_CONTROL.json').read_text())
    rows=[]
    for arm in prior['arms']:
        source=np.load(arm['cache']);x,z,b=source['train'],source['code'],source['dictionary']
        new=sparse_encode(x,b,algorithm='lasso_lars',alpha=.05,n_jobs=1)
        before=objective(x,z,b);after=objective(x,new,b)
        path=Path(arm['cache'].replace('.npz','_recoded.npz'));assert not path.exists()
        np.savez(path,code=new)
        rows.append(dict(kind=arm['kind'],seed=arm['seed'],source=arm['cache'],cache=str(path),
            before_objective=before,after_objective=after,relative_change=(after-before)/before,
            prior_objective_replay=abs(before-arm['objective'])/before,
            before_stationarity=arm['stationarity'],after_stationarity=stationarity(x,new,b,.05)))
    predictions=dict(pred_a_code=all(r['after_stationarity']['code_kkt_max']<=1e-10 for r in rows),
        pred_b_dictionary=all(r['after_stationarity']['dictionary_proximal_gradient_max']<=1e-5 for r in rows),
        pred_c_objective=all(r['relative_change']<=1e-10 and r['prior_objective_replay']<=1e-10 for r in rows))
    result=dict(predictions=predictions,arms=rows,scope='Exact conditional code update; dictionaries, OMP scores and original misses unchanged')
    (p/'OVERCOMPLETE_READER_RECODE_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
