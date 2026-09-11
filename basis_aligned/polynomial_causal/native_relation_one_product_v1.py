"""Frozen one-product change prediction; not data-fitting or rank selection.
A product/trace replay<=1e-9 and bound native cache/factors.
B scalar change relative RMS<=.25 and sign agreement>=.9 in every task/direction/readout.
C dual-lifted physical change relative RMS<=.25 in each task/direction.
"""
import hashlib
import json
import sys
from pathlib import Path
import torch


@torch.no_grad()
def main(label):
    assert label in ('original','holdout')
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    p=Path(__file__).parent;output=p/f'NATIVE_RELATION_ONE_PRODUCT_V1_{label.upper()}.json'
    assert not output.exists()
    stem='FROZEN_BRANCH_MORPHOLOGY_V1' if label=='original' else 'NATIVE_RELATION_HOLDOUT_V1'
    paths=[p/'NATIVE_TOKEN_RELATION_FOLD_V1.pt',p/(stem+'_ENDPOINTS.pt'),p/(stem+'_ROWS.json'),p/(stem+'_RESULT.json')]
    saved=torch.load(paths[0],weights_only=True,map_location='cpu')
    fold=json.loads((p/'NATIVE_TOKEN_RELATION_FOLD_V1.json').read_text())
    assert hashlib.sha256(paths[0].read_bytes()).hexdigest()==fold['artifact_sha256']
    receipt=json.loads(paths[3].read_text());assert receipt['pred_a']
    assert hashlib.sha256(paths[1].read_bytes()).hexdigest()==receipt['cache_sha256']
    cache=torch.load(paths[1],weights_only=True,map_location='cpu');rows=json.loads(paths[2].read_text())['rows']
    assert hashlib.sha256(paths[2].read_bytes()).hexdigest()==cache['rows_sha256']
    r=saved['physical_readouts'][:3];writers=torch.linalg.solve(r@r.T,r).T
    x=cache['ports']['input'].double();reference=cache['ports']['native_output'].double()@r.T
    predictions=[];errors=[]
    for j,c in enumerate(saved['compact'][:3]):
        a,b=c['leading_product_a'],c['leading_product_b'];trace=a@b
        q=(torch.outer(a,b)+torch.outer(b,a))/2
        term=(x@a)*(x@b)
        errors.append(float((term-torch.einsum('ni,ij,nj->n',x,q,x)).norm()/term.norm()))
        alpha=c['radial']-trace/1152
        errors.append(float(abs(q.trace()+1152*alpha-1152*c['radial'])))
        predictions.append(term+alpha*x.square().sum(1)+saved['folded_bias'][j])
    predicted=torch.stack(predictions,1);cells=[]
    for family in ('A1','A2'):
        for direction in ('base_to_suffix','suffix_to_base'):
            ids=torch.tensor([i for i,row in enumerate(rows) if row['family']==family and row['direction']==direction])
            actual=reference[2*ids+1]-reference[2*ids]
            approx=predicted[2*ids+1]-predicted[2*ids]
            metrics=[]
            for j,name in enumerate(saved['labels'][:3]):
                metrics.append(dict(relation=name,relative_rms=float((actual[:,j]-approx[:,j]).norm()/actual[:,j].norm()),
                    sign_agreement=float((actual[:,j].sign()==approx[:,j].sign()).double().mean()),
                    native_mean=float(actual[:,j].mean()),predicted_mean=float(approx[:,j].mean())))
            physical=float(((actual-approx)@writers.T).norm()/(actual@writers.T).norm())
            cells.append(dict(family=family,direction=direction,n=len(ids),scalar_metrics=metrics,physical_relative_rms=physical))
    a=max(errors)<=1e-9 and bool(torch.isfinite(predicted).all())
    result=dict(pred_a=a,pred_b=a and all(m['relative_rms']<=.25 and m['sign_agreement']>=.9 for c in cells for m in c['scalar_metrics']),
        pred_c=a and all(c['physical_relative_rms']<=.25 for c in cells),label=label,replay_errors=errors,cells=cells,
        price=dict(variable_products=3,dense_input_readers=6,physical_writers=3,shared_radius_required=True,native_background_retained=True),
        sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        scope='Saved optimal scalar coefficient products, no data-fit/scale/rank search. This predicts native readout changes and '
        'their lifted physical write, not yet final capped logits, ordinary replacement, selective removal or composition.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main(sys.argv[1])
