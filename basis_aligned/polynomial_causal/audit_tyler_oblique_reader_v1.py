"""CPU oblique dictionary recovery, radial/affine invariance, folding check."""
import hashlib
import json
from pathlib import Path
import torch
from scipy.optimize import linear_sum_assignment
from orthogonal_reader_msp_v1 import fit,polar
from tyler_reader_shape_v1 import tyler_shape,roots,reader_maps,normalize_columns,trace_normalize


def matched(estimate,truth):
    e=normalize_columns(estimate);t=normalize_columns(truth)
    similarities=(e.T@t).abs()
    i,j=linear_sum_assignment(-similarities.numpy())
    return float(similarities[i,j].mean())


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    torch.manual_seed(728)
    d,n=24,4096
    truth=polar(torch.randn(d,d))@torch.diag(torch.logspace(0,torch.log10(torch.tensor(50.)).item(),d))@polar(torch.randn(d,d))
    source=torch.randn(d,n)*(torch.rand(d,n)<.15)
    nonzero=source.norm(dim=0)>0
    source=source[:,nonzero]
    raw=truth@source
    y=normalize_columns(raw)
    shape,shape_fit=tyler_shape(y)
    scales=torch.exp(3*torch.randn(y.shape[1]))*torch.where(torch.rand(y.shape[1])>.5,1.,-1.)
    scaled,scaled_fit=tyler_shape(y*scales)
    transform=polar(torch.randn(d,d))@torch.diag(torch.linspace(.5,2.,d))@polar(torch.randn(d,d))
    transformed,transformed_fit=tyler_shape(transform@y)
    expected=trace_normalize(transform@shape@transform.T)
    invariant_errors=dict(radial=float((shape-scaled).norm()/shape.norm()),
        affine_shape=float((transformed-expected).norm()/expected.norm()))
    naive=trace_normalize(y@y.T/y.shape[1])
    reports=[]
    for name,scatter in [('tyler',shape),('ordinary_covariance',naive)]:
        _,inverse=roots(scatter)
        whitened=normalize_columns(inverse@y)
        for seed in (0,937,1902):
            rotation,optimization=fit(whitened,seed,max_steps=20000,max_seconds=30)
            maps=reader_maps(rotation,scatter)
            reconstructed=maps['synthesis']@(maps['weight_coder']@raw)
            folding_error=float((reconstructed-raw).norm()/raw.norm())
            # Independently verify that encoded weights read the same input.
            x=torch.randn(13,d)
            codes=(maps['weight_coder']@raw).T
            read_error=float(((x@maps['input_features'].T)@codes.T-x@raw).norm()/(x@raw).norm())
            report=dict(method=name,seed=seed,optimization=optimization,
                dictionary_cosine=matched(maps['synthesis'],truth),
                full_reader_replay=folding_error,input_read_replay=read_error)
            reports.append(report)
            print(json.dumps({k:v for k,v in report.items() if k!='optimization'}),flush=True)
    ours=[r for r in reports if r['method']=='tyler']
    base=[r for r in reports if r['method']=='ordinary_covariance']
    gain=sum(r['dictionary_cosine'] for r in ours)/3-sum(r['dictionary_cosine'] for r in base)/3
    result=dict(predictions={
        'pred_a_invariance_and_folding':max(invariant_errors.values())<=1e-8 and all(max(r['full_reader_replay'],r['input_read_replay'])<=1e-10 for r in reports),
        'pred_b_converged_recovery':all(r['converged'] for r in (shape_fit,scaled_fit,transformed_fit)) and all(r['optimization']['converged'] and r['dictionary_cosine']>=.99 for r in ours),
        'pred_c_covariance_gain':gain>=.005},
        invariant_errors=invariant_errors,shape_fit=shape_fit,starts=reports,
        mean_recovery_gain=gain,signals=int(nonzero.sum()),zero_signals_removed=int((~nonzero).sum()),
        planted_dimension=d,planted_dictionary_condition=50.,
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('tyler_reader_shape_v1.py'))},
        gpu_access=False,corpus_access=False,
        scope='Planted sparse oblique reader model; no native recovery guarantee. Tyler shape may fail to exist on concentrated supports; no ridge was added.')
    with Path(__file__).with_name('TYLER_OBLIQUE_READER_V1_CONTROL.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(predictions=result['predictions'],invariant_errors=invariant_errors,mean_recovery_gain=gain)))


if __name__=='__main__':main()
