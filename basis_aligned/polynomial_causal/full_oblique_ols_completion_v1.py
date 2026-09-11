"""All-weight frozen oblique dictionary recoding, with full folded-tensor scoring."""
import hashlib,json,signal,time
from pathlib import Path
import torch
from native_reader_msp_generalization_v1 import P,CK
from lasso_ols_completion_v1 import encode
from structured_branch_amplitudes_v1 import inner
from sparse_reader_program_v1 import SparseReaderProgram

@torch.no_grad()
def main():
    signal.alarm(180);torch.set_num_threads(2);torch.set_default_dtype(torch.float64);started=time.perf_counter()
    assert all(json.loads((P/'LASSO_OLS_COMPLETION_V1_CONTROL.json').read_text())['predictions'].values())
    parent=json.loads((P/'OBLIQUE_READER_DICTIONARY_V1_ordinary_covariance_SEED_0.json').read_text())
    source=parent['cache'];assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
    saved=torch.load(source['path'],weights_only=True,map_location='cpu');basis=saved['analysis_basis']
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ('Left','Right','Down')]
    bias=sd['transformer.h.17.mlp.Down_bias'].double();readers=torch.cat((l,r));norm=readers.norm(dim=1)
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram']
    writers=torch.linalg.cholesky(metric).T@d;native=(l,r,writers)
    total=json.loads((P/'FULLU_TRACE_METRIC_V1_AUDIT.json').read_text())['full_coefficient_metric']['total_energy']
    tic=time.perf_counter();ids,values,report=encode(basis,readers,128,.05);encoding_seconds=time.perf_counter()-tic
    print(json.dumps(dict(phase='encoded',seconds=encoding_seconds,converged=report['converged'],
                         completed_rows=report['rows_requiring_completion'])),flush=True)
    cache=Path('/dev/shm/bilin18_full_oblique_ols_completion_v1.pt');assert not cache.exists()
    torch.save(dict(analysis_basis=basis,code_indices=ids.to(torch.int16),code_values=values,order=saved['order'],
                    source=source,encoding=report,retained_down_key='transformer.h.17.mlp.Down.weight',
                    retained_bias_key='transformer.h.17.mlp.Down_bias'),cache)
    program=SparseReaderProgram(basis,ids,values,d,bias)
    fitted=torch.sparse.mm(program.codes,basis);a,b=fitted.split(4608)
    errors=((fitted-readers)/norm[:,None]).square().sum(1)
    order=saved['order'];train=torch.cat((order[:3072],order[:3072]+4608));test=torch.cat((order[3072:],order[3072:]+4608))
    proposal=(a,b,writers);capture=1-float((inner(proposal,proposal)-2*inner(native,proposal)+total)/total)
    old_codes=torch.zeros(9216,1152).scatter_(1,saved['code_indices'].long(),saved['code_values'])
    old_a,old_b=(old_codes@basis).split(4608);old=(old_a,old_b,writers)
    old_capture=1-float((inner(old,old)-2*inner(native,old)+total)/total)
    parent_replay=abs(old_capture-parent['scores']['coefficient_capture'])
    x=torch.randn(8,1152,generator=torch.Generator().manual_seed(836))
    direct=((x@a.T)*(x@b.T))@d.T+bias
    execution=float((program(x)-direct).norm()/direct.norm())
    heldout=1-float(errors[test].mean());gain=capture-old_capture
    valid=report['converged'] and report['maximum_code_kkt']<=1e-5 and max(
        report['support_ls_normal_residual'],parent_replay,execution)<=1e-8 and bool(torch.isfinite(errors).all()) and math_finite(capture)
    result=dict(predictions=dict(pred_a_instrument=valid,pred_b_tensor_gain=gain>=.01,
        pred_c_historical_reader_gain=heldout-parent['scores']['test_normalized_reader_capture']>=.02),
        coefficient_capture_before=old_capture,coefficient_capture_after=capture,coefficient_gain=gain,
        training_reader_capture=1-float(errors[train].mean()),historical_test_reader_capture=heldout,
        parent_historical_test_capture=parent['scores']['test_normalized_reader_capture'],
        parent_capture_replay=parent_replay,executor_replay=execution,encoding=report,
        encoding_seconds=encoding_seconds,seconds=time.perf_counter()-started,
        cache=dict(path=str(cache),sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),bytes=cache.stat().st_size),
        price=program.price(),source=source,
        scope='Full weight-only recoding of one frozen dictionary; same128-term price and nativeDown. Historical split is not fresh encoder-selection validation; no text, semantic or circuit-property claim.')
    with (P/'FULL_OBLIQUE_OLS_COMPLETION_V1_RESULT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='encoding'}),flush=True)

def math_finite(value):return bool(torch.isfinite(torch.tensor(value)))

if __name__=='__main__':main()
