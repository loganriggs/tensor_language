"""Freeze graph boundary and test a later optimization checkpoint and spectra.

A exact core/CP/execution<=1e-8; B group53cos>=.99/drift<=.0002;
C rank1output>=.9 at both checkpoints; D dominant outputcos>=.95.
Temporal optimization check, not text/seed holdout or semantics.
"""
import hashlib,io,json,time
from pathlib import Path
import torch
from cancellation_group_v1_audit import load
from native_support_exchange_v1_audit import P
from chunked_bilinear_coefficient_v1 import dense
from structured_branch_amplitudes_v1 import inner


def main():
    torch.set_grad_enabled(False);torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    out=P/'CANCELLATION_BOUNDARY_FUTURE_V1_AUDIT.json';assert not out.exists();started=time.perf_counter()
    prior=json.loads((P/'CANCELLATION_BOUNDARY_V1_AUDIT.json').read_text())
    group=next(r['products'] for r in prior['rows'] if r['method']=='graph' and r['edge_count']==128)
    payload=Path('/dev/shm/bilin18_projected_sparse_dictionary_fit_v1_s0.pt').read_bytes()
    sha=hashlib.sha256(payload).hexdigest();source=Path('/dev/shm/bilin18_cancellation_boundary_future_v1_source.pt')
    with source.open('xb') as f:f.write(payload)
    saved=torch.load(io.BytesIO(payload),weights_only=True,map_location='cpu');del payload
    later_source=dict(path=str(source),sha256=sha,iteration=saved['history'][-1]['iteration'],loss=saved['loss'])
    assert later_source['iteration']>prior['source']['iteration']
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram'].double()
    wh=torch.linalg.cholesky(metric).T;earlier=load(prior['source'],wh);later=load(later_source,wh)
    total=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())['native_total']
    groups={};directions={};rows=[];worst=0.
    for label,function in [('earlier',earlier),('later',later)]:
        for size,ids in [('16',prior['base']['products']),('53',group)]:
            a,b,w=function[0][ids],function[1][ids],function[2][:,ids];cp=(a,b,w)
            ib=torch.linalg.qr(torch.cat((a,b)).T,mode='reduced').Q
            ob=torch.linalg.qr(w,mode='reduced').Q;core=dense(a@ib,b@ib,ob.T@w)
            u,s,_=torch.linalg.svd(core.reshape(len(ids),-1),full_matrices=False)
            energy=inner(cp,cp);replay=float(abs(core.square().sum()-energy)/energy.clamp_min(1e-30))
            x=torch.randn(8,1152,generator=torch.Generator().manual_seed(341));z=x@ib
            direct=((x@a.T)*(x@b.T))@w.T;folded=torch.einsum('ni,oij,nj->no',z,core,z)@ob.T
            execution=float((direct-folded).norm()/direct.norm().clamp_min(1e-30));worst=max(worst,replay,execution)
            rows.append(dict(checkpoint=label,group=size,count=len(ids),energy=float(energy/total),
                output_captures={str(k):float(s[:k].square().sum()/s.square().sum()) for k in (1,2,4,8,16)},
                core_cp_replay=replay,executor_replay=execution))
            groups[(label,size)]=cp;directions[(label,size)]=ob@u[:,0]
    drift=[]
    for size in ('16','53'):
        f,g=groups[('earlier',size)],groups[('later',size)];ef,eg=inner(f,f),inner(g,g);cross=inner(f,g)
        drift.append(dict(group=size,cosine=float(cross/(ef*eg).sqrt()),
            squared_function_change=float((ef+eg-2*cross)/total),
            dominant_output_cosine=float(abs(directions[('earlier',size)]@directions[('later',size)]))))
    large=next(r for r in drift if r['group']=='53')
    result=dict(predictions=dict(pred_a_instrument=worst<=1e-8,
        pred_b_future=large['cosine']>=.99 and large['squared_function_change']<=.0002,
        pred_c_simple_output=all(r['output_captures']['1']>=.9 for r in rows if r['group']=='53'),
        pred_d_output_stability=large['dominant_output_cosine']>=.95),rows=rows,drift=drift,
        source=prior['source'],later_source=later_source,products=group,seconds=time.perf_counter()-started,
        scope='Frozen boundary applied to a later optimization snapshot. No independent-start, '
              'data/OOD, behavioral or semantic circuit claim; small CP core not automatically cheaper.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='products'},indent=2))


if __name__=='__main__':main()
