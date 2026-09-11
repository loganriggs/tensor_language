#!/usr/bin/env python3
"""pred_a: identities1e-8/replay1e-10; pred_b: beat16controls by.05; pred_c:6heads cosine>=.95."""
# BQGATE: 0forwards0seq; exact native quadratic producer-function geometry.
import os,sys,json,time,signal,hashlib
from pathlib import Path
RUNNER=Path(__file__).resolve();P=RUNNER.parents[3]/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from producer_function_overlap_v1 import product_gram,overlap
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def mean(values):return sum(values)/len(values)


def main():
    binding=json.loads((P/'MLP16_PRODUCER_OVERLAP_V1_BINDING.json').read_text())
    assert all(digest(f)==h for f,h in binding.items())
    assert json.loads((P/'PRODUCER_FUNCTION_OVERLAP_V1_CONTROL.json').read_text())['instrument_passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=0,corpus_access=False,heads=9,rotations=16)));return
    out=P/'MLP16_PRODUCER_OVERLAP_V1_RESULT.json';assert not out.exists()
    cache=Path('/dev/shm/bilin18_mlp16_producer_overlap_v1.pt');assert not cache.exists()
    signal.alarm(1200);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    start=time.perf_counter();sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    source=json.loads((P/'POSITION_SHARED_QK_SOURCE_V1_RESULT.json').read_text())['cache']
    component=json.loads((P/'JOINT_SHARED_READER_RANK16_V1_RESULT.json').read_text())['cache']
    assert digest(source['path'])==source['sha256'] and digest(component['path'])==component['sha256']
    a=torch.load(source['path'],weights_only=True,map_location='cpu')['frames'].transpose(1,2).double().cuda()
    saved=torch.load(component['path'],weights_only=True,map_location='cpu');selected=saved['programs'][saved['best']]
    frame=torch.cat((selected['reader'][None,:],selected['partner_readers']),0).double().cuda()
    o=sd['transformer.h.17.attn.c_proj.weight'].double().cuda()
    v=sd['transformer.h.17.attn.c_v.weight'].double().cuda();mix=float(sd['transformer.h.17.attn.lamb'])
    b=[];raw_ranks=[]
    for h in range(9):
        raw=frame@o[:,h*128:(h+1)*128]@v[h*128:(h+1)*128]*(1-mix)
        _,sing,right=torch.linalg.svd(raw,full_matrices=False)
        raw_ranks.append(int((sing>1e-10*sing[0]).sum()));b.append(right[:17])
    b=torch.stack(b)
    l,r,down=[sd[f'transformer.h.16.mlp.{key}.weight'].double().cuda() for key in ['Left','Right','Down']]
    bias=sd['transformer.h.16.mlp.Down_bias'].double().cuda()
    kg=product_gram(l,r);g=down@kg@down.T;g=(g+g.T)/2
    trace=down@(l*r).sum(1);gtr=g-trace[:,None]*trace[None,:]/1152
    eye=torch.eye(1152,dtype=g.dtype,device=g.device)
    raw=[];folded=[];tracefree=[];pairings=[];checks=[];errors=[]
    def inspect(row):
        checks.append(row['ranks']==[17,17])
        checks.append(all(0<=s<=1+1e-8 for s in row['principal_cosines']))
        return row
    for h in range(9):
        raw.append(inspect(overlap(a[h],b[h],eye)))
        folded.append(inspect(overlap(a[h],b[h],g)))
        tracefree.append(inspect(overlap(a[h],b[h],gtr)))
        for j in range(9):
            row=inspect(overlap(a[h],b[j],g));pairings.append(dict(qk_head=h,ov_head=j,**row))
    control=[];control_raw_errors=[]
    for seed in range(1361,1377):
        torch.manual_seed(seed);perm=torch.randperm(1152,device=g.device)
        signs=torch.randint(0,2,(1152,),device=g.device,dtype=torch.int64).to(g.dtype)*2-1
        ar=a[:,:,perm]*signs;br=b[:,:,perm]*signs
        rows=[inspect(overlap(ar[h],br[h],g)) for h in range(9)]
        rawerr=max(abs(overlap(ar[h],br[h],eye)['mean_squared_cosine']-raw[h]['mean_squared_cosine']) for h in range(9))
        control_raw_errors.append(rawerr)
        control.append(dict(seed=seed,mean_overlap=mean([x['mean_squared_cosine'] for x in rows]),heads=rows))
    # Independent native dense forms audit the implicit producer Gram.
    torch.manual_seed(1381)
    for _ in range(4):
        x,y=torch.randn(2,1152,device=g.device,dtype=g.dtype)
        def form(reader):
            matrix=l.T@((reader@down)[:,None]*r);return (matrix+matrix.T)/2
        qx,qy=form(x),form(y)
        expected=(qx*qy).sum();observed=x@g@y
        errors.append(float(abs(expected-observed)/(qx.norm()*qy.norm())))
    # Exact local source port including reentry, base state, bias and RMS.
    z,x0=torch.randn(2,32,1152,device=g.device,dtype=g.dtype)
    eps=torch.finfo(torch.float32).eps
    xn=z/(z.square().mean(1,keepdim=True)+eps).sqrt()
    products=(xn@l.T)*(xn@r.T)
    lambdas=sd['transformer.h.17.lambdas'].double().cuda()
    t=lambdas[0]*(z+products@down.T+bias)+lambdas[1]*x0
    rho=(t.square().mean(1,keepdim=True)+eps).sqrt()
    readers=torch.cat((a.flatten(0,1),b.flatten(0,1)),0)
    direct=(t/rho)@readers.T
    predicted=(lambdas[0]*(z@readers.T+products@(readers@down).T+bias@readers.T)+lambdas[1]*(x0@readers.T))/rho
    replay=float((direct-predicted).norm()/direct.norm())
    checks.extend([raw_ranks==[17]*9,max(errors)<1e-8,max(control_raw_errors)<1e-8,replay<1e-10,
                   bool(torch.isfinite(g).all()),bool(torch.isfinite(predicted).all())])
    observed_mean=mean([row['mean_squared_cosine'] for row in folded])
    control_means=[row['mean_overlap'] for row in control]
    eigen=torch.linalg.eigvalsh(g).flip(0)
    count_shared=sum(row['principal_cosines'][0]>=.95 for row in folded)
    torch.save(dict(metric=g.cpu(),traceless_metric=gtr.cpu(),qk_readers=a.cpu(),ov_readers=b.cpu(),producer_trace=trace.cpu()),cache)
    valid=all(checks)
    result=dict(predictions={'pred_a_instrument':valid,
            'pred_b_specific_producer_alignment':valid and observed_mean>max(control_means) and observed_mean-mean(control_means)>=.05,
            'pred_c_shared_function_screen':valid and count_shared>=6},
        raw=raw,folded=folded,traceless=tracefree,head_pairings=pairings,rotated_controls=control,
        summary=dict(raw_mean=mean([row['mean_squared_cosine'] for row in raw]),folded_mean=observed_mean,
            traceless_mean=mean([row['mean_squared_cosine'] for row in tracefree]),
            rotated_control_mean=mean(control_means),rotated_control_max=max(control_means),
            mismatched_head_mean=mean([row['mean_squared_cosine'] for row in pairings if row['qk_head']!=row['ov_head']]),
            heads_with_cosine_at_least_point95=count_shared),
        geometry=dict(producer_top1_energy=float(eigen[0]/eigen.sum()),producer_top16_energy=float(eigen[:16].sum()/eigen.sum()),
            producer_rank90=int(torch.searchsorted(eigen.cumsum(0),.9*eigen.sum()))+1,
            radial_coefficient_fraction=float(trace.square().sum()/1152/g.trace())),
        controls=dict(maximum_dense_gram_error=max(errors),maximum_raw_geometry_change=max(control_raw_errors),
            local_full_port_replay_error=replay,raw_value_ranks=raw_ranks),
        interface=dict(mlp16_bias_retained=True,block17_reentry=lambdas.tolist(),source_rms_retained=True,
            base_value_stream_separate=True,attention_value_mix=mix,earlier_model_background_retained=True),
        cache=dict(path=str(cache),sha256=digest(cache),bytes=cache.stat().st_size,ephemeral=True),
        source_feature_cache_sha256=source['sha256'],component_cache_sha256=component['sha256'],
        price=dict(body_forwards=0,corpus_access=False,optimized_parameters=0,producer_gram_numbers=1152**2),
        wall_seconds=time.perf_counter()-start,
        scope='Frozen QK/OV source function-space comparison through MLP16 quadratic producer. Local interfaces replayed; no semantic identification, behavioral evidence or full-path replacement.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ['raw','folded','traceless','head_pairings','rotated_controls']},indent=2),flush=True)


if __name__=='__main__':main()
