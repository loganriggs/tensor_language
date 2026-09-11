#!/usr/bin/env python3
"""pred_a: compressed metric checks; pred_b: prior max replay1e-8; pred_c: bases1e-10."""
# BQGATE: 0forwards0seq; GPU preparation for CPU coupled subspace optimization.
import os,sys,json,hashlib,time,signal
from pathlib import Path
RUNNER=Path(__file__).resolve();P=RUNNER.parents[3]/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from folded_normalized_router_v1 import rotary
from joint_qk_position_influence_v1 import coefficient
from joint_qk_position_space_v1 import norm
from producer_function_overlap_v1 import support_whitener
from coupled_producer_routing_objective_v1 import trace_quotient
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get_cache(name):
    receipt=json.loads((P/name).read_text());cache=receipt['cache'];assert digest(cache['path'])==cache['sha256']
    return receipt,torch.load(cache['path'],weights_only=True,map_location='cpu')


def main():
    binding=json.loads((P/'COUPLED_PRODUCER_NATIVE_V1_BINDING.json').read_text());assert all(digest(f)==h for f,h in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=0,corpus_access=False,stage='prepare',heads=9)));return
    out=P/'COUPLED_PRODUCER_PREPARE_V1_RESULT.json';assert not out.exists();signal.alarm(1200)
    start=time.perf_counter();torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False
    old,oldcache=get_cache('POSITION_SHARED_QK_SOURCE_V1_RESULT.json')
    prod,prodcache=get_cache('MLP16_PRODUCER_KEY_ENVELOPE_V1_AUDIT.json')
    _,metriccache=get_cache('MLP16_PRODUCER_OVERLAP_V1_RESULT.json');g=metriccache['metric'].cuda();b=metriccache['ov_readers'].cuda()
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    weights=[sd[f'transformer.h.17.attn.{key}.weight'].double().cuda().reshape(9,128,1152) for key in ['c_q','c_k','c_q2','c_k2']]
    q1,k1,q2,k2=weights;positions=list(range(0,511,2))
    rotations=rotary(511,128).cuda().T@torch.stack([rotary(s,128) for s in positions]).cuda()
    heads=[];metric_errors=[];prior_errors=[];basis_errors=[]
    for h in range(9):
        stack=torch.cat((k1[h],k2[h]),0);basis=torch.linalg.qr(stack.T).Q;compressed=stack@basis
        base=(q1[h]@q1[h].T,q2[h]@q2[h].T,q1[h]@q2[h].T)
        grams=tuple(rotations.transpose(-1,-2)@x@rotations for x in base)
        keys=(k1[h]@k1[h].T,k2[h]@k2[h].T,k1[h]@k2[h].T)
        c=coefficient(*grams,*keys);total=norm(*grams,*keys)
        s=compressed.T@(c/total[:,None,None]).mean(0)@compressed;s=(s+s.T)/2
        gc=basis.T@g@basis;gc=(gc+gc.T)/2
        wb=support_whitener(b[h]@g@b[h].T);t=basis.T@g@b[h].T@wb;m=t@t.T
        mu=float(torch.linalg.eigvalsh(s)[-17:].sum());ceiling=prod['heads'][h]['optimal_rank17_mean']
        routing=basis.T@oldcache['frames'][h].cuda();producer=basis.T@prodcache['frames'][h].cuda()
        sharing,_=trace_quotient(producer,m,gc);read,_=trace_quotient(routing,s,torch.eye(256,device='cuda'))
        prior_errors.extend([abs(float(sharing/17)/ceiling-1),abs(float(read)/mu-1),
            abs(min(1.,2*mu)-old['heads'][h]['discovery_mean_touch_upper_bound'])])
        metric_errors.extend([abs(float(s.trace())-1),max(0.,-float(torch.linalg.eigvalsh(gc)[0]))])
        basis_errors.extend([float((basis.T@basis-torch.eye(256,device='cuda')).abs().max()),
            float((routing.T@routing-torch.eye(17,device='cuda')).abs().max()),
            float((producer.T@producer-torch.eye(17,device='cuda')).abs().max())])
        heads.append(dict(basis=basis.cpu(),g=gc.cpu(),m=m.cpu(),s=s.cpu(),
            routing=routing.cpu(),producer=producer.cpu(),sharing_ceiling=ceiling,influence_ceiling=mu))
    cache=Path('/dev/shm/bilin18_coupled_producer_native_v1_inputs.pt');assert not cache.exists()
    torch.save(dict(heads=heads,weights=[w.cpu() for w in weights],binding=binding),cache)
    result=dict(predictions={'pred_a_metrics':max(metric_errors)<1e-8,
            'pred_b_prior_replay':max(prior_errors)<1e-8,'pred_c_basis':max(basis_errors)<1e-10},
        maximum_metric_error=max(metric_errors),maximum_prior_replay_error=max(prior_errors),maximum_basis_error=max(basis_errors),
        cache=dict(path=str(cache),sha256=digest(cache),bytes=cache.stat().st_size,ephemeral=True),
        wall_seconds=time.perf_counter()-start,body_forwards=0,corpus_access=False,
        scope='Prepared exact native matrices only; joint native optimizer has not run.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
