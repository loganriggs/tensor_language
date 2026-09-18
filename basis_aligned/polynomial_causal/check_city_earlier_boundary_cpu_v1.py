"""CPU full-suffix certificate for a new residual6-prefix input boundary."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path[:0]=[str(ROOT/'basis_aligned/bilinear_quotient/ops'),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v4 import CONTROL_PAIRS
from city_earlier_boundary_v1 import execute

@torch.no_grad()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    out=P/'CITY_EARLIER_BOUNDARY_CPU_V1_RESULT.json';assert not out.exists();start=time.perf_counter();signal.alarm(180);torch.set_num_threads(2)
    rows=json.loads((P/'CITY_FULL_PILE_V2_ROWS.json').read_text())['rows'];groups,mapping=group_rows(rows)
    prefix=torch.load(P/'CITY_PREFIX7_CPU_V1_ARTIFACT.pt',weights_only=True)['fixtures']
    contexts=torch.load(P/'CITY_MLP7_INTEGRATED_V1_CPU_ARTIFACT.pt',weights_only=True)['fixtures']
    attention=torch.load(P/'CITY_ATTENTION7_GENERATOR_V1_PROGRAM.pt',weights_only=True)
    reader=torch.load(P/'CITY_MLP7_READERS_V1_PROGRAM.pt',weights_only=True)
    original_head=torch.load(P/'extracted_circuits/typed_face_single_head_norm_v1/program.pt',weights_only=True)['head8']
    head={k:original_head[k] for k in ['k1','k2','current_value','output','mixture']}
    writes=[];local_errors=[];state_prices=[];outside=[]
    for f,c in zip(prefix,contexts,strict=True):
        x=c['inputs'];city=f['city']
        delta=execute(attention,reader,head,f['residual6'][:,:city+1],f['token_ids'][:,:city+1],city,x['rotated_queries'],x['destination'],input_rms=x['input_rms'])
        writes.append(delta);local_errors.append(float((delta-c['expected_native_delta']).norm()/c['expected_native_delta'].norm()))
        outside.append(float(delta[:,~x['destination']].abs().max()))
        state_prices.append((city+1)*1152+x['rotated_queries'].numel()+x['input_rms'].numel())
    from fastload import load_model_fast
    model=load_model_fast().eval();assert all(p.device.type=='cpu' for p in model.parameters())
    ids=torch.zeros(40,max(len(r['ids']) for r in groups),dtype=torch.long)
    for i,row in enumerate(groups):ids[i,:len(row['ids'])]=torch.tensor(row['ids'])
    edit=torch.zeros(40,ids.shape[1],1152)
    for i,w in enumerate(writes):edit[i,:w.shape[1]]=w[0].float()
    state={'arm':0}
    def post8(module,args,out):return (out[0]+edit,out[1]) if state['arm'] else out
    handle=model.transformer.h[8].attn.register_forward_hook(post8)
    values=torch.zeros(2,40,10,dtype=torch.float64);calls=0
    try:
        for arm in range(2):
            state['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;first=None
            for block in model.transformer.h:x,first=block(x,first,x0);calls+=1
            last=x[torch.arange(40),torch.tensor([len(r['ids'])-1 for r in groups])]
            scores=30*torch.tanh(model.lm_head(F.rms_norm(last,(1152,)))/30)
            for i,row in enumerate(groups):
                for j,(left,right) in enumerate(row['endpoint_pairs']+CONTROL_PAIRS):values[arm,i,j]=scores[i,left]-scores[i,right]
    finally:handle.remove()
    v=expand(values,mapping,6);old=torch.load(P/'CITY_FULL_STRENGTH_V1_ARTIFACT.pt',weights_only=True)['values'][:2]
    absolute=[float((v[i]-old[i]).abs().max()) for i in range(2)];relative=[float((v[i]-old[i]).norm()/old[i].norm()) for i in range(2)]
    effect=v[1]-v[0];target=old[1]-old[0];errors=((effect-target).square().sum(0).sqrt()/target.square().sum(0).sqrt()).tolist()
    count=sum(v.numel() for pack in [attention,reader,head] for v in pack.values() if v.is_floating_point())
    result={'pred_a':absolute[0]<=1e-4 and relative[0]<=1e-5,
            'pred_b':absolute[1]<=1e-4 and relative[1]<=1e-5 and max(errors)<=1e-3,
            'pred_c':max(local_errors)<=1e-4 and max(outside)==0 and bool(torch.isfinite(v).all()) and len(writes)==40 and calls==36 and not torch.cuda.is_initialized(),
            'score_max_abs_by_arm':absolute,'score_relative_by_arm':relative,'effect_errors_by_reader':errors,'local_write_error':max(local_errors),
            'floating_scalars':count,'fp32_bytes':4*count,'native_state_scalar_range':[min(state_prices),max(state_prices)],
            'sequence_equivalent_forwards':80,'batched_block_calls':calls,'seconds':time.perf_counter()-start,
            'scope':'Opened full-model CPU certificate at earlier residual6 boundary, native query fields and mixed8 RMS still external. Not a GPU certificate, fresh transfer, source composition or total compression claim.',
            'source_shas':{str(P/n):hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['CITY_EARLIER_BOUNDARY_CPU_V1_PREREGISTRATION.md','check_city_earlier_boundary_cpu_v1.py','city_earlier_boundary_v1.py']}}
    torch.save({'values':v,'writes':writes},P/'CITY_EARLIER_BOUNDARY_CPU_V1_ARTIFACT.pt');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'},indent=2));signal.alarm(0)
if __name__=='__main__':main()
