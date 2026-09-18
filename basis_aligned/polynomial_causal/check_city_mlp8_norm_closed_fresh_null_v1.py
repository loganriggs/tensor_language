"""Fresh same-boundary random-value null for the norm-closed MLP8 mediator."""
import hashlib,json,os,signal,time
from pathlib import Path
from types import MethodType
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent;ROOT=P.parent
import sys
sys.path[:0]=[str(ROOT/'bilinear_quotient/ops'),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v4 import CONTROL_PAIRS
STEM='CITY_MLP8_NORM_CLOSED_FRESH_V1_NULL'
ROWS='CITY_MLP8_NORM_CLOSED_FRESH_V1_ROWS.json'
SCREEN='CITY_MLP8_NORM_CLOSED_FRESH_V1_SCREEN_ARTIFACT.pt'

@torch.no_grad()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(2); start=time.perf_counter(); signal.alarm(600)
    out=P/(STEM+'_RESULT.json'); assert not out.exists()
    rows=json.loads((P/ROWS).read_text())['rows']; groups,mapping=group_rows(rows); assert len(groups)==40
    screen=torch.load(P/SCREEN,weights_only=True); fixtures=screen['fixtures']; assert len(fixtures)==40
    ids=torch.tensor([r['ids'] for r in groups]); edit=torch.cat([f['inputs']['delta'] for f in fixtures])
    real=torch.cat([f['mlp8_value_delta'] for f in fixtures]).float()
    changes={2:real}; norm_errors=[]
    for k in range(16):
        generated=[]
        for i,row in enumerate(groups):
            b=real[i]; gen=torch.Generator().manual_seed(18104000+1000*k+row['context_id'])
            random=torch.randn(b.shape,generator=gen,dtype=torch.float64)
            random=random-(random*b).sum(-1,keepdim=True)/b.square().sum(-1,keepdim=True).clamp_min(1e-30)*b
            random=random*(b.double().norm(dim=-1,keepdim=True)/random.norm(dim=-1,keepdim=True).clamp_min(1e-30))
            generated.append(random.float())
            norm_errors.append(float(((random.norm(dim=-1)-b.norm(dim=-1)).abs()/b.norm(dim=-1).clamp_min(1e-8)).max()))
        changes[3+ k]=torch.stack(generated)
    state={'arm':0}; errors=[]; other_errors=[]
    from fastload import load_model_fast
    model=load_model_fast().eval(); attn8=model.transformer.h[8].attn; old8=attn8.squared_attention
    def post8(module,args,out):
        return (out[0]+edit,out[1]) if state['arm']>=1 else out
    h8=attn8.register_forward_hook(post8); attn9=model.transformer.h[9].attn; old9=attn9.squared_attention
    def changed(module,q,k,v,q2,k2):
        arm=state['arm']
        if arm>=2:
            new=v.clone(); new[:,:,8]-=changes[arm]
            errors.append(float((v[:,:,8]-new[:,:,8]-changes[arm]).double().norm()/changes[arm].double().norm().clamp_min(1e-30)))
            other_errors.append(float((new[:,:,:8]-v[:,:,:8]).abs().max())); v=new
        return old9(q,k,v,q2,k2)
    attn9.squared_attention=MethodType(changed,attn9)
    values=torch.zeros(19,40,10,dtype=torch.float64); calls=0
    try:
        for arm in range(19):
            state['arm']=arm; x=F.rms_norm(model.transformer.wte(ids),(1152,)); x0=x; first=None
            for block in model.transformer.h: x,first=block(x,first,x0); calls+=1
            scores=30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30)
            for i,row in enumerate(groups):
                for j,(left,right) in enumerate(row['endpoint_pairs']+CONTROL_PAIRS): values[arm,i,j]=scores[i,left]-scores[i,right]
    finally:
        h8.remove(); attn8.squared_attention=old8; attn9.squared_attention=old9
    v=expand(values,mapping,6); e=v-v[1:2]; target=e[2,:,0]; target_rms=float(target.norm())
    null_rms=[float(e[3+k,:,0].norm()) for k in range(16)]
    controls=(e[2,:,1:].square().mean(0).sqrt()/e[2,:,0].square().mean().sqrt().clamp_min(1e-30)).tolist()
    capable=(v[0,:,0][::2]-v[0,:,0][1::2])>=.1
    reversed_rows=[int(i) for i,x in enumerate((v[0,:,0][::2]-v[0,:,0][1::2])) if x<0]
    r={'pred_a':max(norm_errors)<=1e-5 and max(errors)<=1e-5 and max(other_errors)==0 and bool(torch.isfinite(v).all()) and calls==342 and not torch.cuda.is_initialized(),
       'pred_b':target_rms>=2*sorted(null_rms)[7] and all(target_rms>x for x in null_rms),
       'pred_c':max(controls)<=.5,
       'target_rms':target_rms,'null_rms':null_rms,'nulls_beaten':sum(target_rms>x for x in null_rms),'control_over_target':controls,
       'reversed_native_rows':reversed_rows,'max_norm_error':max(norm_errors),'max_subtraction_relative_error':max(errors),'other_head_error':max(other_errors),
       'rows':240,'sequences':40,'body_forwards':760,'batched_block_calls':calls,'seconds':time.perf_counter()-start,
       'scope':'Fresh same-boundary norm-matched random value corrections against norm-closed MLP8 mediator; native z8/upstream city edit/full suffix external; no independent-composition claim.',
       'source_shas':{name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in [ROWS,SCREEN,'CITY_MLP8_NORM_CLOSED_FRESH_V1_NULL_PREREGISTRATION.md','check_city_mlp8_norm_closed_fresh_null_v1.py']}}
    out.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r,indent=2)); signal.alarm(0)
if __name__=='__main__': main()
