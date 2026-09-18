"""Native city-side MLP7 key/value slot factorial at fixed denominators."""
import hashlib,json,os,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path[:0]=[str(ROOT/'basis_aligned/bilinear_quotient/ops'),str(ROOT)]
from city_source7_present_generator_v1 import from_projected
@torch.no_grad()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized();torch.set_num_threads(2);start=time.perf_counter()
    fixtures=torch.load(P/'CITY_SOURCE7_FRESH_V1_ARTIFACT.pt',weights_only=True)['fixtures']
    program=torch.load(P/'CITY_SOURCE7_FRESH_V1_ATTENTION7.pt',weights_only=True);head=torch.load(P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_HEAD8.pt',weights_only=True)
    ids=torch.cat([f['inputs']['token_ids'] for f in fixtures]);res6=torch.cat([f['inputs']['residual6'] for f in fixtures])
    lookup={int(t):i for i,t in enumerate(program['token_ids'])};indices=torch.tensor([[lookup[int(t)] for t in row] for row in ids])
    initial=F.embedding(indices,program['initial_table']);first=F.embedding(indices,program['first_table']);cache={}
    from fastload import load_model_fast
    model=load_model_fast().eval()
    def capture(module,args,out):cache['mlp7']=out.clone()
    hook=model.transformer.h[7].mlp.register_forward_hook(capture)
    try:res7,_=model.transformer.h[7](res6,first,initial)
    finally:hook.remove()
    lam=program['lambdas8'];mixed=lam[0]*res7+lam[1]*initial;current=F.rms_norm(mixed,(1152,))
    rho=(mixed.double().square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
    projected=F.linear(current.double(),head['sources'].double());m=F.linear(lam[0].double()*cache['mlp7'].double()/rho,head['sources'].double())
    writes=[];full_errors=[];group_errors=[];closure=[];outside=[]
    for i,f in enumerate(fixtures):
        city=f['inputs']['city'];mask=f['inputs']['destination'];inherit=first[i:i+1,city,256:384]
        full,present,_=from_projected(projected[i:i+1],m[i:i+1],inherit,head,city,mask)
        keys=m[i:i+1].clone();keys[...,512:]=0
        values=torch.zeros_like(keys);values[...,512:]=m[i:i+1,...,512:]
        _,key_inclusive,_=from_projected(projected[i:i+1],keys,inherit,head,city,mask)
        _,value_inclusive,_=from_projected(projected[i:i+1],values,inherit,head,city,mask)
        reference=f['native_delta'].double();keymain=present-value_inclusive;valuemain=present-key_inclusive
        cross=reference-keymain-valuemain;main=keymain+valuemain;joint=main+cross
        packed=torch.stack([reference,keymain,valuemain,cross,main,joint]);writes.append(packed)
        full_errors.append(float((full-f['native_full_delta'].double()).norm()/f['native_full_delta'].double().norm()))
        group_errors.append(float((present-reference).norm()/reference.norm()));closure.append(float((joint-reference).norm()/reference.norm()))
        outside.append(float(packed[:,:,~mask].abs().max()))
    result={'pred_a':max(full_errors)<=1e-4 and max(group_errors)<=1e-4,'pred_b':max(closure)<=1e-12 and max(outside)==0 and all(bool(torch.isfinite(w).all()) for w in writes),'full_write_replay_error':max(full_errors),'group_replay_error':max(group_errors),'closure_error':max(closure),'sequences':len(writes),'native_block_calls':1,'full_model_forwards':0,'seconds':time.perf_counter()-start,'scope':'Opened native city-source key/value factorial; both key factors changed together; no behavioral or composition verdict.','source_shas':{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['CITY_SOURCE7_SLOT_FACTORIAL_V1_PREREGISTRATION.md','build_city_source7_slot_factorial_v1.py','city_source7_present_generator_v1.py']}}
    torch.save({'writes':writes,'arms':['present','keymain','valuemain','cross','mains','joint']},P/'CITY_SOURCE7_SLOT_FACTORIAL_V1_WRITES.pt');(P/'CITY_SOURCE7_SLOT_FACTORIAL_V1_CPU_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
