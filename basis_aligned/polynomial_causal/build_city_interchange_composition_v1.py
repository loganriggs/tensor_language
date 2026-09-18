"""Native city key/value exchange factorial at fixed recipient query fields."""
import hashlib,json,os,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path[:0]=[str(ROOT/'basis_aligned/bilinear_quotient/ops'),str(ROOT)]
from city_residual6_fields_v1 import from_projected,city_write
@torch.no_grad()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(2);start=time.perf_counter()
    fixtures=torch.load(P/'CITY_FULL_INTERCHANGE_V1_ARTIFACT.pt',weights_only=True)['fixtures']
    table=torch.load(P/'CITY_FULL_INTERCHANGE_V1_ATTENTION7.pt',weights_only=True);head=torch.load(P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_HEAD8.pt',weights_only=True)
    ids=torch.cat([f['inputs']['recipient_tokens'] for f in fixtures]);res6=torch.cat([f['inputs']['recipient_residual6'] for f in fixtures])
    lookup={int(t):i for i,t in enumerate(table['token_ids'])};index=torch.tensor([[lookup[int(t)] for t in row] for row in ids]);initial=F.embedding(index,table['initial_table']);first=F.embedding(index,table['first_table'])
    from fastload import load_model_fast
    model=load_model_fast().eval();res7,_=model.transformer.h[7](res6,first,initial);lam=table['lambdas8'];current=F.rms_norm(lam[0]*res7+lam[1]*initial,(1152,))
    projected=F.linear(current.double(),head['sources'].double());fields=from_projected(projected,first,head)
    # Independent FP32 native-form self reference; full swap also has a prior actual-native capture.
    raw=F.linear(current,head['sources']).split(128,-1);t=ids.shape[1]
    inv=1/(10000**(torch.arange(0,128,2,dtype=torch.float32)/128));angles=torch.outer(torch.arange(t,dtype=torch.float32),inv);co=angles.cos().bfloat16().float();si=angles.sin().bfloat16().float()
    def rotate(x):
        x=F.rms_norm(x,(128,));a,b=x.chunk(2,-1);return torch.cat([a*co+b*si,-a*si+b*co],-1)
    q1,k1,q2,k2=[rotate(x) for x in raw[:4]];value=(1-head['mixture'])*raw[4]+head['mixture']*first[...,256:384]
    writes=[];errors=[];self_errors=[];closure=[];outside=[];cross_ratios=[]
    for i,f in enumerate(fixtures):
        donor=i^1;city=f['inputs']['city'];mask=f['inputs']['destination'];recipient={n:x[i:i+1] for n,x in fields.items()};other={n:x[donor:donor+1] for n,x in fields.items()}
        own=city_write(recipient,recipient,head,city,mask)
        keysource=dict(recipient,k1=other['k1'],k2=other['k2']);valuesource=dict(recipient,value=other['value'])
        dk=city_write(recipient,keysource,head,city,mask)-own;dv=city_write(recipient,valuesource,head,city,mask)-own
        joint=city_write(recipient,other,head,city,mask)-own;reference=f['native_delta'].double();cross=reference-dk-dv
        route=((q1[i:i+1]*k1[i:i+1,city,None]).sum(-1)/128)*((q2[i:i+1]*k2[i:i+1,city,None]).sum(-1)/128)
        own32=F.linear(route[...,None]*value[i:i+1,city,None],head['output'])*mask[None,:,None]
        self_errors.append(float((own-own32.double()).norm()/own32.double().norm()));errors.append(float((joint-reference).norm()/reference.norm()))
        packed=torch.stack([dk,dv,dk+dv,reference]);writes.append(packed);closure.append(float((dk+dv+cross-reference).norm()/reference.norm()));outside.append(float(packed[:,:,~mask].abs().max()));cross_ratios.append(float(cross.norm()/reference.norm()))
    r={'pred_a':max(errors)<=1e-4 and max(self_errors)<=1e-4,'pred_b':max(closure)<=1e-12 and max(outside)==0 and all(bool(torch.isfinite(w).all()) for w in writes),'max_native_swap_replay_error':max(errors),'max_native_self_replay_error':max(self_errors),'max_head_closure_error':max(closure),'local_cross_over_joint_range':[min(cross_ratios),max(cross_ratios)],'sequences':40,'native_block_calls':1,'full_model_forwards':0,'seconds':time.perf_counter()-start,'scope':'Opened local head exchange algebra only; behavioral composition and random specificity untested.','source_shas':{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['CITY_INTERCHANGE_COMPOSITION_V1_PREREGISTRATION.md','build_city_interchange_composition_v1.py','city_residual6_fields_v1.py']}}
    torch.save({'writes':writes,'arms':['keys','value','mains','joint']},P/'CITY_INTERCHANGE_COMPOSITION_V1_WRITES.pt');(P/'CITY_INTERCHANGE_COMPOSITION_V1_CPU_RESULT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
