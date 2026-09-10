"""Exact mean/difference regrouping of cancelling fitted products; CPU only.

Greedily pair disjoint products with |input-form cosine|>=.99 and a signed
input/output cosine product<=-.99. Replay bar <=1e-9 on fixed Gaussian inputs;
cancellation-reduction hypothesis: component energy ratio reduces by >=10x.
This changes arithmetic coordinates only, never the fitted function or error.
"""
import hashlib,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from structured_quadratic_models_v1 import QuadraticModel
from joint_quadratic_fit_v1 import product_cross

P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')

def main():
    start=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(9116301)
    cp=P/'STRUCTURED_FIT_V1_weight_product_s0_CHECKPOINT.pt'
    state=torch.load(cp,map_location='cpu',weights_only=False)
    model=QuadraticModel('product',1152);model.load_state_dict(state['best']['model'])
    with torch.no_grad():a,b,_=model.components()
    w=state['best']['writer'];n=len(a)
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    uw=sd['lm_head.weight'].double()@w;kg=uw.T@uw;g=product_cross(a,b,a,b)
    ic=g/(g.diag()[:,None]*g.diag()[None,:]).sqrt()
    oc=kg/(kg.diag()[:,None]*kg.diag()[None,:]).sqrt()
    candidates=[]
    for i in range(n):
        for j in range(i+1,n):
            if abs(float(ic[i,j]))>=.99 and float(ic[i,j]*oc[i,j])<=-.99:
                candidates.append((abs(float(ic[i,j])),i,j))
    pairs=[];used=set()
    for _,i,j in sorted(candidates,reverse=True):
        if i not in used and j not in used:pairs.append((i,j));used.update([i,j])
    aa=[];bb=[];ww=[]
    def append(x,y,z):
        nx=x.norm();ny=y.norm()
        if nx==0 or ny==0:return
        aa.append(x/nx);bb.append(y/ny);ww.append(z*nx*ny)
    for i in range(n):
        if i not in used:append(a[i],b[i],w[:,i])
    for i,j in pairs:
        a1,b1,w1=a[i],b[i],w[:,i]
        sign=1 if ic[i,j]>=0 else -1
        a2,b2,w2=sign*a[j],b[j],sign*w[:,j]
        if abs(float(a1@b2))+abs(float(b1@a2))>abs(float(a1@a2))+abs(float(b1@b2)):a2,b2=b2,a2
        if a1@a2<0:a2,b2=-a2,-b2
        A,B=(a1+a2)/2,(b1+b2)/2
        da,db=(a1-a2)/2,(b1-b2)/2
        W,dw=(w1+w2)/2,(w1-w2)/2
        append(A,B,2*W);append(da,db,2*W)
        append(A,db,2*dw);append(da,B,2*dw)
    A,B,W=torch.stack(aa),torch.stack(bb),torch.stack(ww,dim=1)
    Unew=sd['lm_head.weight'].double()@W;gnew=product_cross(A,B,A,B);knew=Unew.T@Unew
    old_energy=(g*kg).sum();new_energy=(gnew*knew).sum()
    old_ratio=float((g.diag()*kg.diag()).sum()/old_energy)
    new_ratio=float((gnew.diag()*knew.diag()).sum()/new_energy)
    x=torch.randn(256,1152,dtype=torch.float64)
    y=((x@a.T)*(x@b.T))@w.T;z=((x@A.T)*(x@B.T))@W.T
    replay=float((y-z).norm()/y.norm())
    energy_replay=abs(float(new_energy/old_energy)-1)
    program=P/'CANCELLING_PRODUCT_BLOCKS_V1_PROGRAM.pt'
    torch.save(dict(a=A,b=B,w=W,pairs=pairs,original_checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest()),program)
    result=dict(schema='cancelling.product.blocks.v1',pairs=pairs,original_products=n,expanded_products=len(A),
        original_cancellation_ratio=old_ratio,regrouped_cancellation_ratio=new_ratio,reduction_factor=old_ratio/new_ratio,
        fixed_gaussian_states=len(x),residual_function_replay_relative_error=replay,coefficient_energy_replay_relative_error=energy_replay,
        predictions=dict(function_replay=replay<=1e-9 and energy_replay<=1e-9,cancellation_reduction=old_ratio/new_ratio>=10),
        price=dict(original_scalar_products=n,expanded_scalar_products=len(A),
            explicit_program_parameter_numbers=A.numel()+B.numel()+W.numel(),
            shared_block_reader_vectors=4*len(pairs)+2*(n-2*len(pairs)),
            shared_block_writer_vectors=2*len(pairs)+(n-2*len(pairs)),
            artifact_bytes=program.stat().st_size),
        artifact_sha256=hashlib.sha256(program.read_bytes()).hexdigest(),wall_seconds=time.perf_counter()-start,
        scope='Same fitted function regrouped exactly. No better reconstruction, convergence, causal identification or simplicity claim; expanded arithmetic costs more products.')
    with (P/'CANCELLING_PRODUCT_BLOCKS_V1_AUDIT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
