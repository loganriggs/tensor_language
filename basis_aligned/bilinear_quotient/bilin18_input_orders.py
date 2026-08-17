"""Does shallow=compressible go the other way, on the input side?

User question (2026-08-17). Within one bilinear layer the input-side interaction order
is bounded at 2 BY ARCHITECTURE (each coefficient is exactly quadratic in the layer's
own input) -- so any depth beyond pairwise on the input side must come from downstream
propagation, not from the layer itself. Measure: 5-band Mobius on the layer's INPUT
(patch the xhat fed to the MLP only, bands of the input PCA; residual bypass
untouched), solo+pair share of the full-input deletion, at layers 1 and 16.
REGISTERED PREDICTIONS:
  (a) input-side solo+pair share at layer 1 EXCEEDS its output-side 24% (the layer's
      own order-2 bound pulls the input side shallower);
  (b) the L1-vs-L16 input-side gap is SMALLER than the output-side gap (24% vs 99%),
      since the architecture equalises the layer-local part;
  (c) but layer 1's input side stays BELOW layer 16's (downstream propagation of the
      deleted-input effects still runs through the deep middle)."""
import json, sys, time, torch, itertools
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from tier2_model import rope_tables, apply_rot
NH,HD,D=9,128,1152
BANDS=((0,32),(32,128),(128,256),(256,512),(512,1152))
IN_PATCH={}     # li -> (Q, cbar)

@torch.no_grad()
def fwd_in(idx):
    B,T=idx.shape
    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
    cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16')
    cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
    for li in range(len(m.transformer.h)):
        blk=m.transformer.h[li]; x=blk.lambdas[0]*x+blk.lambdas[1]*x0
        a=blk.attn
        hcur=F.rms_norm(x,(D,))
        def qk(l):
            z=F.rms_norm(l(hcur).view(B,T,NH,HD),(HD,))
            return apply_rot(z,cosb,sinb)
        v=a.c_v(hcur).view(B,T,NH,HD)
        if v1 is None: v1=v
        v=(1-a.lamb)*v+a.lamb*v1.view_as(v)
        q,k1_,q2,k2=qk(a.c_q),qk(a.c_k),qk(a.c_q2),qk(a.c_k2)
        s1=torch.einsum('bqhd,bkhd->bhqk',q,k1_)/HD
        s2=torch.einsum('bqhd,bkhd->bhqk',q2,k2)/HD
        pat=(s1*s2).masked_fill(~mask,0.0)
        x=x+a.c_proj(torch.einsum('bhqk,bkhd->bqhd',pat,v).reshape(B,T,-1))
        xhat=F.rms_norm(x,(D,)); mlp=blk.mlp
        xin=xhat
        if li in IN_PATCH:
            Q,cbar=IN_PATCH[li]
            c=xhat.float()@Q
            xin=(xhat.float()-(c-cbar)@Q.T).to(xhat.dtype)
        mo=mlp.Down(mlp.Left(xin)*mlp.Right(xin))+mlp.Down_bias
        x=x+mo
    logits=30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)
    V_=logits.shape[-1]
    return F.cross_entropy(logits[:,:-1].reshape(-1,V_).float(),
                           idx[:,1:].reshape(-1),reduction='none').view(B,T-1)

def held_in():
    from bilin18_joint_removal import HELD, B0
    tot,n=0.0,0
    for i in range(0,HELD.shape[0],B0):
        ce=fwd_in(HELD[i:i+B0].to(DEV))
        tot+=float(ce.sum()); n+=ce.numel()
    return tot/n

@torch.no_grad()
def collect_in(li):
    ins=[]
    def hook(mod,inp,o): ins.append(inp[0].detach().reshape(-1,D).float())
    h=m.transformer.h[li].mlp.register_forward_hook(hook)
    for i in range(0,60,6):
        b=FW[i:i+6,:513].to(DEV)
        m(b[:,:-1].contiguous(), b[:,1:].contiguous())
    h.remove()
    X=torch.cat(ins); return X.mean(0), X

def main():
    t0=time.time()
    base=held_in()
    out={'layers':{}}
    for li in (1,16):
        Xb,X=collect_in(li)
        _,_,Vh=torch.linalg.svd((X-Xb).float(), full_matrices=False)
        spans={b: orth(Vh[b[0]:b[1]].T) for b in BANDS}
        def val(bs):
            Q=torch.cat([spans[b] for b in bs],1)
            IN_PATCH[li]=(Q,Xb@Q)
            try: return held_in()-base
            finally: IN_PATCH.pop(li)
        solo={b: val([b]) for b in BANDS}
        psum=sum(val([a,b])-solo[a]-solo[b]
                 for a,b in itertools.combinations(BANDS,2))
        full=val(list(BANDS))
        share=(sum(solo.values())+psum)/max(abs(full),1e-9)
        out['layers'][li]={'solo_sum':sum(solo.values()),'pair':psum,'full':full,
                           'share':share}
        print(f'layer {li:2d} INPUT-side: full {full:+.4f} -> solo+pair share '
              f'{100*share:.0f}%  (output-side was {24 if li==1 else 99}%)',flush=True)
    s1=out['layers'][1]['share']; s16=out['layers'][16]['share']
    pa=s1>0.24; pb=(s16-s1)<(0.99-0.24); pc=s1<s16
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['pred_c']=bool(pc)
    print(f"\n(a) L1 input-side > 24%: {'HELD' if pa else 'FAILED'} | "
          f"(b) gap smaller than output-side: {'HELD' if pb else 'FAILED'} | "
          f"(c) L1 still below L16: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                       'bilin18_input_orders_results.json','w'),indent=1)
    print(f'wrote bilin18_input_orders_results.json ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
