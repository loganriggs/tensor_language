"""Where does the refit pipe's drift flow? Section 105-106: individually each of
layers 5-17 linearizes for ~0.03 nats, but the sequential pipe costs +1.56 --
marginals ~4x the individual costs. Hypothesis: the drift compounds through
ATTENTION -- real attention heads read the hybrid's slightly-off residual stream
and produce off-manifold mixes that the downstream linear stand-ins were never
fit for. Test: clamp every layer's attention PATTERN (the s1*s2 score matrices)
to the base model's values (computed on the same input by a parallel clean
forward), while keeping the hybrid's values/streams live.

REGISTERED PREDICTIONS: (a) pattern-clamping cuts the refit-pipe cost by >= 40%
(patterns are the drift channel); (b) clamping alone (no linearization) is nearly
free (<= 0.02 -- control that clamping to the model's own patterns on-distribution
is a no-op up to numerics)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from tier2_model import rope_tables, apply_rot
import bilin18_pipe_refit as PR
NH,HD,D=9,128,1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_clamped_pipe_results.json')

@torch.no_grad()
def fwd_dual(idx, lins, clamp):
    """Run clean and hybrid models in lockstep; if clamp, hybrid uses clean's
    attention patterns."""
    B,T=idx.shape
    cos,sin=rope_tables(T,HD,DEV,torch.float32,'bf16')
    cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
    def emb():
        x=F.rms_norm(m.transformer.wte(idx),(D,)); return x,x,None
    xc,x0c,v1c=emb(); xh,x0h,v1h=emb()
    for li in range(18):
        blk=m.transformer.h[li]
        a=blk.attn; mlp=blk.mlp
        def attn_step(x,x0,v1,pat_override=None):
            x=blk.lambdas[0]*x+blk.lambdas[1]*x0
            xin=x
            hcur=F.rms_norm(x,(D,))
            def qk(l):
                z=F.rms_norm(l(hcur).view(B,T,NH,HD),(HD,))
                return apply_rot(z,cosb,sinb)
            v=a.c_v(hcur).view(B,T,NH,HD)
            v1n=v if v1 is None else v1
            v=(1-a.lamb)*v+a.lamb*v1n.view_as(v)
            q,k1_,q2,k2=qk(a.c_q),qk(a.c_k),qk(a.c_q2),qk(a.c_k2)
            s1=torch.einsum('bqhd,bkhd->bhqk',q,k1_)/HD
            s2=torch.einsum('bqhd,bkhd->bhqk',q2,k2)/HD
            pat=(s1*s2).masked_fill(~mask,0.0)
            use=pat if pat_override is None else pat_override
            x=x+a.c_proj(torch.einsum('bhqk,bkhd->bqhd',use,v).reshape(B,T,-1))
            return x,xin,v1n,pat
        xc,xinc,v1c,patc=attn_step(xc,x0c,v1c)
        xh,xinh,v1h,_=attn_step(xh,x0h,v1h,pat_override=patc if clamp else None)
        def mlp_step(x,xin,lin):
            xhat=F.rms_norm(x,(D,))
            if lin is not None:
                xi=xin.reshape(-1,D).float()
                mo=((xi-lin['bx'])@lin['W']+lin['by']).to(x.dtype).view_as(x)
            else:
                mo=mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
            return x+mo
        xc=mlp_step(xc,xinc,None)
        xh=mlp_step(xh,xinh,lins.get(li))
    lg=m.lm_head(F.rms_norm(xh,(D,)))
    return (30*torch.tanh(lg/30)).float()

@torch.no_grad()
def ce_dual(lins, clamp):
    tot,n=0.0,0
    for i in range(300,380,4):
        b=FW[i:i+4,:257].to(DEV)
        lg=fwd_dual(b[:,:-1].contiguous(), lins, clamp)
        ce=F.cross_entropy(lg.view(-1,lg.size(-1)), b[:,1:].reshape(-1))
        tot+=float(ce)*(b.shape[1]-1)*b.shape[0]; n+=(b.shape[1]-1)*b.shape[0]
    return tot/n

@torch.no_grad()
def main():
    t0=time.time()
    base=ce_dual({},False)
    print(f'base {base:.4f}',flush=True)
    clamp_only=ce_dual({},True)-base
    print(f'clamp-only control: +{clamp_only:.4f}',flush=True)
    # rebuild the sequentially-refit maps (same protocol as pipe_refit)
    PR.LINS={}
    lins={}
    for li in range(5,18):
        PR.LINS=dict(lins)
        lins[li]=PR.fit_layer(li)
    PR.LINS={}
    pipe=ce_dual(lins,False)-base
    pipe_clamped=ce_dual(lins,True)-base
    cut=1-pipe_clamped/pipe if pipe>1e-6 else float('nan')
    pa=cut>=0.4; pb=abs(clamp_only)<=0.02
    out={'base':base,'clamp_only':clamp_only,'pipe':pipe,
         'pipe_clamped':pipe_clamped,'cut':cut,
         'pred_a':bool(pa),'ctrl_b':bool(pb)}
    print(f'refit pipe: +{pipe:.4f} | pattern-clamped pipe: +{pipe_clamped:.4f} '
          f'(cut {cut:.0%})')
    print(f"(a) patterns are the drift channel (cut >=40%): {'HELD' if pa else 'FAILED'}")
    print(f"(b) clamp-only near-free (<=0.02): {'HELD' if pb else 'VIOLATED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
