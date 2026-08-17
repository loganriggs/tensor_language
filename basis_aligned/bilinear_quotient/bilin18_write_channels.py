"""The write-side symmetry check. Section 131: attention and MLP maintain
separate input watch-lists. Do they also WRITE into separate stream channels?
Per layer (2,5,9,13,16): top-8 covariance directions of the attention write vs
the MLP write (collected on data), median principal cosine, against a
covariance-matched null (random 8-spans weighted by the layer's stream
covariance).

REGISTERED PREDICTIONS (skeptical, continuing the separation motif): (a) median
alignment <= null + 0.1 at >= 4/5 layers (separate write channels); alternative:
attention and MLP write into the same subspace (shared channel -- would say the
separation is read-side only). Null (b): the matched null itself <= 0.45 (else
the instrument cannot discriminate)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from tier2_model import rope_tables, apply_rot
NH,HD,D=9,128,1152
LAYERS=(2,5,9,13,16)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_write_channels_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    at={li:[] for li in LAYERS}; mo={li:[] for li in LAYERS}
    st={li:[] for li in LAYERS}
    for i in range(0,24,6):
        idx=FW[i:i+6,:257].to(DEV)
        B,T=idx.shape
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16')
        cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
        mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
        for li in range(17):
            blk=m.transformer.h[li]; a=blk.attn
            x=blk.lambdas[0]*x+blk.lambdas[1]*x0
            if li in LAYERS: st[li].append(x.detach().reshape(-1,D).float())
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
            att=a.c_proj(torch.einsum('bhqk,bkhd->bqhd',pat,v).reshape(B,T,-1))
            if li in LAYERS: at[li].append(att.detach().reshape(-1,D).float())
            x=x+att
            xhat=F.rms_norm(x,(D,)); mlp=blk.mlp
            w=mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
            if li in LAYERS: mo[li].append(w.detach().reshape(-1,D).float())
            x=x+w
    g=torch.Generator(device=DEV).manual_seed(0)
    res={}; ok=0; nullmax=0
    for li in LAYERS:
        def top8(lst):
            X=torch.cat(lst); Xc=X-X.mean(0)
            _,_,Vh=torch.linalg.svd(Xc[:20000],full_matrices=False)
            return orth(Vh[:8].T)
        A=top8(at[li]); Mo=top8(mo[li])
        X=torch.cat(st[li]); Xc=X-X.mean(0)
        C=Xc.T@Xc/Xc.shape[0]
        ev,U=torch.linalg.eigh(C.double())
        Ch=((U*ev.clamp_min(0).sqrt())@U.T).float()
        cs=[]
        for _ in range(60):
            R1=orth(Ch@torch.randn(D,8,device=DEV,generator=g))
            R2=orth(Ch@torch.randn(D,8,device=DEV,generator=g))
            s_=torch.linalg.svdvals(R1.T@R2)
            cs.append(float(sorted(s_.tolist())[4]))
        null=sorted(cs)[len(cs)//2]
        s_=torch.linalg.svdvals(A.T@Mo)
        med=float(sorted(s_.tolist())[4])
        res[li]={'align':med,'null':null}
        nullmax=max(nullmax,null)
        if med<=null+0.1: ok+=1
        print(f'L{li:2d}: attn-write vs mlp-write median cos {med:.2f} '
              f'(matched null {null:.2f})',flush=True)
    pa=ok>=4; pb=nullmax<=0.45
    out={'per_layer':{str(k):v for k,v in res.items()},
         'pred_a_separate':bool(pa),'instrument_b':bool(pb)}
    print(f"\n(a) separate write channels (>=4/5): {'HELD' if pa else 'FAILED'} ({ok}/5)")
    print(f"(b) instrument discriminates (null <=0.45): {'HELD' if pb else 'VIOLATED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
