"""Content-level reinterpretation of the tail profiles. Section 117: L17's span
damage is 82% norm-mediated, L16's 35%. Section 96 profiled every tail layer with
raw span-ablation CE -- how much of each was the norm channel? Recompute the
PCA-8 span damages for layers 5-17 with the final gain frozen to the same
model's no-damage per-token rms.

REGISTERED PREDICTIONS: (a) norm-mediated share grows with depth (Spearman >=
0.6 across 13 layers -- late spans hold more of the final vector's energy);
(b) the CONTENT-level damage ranking differs from the raw ranking (Kendall tau
<= 0.7 -- the norm channel reordered section 96's profile); (c) control: frozen
gain exact at zero damage (<= 0.002)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from tier2_model import rope_tables, apply_rot
NH,HD,D=9,128,1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_content_profiles_results.json')

@torch.no_grad()
def fwd_arm(idx, span, freeze):
    B,T=idx.shape
    cos,sin=rope_tables(T,HD,DEV,torch.float32,'bf16')
    cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
    def emb(): x=F.rms_norm(m.transformer.wte(idx),(D,)); return x,x,None
    xc,x0c,v1c=emb(); xh,x0h,v1h=emb()
    for li in range(18):
        blk=m.transformer.h[li]; a=blk.attn; mlp=blk.mlp
        def step(x,x0,v1,dmg):
            x=blk.lambdas[0]*x+blk.lambdas[1]*x0
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
            x=x+a.c_proj(torch.einsum('bhqk,bkhd->bqhd',pat,v).reshape(B,T,-1))
            xhat=F.rms_norm(x,(D,))
            mo=mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
            if dmg is not None and li==dmg[0]:
                Q,cbar=dmg[1]
                c=mo.float().reshape(-1,D)@Q
                mo=mo-((c-cbar)@Q.T).to(mo.dtype).view_as(mo)
            return x+mo,v1n
        xc,v1c=step(xc,x0c,v1c,None)
        xh,v1h=step(xh,x0h,v1h,span)
    if freeze:
        rms_c=xc.float().pow(2).mean(-1,keepdim=True).sqrt()
        xn=(xh.float()/rms_c.clamp_min(1e-8)).to(xh.dtype)
    else:
        xn=F.rms_norm(xh,(D,))
    return (30*torch.tanh(m.lm_head(xn)/30)).float()

@torch.no_grad()
def ce(span,freeze):
    tot,n=0.0,0
    for i in range(300,364,4):
        b=FW[i:i+4,:257].to(DEV)
        lg=fwd_arm(b[:,:-1].contiguous(),span,freeze)
        c=F.cross_entropy(lg.view(-1,lg.size(-1)),b[:,1:].reshape(-1))
        tot+=float(c)*(b.shape[1]-1)*b.shape[0]; n+=(b.shape[1]-1)*b.shape[0]
    return tot/n

@torch.no_grad()
def main():
    t0=time.time()
    base_f=ce(None,False); base_z=ce(None,True)
    ctrl=abs(base_z-base_f)
    print(f'base {base_f:.4f} | frozen-gain base {base_z:.4f} (ctrl {ctrl:.4f})\n',
          flush=True)
    raw={};con={};share={}
    for li in range(5,18):
        accs=[]
        for i in range(0,36,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc); accs.append(acc[0])
        Y=torch.cat(accs); Ybar=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Ybar).float(), full_matrices=False)
        Q=orth(Vh[:8].T)
        span=(li,(Q,Ybar@Q))
        raw[li]=ce(span,False)-base_f
        con[li]=ce(span,True)-base_z
        share[li]=1-con[li]/raw[li] if raw[li]>1e-4 else float('nan')
        print(f'L{li:2d}: raw +{raw[li]:.4f} | content +{con[li]:.4f} | '
              f'norm share {share[li]:.0%}',flush=True)
    ls=sorted(raw)
    sh=[share[li] for li in ls if share[li]==share[li]]
    li_s=[li for li in ls if share[li]==share[li]]
    a=torch.tensor(li_s,dtype=torch.float); b=torch.tensor(sh)
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std(); rb=(rb-rb.mean())/rb.std()
    sp=float((ra*rb).mean())
    rr=torch.tensor([raw[li] for li in ls]).argsort()
    cr=torch.tensor([con[li] for li in ls]).argsort()
    n=len(ls); conc=0
    for i in range(n):
        for j in range(i+1,n):
            s1=(raw[ls[i]]-raw[ls[j]])*(con[ls[i]]-con[ls[j]])
            conc+=1 if s1>0 else -1
    tau=conc/(n*(n-1)/2)
    pa=sp>=0.6; pb=tau<=0.7; pc=ctrl<=0.002
    out={'raw':{str(k):v for k,v in raw.items()},
         'content':{str(k):v for k,v in con.items()},
         'norm_share':{str(k):(share[k] if share[k]==share[k] else None) for k in share},
         'spearman_share_depth':sp,'kendall_raw_vs_content':tau,
         'pred_a':bool(pa),'pred_b':bool(pb),'ctrl_c':bool(pc)}
    print(f"\n(a) norm share grows with depth (>=0.6): {'HELD' if pa else 'FAILED'} ({sp:+.2f})")
    print(f"(b) ranking reordered (tau <=0.7): {'HELD' if pb else 'FAILED'} ({tau:+.2f})")
    print(f"(c) control exact: {'HELD' if pc else 'VIOLATED'} ({ctrl:.4f})")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
