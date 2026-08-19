"""Typed-edge audit extended to the attention profile: how much of each
attention ablation's damage (section 137) is norm-typed (energy->final gain)?
Reuse the per-arm gain-frozen instrument: attention mean-ablation at L1 (raw
+0.302), L2 (+0.205), L6 (+0.073), L14 (-0.035), with the final per-token rms
clamped to the same model's no-damage run.

REGISTERED PREDICTIONS: (a) the front attention damages are substantially
norm-typed: L1 and L2 drop >= 40% when gain-frozen (their huge writes carry
energy the final norm depends on); (b) L6's cargo edge is content-typed: drops
< 30%; (c) L14's negative attenuates toward zero under frozen gain (its benefit
partly flows through the gain channel)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from tier2_model import rope_tables, apply_rot
NH,HD,D=9,128,1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_attn_norm_share_results.json')

@torch.no_grad()
def run(ablate_li=None, att_mean=None, freeze=False):
    tot,n=0.0,0
    for i in range(300,364,4):
        bb=FW[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        B,T=idx.shape
        cos,sin=rope_tables(T,HD,DEV,torch.float32,'bf16')
        cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
        mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
        def fw(dmg):
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for li in range(18):
                blk=m.transformer.h[li]; a=blk.attn
                x=blk.lambdas[0]*x+blk.lambdas[1]*x0
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
                if dmg and li==ablate_li:
                    att=att_mean[None,None,:].to(att.dtype).expand_as(att)
                x=x+att
                xhat=F.rms_norm(x,(D,)); mlp=blk.mlp
                x=x+mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
            return x
        xh=fw(ablate_li is not None)
        if freeze and ablate_li is not None:
            xc=fw(False)
            rms_c=xc.float().pow(2).mean(-1,keepdim=True).sqrt()
            xn=(xh.float()/rms_c.clamp_min(1e-8)).to(xh.dtype)
        else:
            xn=F.rms_norm(xh,(D,))
        lg=(30*torch.tanh(m.lm_head(xn)/30)).float()
        c=F.cross_entropy(lg.view(-1,lg.size(-1)),tg)
        tot+=float(c)*tg.numel(); n+=tg.numel()
    return tot/n

@torch.no_grad()
def attn_mean(li):
    caps=[]
    for i in range(0,12,6):
        idx=FW[i:i+6,:257].to(DEV)
        B,T=idx.shape
        cos,sin=rope_tables(T,HD,DEV,torch.float32,'bf16')
        cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
        mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for li2 in range(li+1):
            blk=m.transformer.h[li2]; a=blk.attn
            x=blk.lambdas[0]*x+blk.lambdas[1]*x0
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
            if li2==li:
                caps.append(att.detach().reshape(-1,D).float()); break
            x=x+att
            xhat=F.rms_norm(x,(D,)); mlp=blk.mlp
            x=x+mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
    return torch.cat(caps).mean(0)

@torch.no_grad()
def main():
    t0=time.time()
    base_f=run(); res={}
    for li in (1,2,6,14):
        mu=attn_mean(li)
        raw=run(li,mu,False)-base_f
        con=run(li,mu,True)-base_f
        share=1-con/raw if abs(raw)>1e-4 else float('nan')
        res[li]={'raw':raw,'content':con,'norm_share':share}
        print(f'L{li:2d}: raw {raw:+.4f} | gain-frozen {con:+.4f} | '
              f'norm share {share:.0%}',flush=True)
    pa=res[1]['norm_share']>=0.4 and res[2]['norm_share']>=0.4
    pb=res[6]['norm_share']<0.3
    pc=abs(res[14]['content'])<abs(res[14]['raw'])
    out={'per_layer':{str(k):v for k,v in res.items()},
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"\n(a) front norm-typed (>=40%): {'HELD' if pa else 'FAILED'}")
    print(f"(b) L6 content-typed (<30%): {'HELD' if pb else 'FAILED'}")
    print(f"(c) L14 negative attenuates: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
