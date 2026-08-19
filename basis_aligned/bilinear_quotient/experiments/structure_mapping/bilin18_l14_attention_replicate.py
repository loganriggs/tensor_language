"""Replication of the L14 negative (section 137): deleting L14's entire
attention output improved held-out CE by 0.036 on rows 300-364. Replicate on
disjoint rows 384-448, with L10/L15/L16 (the other non-positives) and L13 (a
positive control) alongside. REGISTERED: (a) L14 keeps its sign; (b) magnitude
>= 40% of original; (c) L13 stays positive (the instrument discriminates).

Prior context -- the per-layer ATTENTION ablation profile -- the one component/operator cell the
program never measured (MLP writes are profiled exhaustively; attention writes
never). Per layer 0-17: replace the attention output with its training-rows mean
(mean-ablation, the standard operator), held-out CE damage.

REGISTERED PREDICTIONS: (a) dilution extends to attention: damage rank-correlates
with the attention write's share of the stream (Spearman >= 0.6 across layers
2-15, excluding the special ends); (b) L6 is an outlier above its share
(the section-134 cargo edge makes its attention unusually load-bearing);
(c) the front (L0-L2) carries the largest attention damages (context assembly
happens early -- patterns are contextual from the start, section 126)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from tier2_model import rope_tables, apply_rot
NH,HD,D=9,128,1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_l14_attention_results.json')

@torch.no_grad()
def run_ce(ablate_li=None, att_mean=None, collect_li=None):
    tot,n=0.0,0; means={}
    rows=range(384,448,4) if collect_li is None else range(0,24,6)
    for i in rows:
        step=4 if collect_li is None else 6
        bb=FW[i:i+step,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        B,T=idx.shape
        cos,sin=rope_tables(T,HD,DEV,torch.float32,'bf16')
        cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
        mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
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
            if collect_li is not None and li==collect_li:
                means.setdefault('a',[]).append(att.detach().reshape(-1,D).float())
            if ablate_li is not None and li==ablate_li:
                att=att_mean[None,None,:].to(att.dtype).expand_as(att)
            x=x+att
            xhat=F.rms_norm(x,(D,)); mlp=blk.mlp
            x=x+mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
        if collect_li is None:
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            c=F.cross_entropy(lg.view(-1,lg.size(-1)),tg)
            tot+=float(c)*tg.numel(); n+=tg.numel()
    if collect_li is not None:
        A=torch.cat(means['a'])
        return A.mean(0), float((A-A.mean(0)).pow(2).sum(1).mean())
    return tot/n

@torch.no_grad()
def main():
    t0=time.time()
    base=run_ce()
    print(f'base {base:.4f}\n',flush=True)
    ORIG={14:-0.0363,10:-0.0042,15:-0.0014,16:-0.0123,13:0.0375}
    dmg={}
    for li in (10,13,14,15,16):
        mu,_=run_ce(collect_li=li)
        d=run_ce(ablate_li=li,att_mean=mu)-base
        dmg[li]=d
        print(f'L{li:2d}: attention ablation {d:+.4f} (orig {ORIG[li]:+.4f})',
              flush=True)
    pa=dmg[14]<0
    pb=dmg[14]<=0.4*ORIG[14]
    pc=dmg[13]>0
    out={'damage':{str(k):v for k,v in dmg.items()},
         'pred_a':bool(pa),'pred_b':bool(pb),'ctrl_c':bool(pc)}
    print(f"\n(a) L14 sign replicates: {'HELD' if pa else 'FAILED'}")
    print(f"(b) magnitude >=40%: {'HELD' if pb else 'FAILED'}")
    print(f"(c) L13 positive control: {'HELD' if pc else 'VIOLATED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
