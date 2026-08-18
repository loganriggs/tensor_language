"""Bulletproof the report's closing sentence: attn14's certified Track-1
score (+0.277, frozen regime, section 187) was computed on text rows
384-448; its PHENOMENON has a fresh replication but the SCORE itself does
not. Re-score on rows 320-384: gain-frozen attn14 fingerprint (mean-ablate
attention at L14, final rms clamped to the clean run) vs token difficulty.

REGISTERED PREDICTIONS: (a) fresh frozen-regime score >= +0.15 (the
certification bar; original +0.277); (b) the free-regime score on the same
window is lower than the frozen score (the regime declaration mattered on
fresh data too; originally +0.183 free vs +0.277 frozen); (c) if (a) fails,
attn14 downgrades to phenomenon-replicated/score-unreplicated and the
report's 'every certified claim' sentence is corrected."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_fingerprints import attn_mean, spearman, NH, HD
from tier2_model import rope_tables, apply_rot
D=1152; R0,R1=320,384
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_attn14_rescore_results.json')

@torch.no_grad()
def pertok_fresh(attn_li, attn_mu, freeze):
    ces=[]
    for i in range(R0,R1,4):
        bb=FW[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        B,T=idx.shape
        cos,sin=rope_tables(T,HD,DEV,torch.float32,'bf16')
        cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
        mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
        def fw2(dmg):
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
                att=a.c_proj(torch.einsum('bhqk,bkhd->bqhd',pat,v)
                             .reshape(B,T,-1))
                if dmg and li==attn_li:
                    att=attn_mu[None,None,:].to(att.dtype).expand_as(att)
                x=x+att
                xhat=F.rms_norm(x,(D,)); mlp=blk.mlp
                x=x+mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
            return x
        xh=fw2(True)
        if freeze:
            xc=fw2(False)
            rms_c=xc.float().pow(2).mean(-1,keepdim=True).sqrt()
            xn=(xh.float()/rms_c.clamp_min(1e-8)).to(xh.dtype)
        else:
            xn=F.rms_norm(xh,(D,))
        lg=(30*torch.tanh(m.lm_head(xn)/30)).float()
        ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                   reduction='none'))
    return torch.cat(ces)

@torch.no_grad()
def base_fresh():
    ces=[]
    for i in range(R0,R1,4):
        bb=FW[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h:
            x,v1=blk(x,v1,x0)
        lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
        ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                   reduction='none'))
    return torch.cat(ces)

@torch.no_grad()
def main():
    t0=time.time()
    mu=attn_mean(14)
    ce0=base_fresh()
    fro=(pertok_fresh(14,mu,True)-ce0).float()
    fre=(pertok_fresh(14,mu,False)-ce0).float()
    b=ce0.float()
    s_fro=spearman(-b,fro); s_fre=spearman(-b,fre)
    pa=s_fro>=0.15; pb=s_fre<s_fro
    out={'score_frozen':s_fro,'score_free':s_fre,
         'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f'fresh attn14: frozen {s_fro:+.3f} (orig +0.277) | '
          f'free {s_fre:+.3f} (orig +0.183)')
    print(f"(a) frozen >= +0.15: {'HELD -- certification fresh-legged' if pa else 'FAILED -- downgrade'}")
    print(f"(b) frozen > free: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
