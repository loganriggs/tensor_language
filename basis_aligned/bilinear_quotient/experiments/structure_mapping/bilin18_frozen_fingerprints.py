"""Gain-frozen fingerprint variant (Track-1 lesson 1, section 186) for the four
pilot components, then re-score each explanation in its declared regime.

REGISTERED PREDICTIONS: (a) mlp16's content-level explanation ("value on easy
tokens") scores >= +0.15 against its GAIN-FROZEN fingerprint (the -0.135 was
regime mismatch, not a wrong story); (b) attn14's story survives the frozen
regime at >= +0.15 (its relief is content-borne -- section 171 measured its
norm share NEGATIVE); (c) the two regimes' fingerprints differ most for mlp16
(|corr between regimes| lower for mlp16 than for attn14)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from tier2_model import rope_tables, apply_rot
NH,HD,D=9,128,1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_frozen_fp_results.json')
PT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    'bilin18_frozen_fingerprints.pt')

def spearman(a,b):
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std().clamp_min(1e-9)
    rb=(rb-rb.mean())/rb.std().clamp_min(1e-9)
    return float((ra*rb).mean())

@torch.no_grad()
def pertok(mlp_span=None, attn_li=None, attn_mu=None, freeze=True):
    ces=[]
    for i in range(384,448,4):
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
                if dmg and attn_li is not None and li==attn_li:
                    att=attn_mu[None,None,:].to(att.dtype).expand_as(att)
                x=x+att
                xhat=F.rms_norm(x,(D,)); mlp=blk.mlp
                mo=mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
                if dmg and mlp_span is not None and li==mlp_span[0]:
                    Q,cbar=mlp_span[1]
                    c=mo.float().reshape(-1,D)@Q
                    mo=mo-((c-cbar)@Q.T).to(mo.dtype).view_as(mo)
                x=x+mo
            return x
        xh=fw(True)
        if freeze:
            xc=fw(False)
            rms_c=xc.float().pow(2).mean(-1,keepdim=True).sqrt()
            xn=(xh.float()/rms_c.clamp_min(1e-8)).to(xh.dtype)
        else:
            xn=F.rms_norm(xh,(D,))
        lg=(30*torch.tanh(m.lm_head(xn)/30)).float()
        ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                   reduction='none'))
    return torch.cat(ces)

@torch.no_grad()
def base_pertok():
    ces=[]
    for i in range(384,448,4):
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
    ce0=base_pertok()
    def mlp_span(li):
        accs=[]
        for i in range(0,36,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc); accs.append(acc[0])
        Y=torch.cat(accs); Ybar=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Ybar).float(), full_matrices=False)
        Q=orth(Vh[:8].T)
        return (li,(Q,Ybar@Q))
    def attn_mean(li):
        caps=[]
        h=m.transformer.h[li].attn
        # reuse pertok's manual loop by capturing on a stats pass
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
    frozen={}
    frozen['mlp16']=(pertok(mlp_span=mlp_span(16))-ce0).cpu().float()
    frozen['mlp9']=(pertok(mlp_span=mlp_span(9))-ce0).cpu().float()
    mu14=attn_mean(14)
    frozen['attn14']=(pertok(attn_li=14,attn_mu=mu14)-ce0).cpu().float()
    mu1=attn_mean(1)
    frozen['attn1']=(pertok(attn_li=1,attn_mu=mu1)-ce0).cpu().float()
    torch.save({'base':ce0.cpu(),'fingerprints':frozen},PT)
    d=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                 'bilin18_fingerprint_atlas.pt')
    free={k:d['fingerprints'][k].float() for k in frozen}
    base=ce0.cpu().float()
    s16=spearman(-base,frozen['mlp16'])
    s14=spearman(-base,frozen['attn14'])
    reg={k:abs(spearman(frozen[k],free[k])) for k in frozen}
    for k in frozen:
        print(f'{k:7s}: net {float(frozen[k].mean()):+.4f} | regime corr '
              f'{reg[k]:.2f}',flush=True)
    pa=s16>=0.15; pb=s14>=0.15; pc=reg['mlp16']<reg['attn14']
    out={'score_mlp16_frozen':s16,'score_attn14_frozen':s14,
         'regime_corr':reg,'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f'\nmlp16 frozen-regime score {s16:+.3f} | attn14 {s14:+.3f}')
    print(f"(a) mlp16 story vindicated (>=+0.15): {'HELD' if pa else 'FAILED'}")
    print(f"(b) attn14 survives frozen: {'HELD' if pb else 'FAILED'}")
    print(f"(c) regime shift largest at mlp16: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} and {PT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
