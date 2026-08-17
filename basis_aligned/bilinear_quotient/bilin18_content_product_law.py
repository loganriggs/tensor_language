"""Content-level product law. The composition law (excess ~ c*d16*d17, c ~ 23)
was fit on RAW damages, which section 117 showed are norm-inflated at the loud
end (L17 82%). Re-fit at content level: 3x3 grid of span sizes (k in 2,8,32 at
L16 and L17), per-arm frozen final gain, content-level excess per cell.
REGISTERED: (a) the product law survives at content level: regressing excess on
d16*d17 gives R^2 >= 0.7 across the 9 cells; (b) the constant changes by >= 2x
from the raw-damage fit on the same cells (the old c was a norm-channel
artifact); (c) both raw and content grids are monotone in both k's.

Prior context -- replication before interpretation: section 121 found deleting L9's PCA-8 span
(the strongest deletion-improves regularizer span) TOGETHER with L16's span (the
model's biggest content span) is net negative: joint delta -0.024 on rows
300-348. Replicate on disjoint rows 352-448 with a fresh random-pair control.

REGISTERED PREDICTIONS: (a) the joint L9+L16 delta stays negative on disjoint
rows; (b) it is below the sum of the individual deltas by >= 0.01 (genuine
beneficial interaction, not just L9's own negativity); control (c): L9 + a
random-8 span at L16 shows joint ~= sum (|excess| <= 0.005)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_content_product_law_results.json')

@torch.no_grad()
def ce_eval(patches):
    hs=[]
    for li,(Q,cbar) in patches.items():
        def mk(Q=Q,cbar=cbar):
            def hook(mod,i_,o_):
                c=o_.float()@Q
                return (o_-((c-cbar)@Q.T).to(o_.dtype))
            return hook
        hs.append(m.transformer.h[li].mlp.register_forward_hook(mk()))
    tot,n=0.0,0
    for i in range(352,448,4):
        b=FW[i:i+4,:257].to(DEV)
        loss=m(b[:,:-1].contiguous(), b[:,1:].contiguous())
        ntok=(b.shape[1]-1)*b.shape[0]
        tot+=float(loss)*ntok; n+=ntok
    for h in hs: h.remove()
    return tot/n

@torch.no_grad()
def main():
    t0=time.time()
    from bilin18_content_profiles import fwd_arm as _unused  # noqa
    import bilin18_content_profiles as CP
    Vs={}
    for li in (16,17):
        accs=[]
        for i in range(0,36,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc); accs.append(acc[0])
        Y=torch.cat(accs); Ybar=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Ybar).float(), full_matrices=False)
        Vs[li]=(Vh,Ybar)
    def spank(li,k):
        Vh,Ybar=Vs[li]; Q=orth(Vh[:k].T); return (Q,Ybar@Q)
    def ce2(d16=None,d17=None,freeze=False):
        # dual forward with up to two spans, per-arm frozen gain
        import torch.nn.functional as F2
        from tier2_model import rope_tables, apply_rot
        tot,n=0.0,0
        for i in range(300,364,4):
            bb=FW[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B,T=idx.shape
            cos,sin=rope_tables(T,128,DEV,torch.float32,'bf16')
            cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
            mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
            def run(dmg):
                x=F2.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
                for li in range(18):
                    blk=m.transformer.h[li]; a=blk.attn; mlp=blk.mlp
                    x=blk.lambdas[0]*x+blk.lambdas[1]*x0
                    hcur=F2.rms_norm(x,(D,))
                    def qk(l):
                        z=F2.rms_norm(l(hcur).view(B,T,9,128),(128,))
                        return apply_rot(z,cosb,sinb)
                    v=a.c_v(hcur).view(B,T,9,128)
                    nonlocal_v1=v1
                    v1n=v if v1 is None else v1
                    v=(1-a.lamb)*v+a.lamb*v1n.view_as(v)
                    q,k1_,q2,k2=qk(a.c_q),qk(a.c_k),qk(a.c_q2),qk(a.c_k2)
                    s1=torch.einsum('bqhd,bkhd->bhqk',q,k1_)/128
                    s2=torch.einsum('bqhd,bkhd->bhqk',q2,k2)/128
                    pat=(s1*s2).masked_fill(~mask,0.0)
                    x=x+a.c_proj(torch.einsum('bhqk,bkhd->bqhd',pat,v).reshape(B,T,-1))
                    xhat=F2.rms_norm(x,(D,))
                    mo=mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
                    if dmg.get(li) is not None:
                        Q,cbar=dmg[li]
                        c=mo.float().reshape(-1,D)@Q
                        mo=mo-((c-cbar)@Q.T).to(mo.dtype).view_as(mo)
                    x=x+mo
                    v1=v1n
                return x
            dmg={}
            if d16 is not None: dmg[16]=d16
            if d17 is not None: dmg[17]=d17
            xh=run(dmg)
            if freeze:
                xc=run({})
                rms_c=xc.float().pow(2).mean(-1,keepdim=True).sqrt()
                xn=(xh.float()/rms_c.clamp_min(1e-8)).to(xh.dtype)
            else:
                xn=F2.rms_norm(xh,(D,))
            lg=(30*torch.tanh(m.lm_head(xn)/30)).float()
            c=F2.cross_entropy(lg.view(-1,lg.size(-1)),tg)
            tot+=float(c)*tg.numel(); n+=tg.numel()
        return tot/n
    out={'cells':{}}
    for freeze,tag in ((False,'raw'),(True,'content')):
        base=ce2(freeze=freeze)
        d16s={k: ce2(d16=spank(16,k),freeze=freeze)-base for k in (2,8,32)}
        d17s={k: ce2(d17=spank(17,k),freeze=freeze)-base for k in (2,8,32)}
        xs=[];ys=[]
        for k1 in (2,8,32):
            for k2 in (2,8,32):
                j=ce2(d16=spank(16,k1),d17=spank(17,k2),freeze=freeze)-base
                exc=j-d16s[k1]-d17s[k2]
                out['cells'][f'{tag}_{k1}_{k2}']={'d16':d16s[k1],'d17':d17s[k2],
                                                  'excess':exc}
                xs.append(d16s[k1]*d17s[k2]); ys.append(exc)
                print(f'{tag} k16={k1:2d} k17={k2:2d}: d16 {d16s[k1]:+.4f} '
                      f'd17 {d17s[k2]:+.4f} excess {exc:+.4f}',flush=True)
        X=torch.tensor(xs); Yv=torch.tensor(ys)
        c=float((X*Yv).sum()/(X*X).sum())
        r2=1-float(((Yv-c*X)**2).sum()/((Yv-Yv.mean())**2).sum())
        out[tag+'_c']=c; out[tag+'_r2']=r2
        print(f'{tag}: product-law c={c:.1f}, R^2={r2:.2f}\n',flush=True)
    pa=out['content_r2']>=0.7
    pb=abs(out['content_c'])>=2*abs(out['raw_c']) or        abs(out['raw_c'])>=2*abs(out['content_c'])
    out['pred_a']=bool(pa); out['pred_b']=bool(pb)
    print(f"(a) product law survives at content level (R^2>=0.7): "
          f"{'HELD' if pa else 'FAILED'} ({out['content_r2']:.2f})")
    print(f"(b) constant changes >=2x: {'HELD' if pb else 'FAILED'} "
          f"(raw {out['raw_c']:.1f} vs content {out['content_c']:.1f})")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
