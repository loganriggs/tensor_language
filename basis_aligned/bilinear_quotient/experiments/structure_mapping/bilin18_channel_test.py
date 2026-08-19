"""Is the type separation an end-to-end CHANNEL or only local? Sections 131-132
showed attention and MLP at each layer read and write different stream
directions. Untested (user caught it): does downstream attention preferentially
read what upstream ATTENTION wrote? For reader layers lj in (6,10,14): overlap
of lj's attention watch-list (top-8 stacked score filters) with (i) the top-8
write directions of upstream attention (pooled layers lj-3..lj-1), (ii) upstream
MLP writes, same pooling. Same for lj's MLP watch-list.

REGISTERED PREDICTIONS: (a) attention watch-lists align more with upstream
attention writes than upstream MLP writes at >= 2/3 reader layers (a true
channel); (b) symmetric for MLP watch-lists vs MLP writes; alternative to
either: types separate locally but read the common mix -- 'channel' would be
corrected to 'aperture' in the report. Null (c): covariance-matched random
subspace pairs give alignment below both."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from tier2_model import rope_tables, apply_rot
NH,HD,D=9,128,1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_channel_test_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    ats={li:[] for li in range(3,15)}; mos={li:[] for li in range(3,15)}
    hcs={lj:[] for lj in (6,10,14)}; mins={lj:[] for lj in (6,10,14)}
    for i in range(0,24,6):
        idx=FW[i:i+6,:257].to(DEV)
        B,T=idx.shape
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16')
        cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
        mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
        for li in range(15):
            blk=m.transformer.h[li]; a=blk.attn
            x=blk.lambdas[0]*x+blk.lambdas[1]*x0
            hcur=F.rms_norm(x,(D,))
            if li in hcs: hcs[li].append(hcur.detach().reshape(-1,D).float())
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
            if li in ats: ats[li].append(att.detach().reshape(-1,D).float())
            x=x+att
            xin_mlp=x
            if li in mins: mins[li].append(xin_mlp.detach().reshape(-1,D).float())
            xhat=F.rms_norm(x,(D,)); mlp=blk.mlp
            mo=mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
            if li in mos: mos[li].append(mo.detach().reshape(-1,D).float())
            x=x+mo
    def top8(lst):
        X=torch.cat(lst); Xc=X-X.mean(0)
        _,_,Vh=torch.linalg.svd(Xc[:20000],full_matrices=False)
        return orth(Vh[:8].T)
    g=torch.Generator(device=DEV).manual_seed(0)
    res={}; winsA=0; winsM=0; nullmax=0
    for lj in (6,10,14):
        X=torch.cat(hcs[lj]); Xc=X-X.mean(0)
        C=Xc.T@Xc/Xc.shape[0]
        ev,U=torch.linalg.eigh(C.double())
        Ch=((U*ev.clamp_min(0).sqrt())@U.T).float()
        a=m.transformer.h[lj].attn
        mats=[]
        for h_ in range(NH):
            for wq,wk in ((a.c_q,a.c_k),(a.c_q2,a.c_k2)):
                Wq=wq.weight.detach().float().view(NH,HD,D)[h_]
                Wk=wk.weight.detach().float().view(NH,HD,D)[h_]
                Uk,S,Vk=torch.linalg.svd(Ch@Wq.T@Wk@Ch)
                mats.append(torch.cat([Uk[:,:4],Vk[:4].T],dim=1))
        Sf=torch.cat(mats,dim=1)
        Ua,_,_=torch.linalg.svd(Sf@Sf.T)
        AW=orth(Ua[:,:8])
        Xm=torch.cat(mins[lj]); S=Xm.T@Xm/Xm.shape[0]
        mlp=m.transformer.h[lj].mlp
        L=mlp.Left.weight.detach().float(); R=mlp.Right.weight.detach().float()
        Dw=mlp.Down.weight.detach().float(); DD=Dw.T@Dw
        G=L.T@(DD*(R@S@R.T))@L + R.T@(DD*(L@S@L.T))@R
        evg,Ug=torch.linalg.eigh(G.double())
        MW=orth(Ug[:,evg.argsort(descending=True)[:8]].float())
        upA=top8(sum([ats[li] for li in (lj-3,lj-2,lj-1)],[]))
        upM=top8(sum([mos[li] for li in (lj-3,lj-2,lj-1)],[]))
        def medcos(A,B):
            s_=torch.linalg.svdvals(A.T@B)
            return float(sorted(s_.tolist())[4])
        row={'attnwatch_vs_attnwrite':medcos(AW,upA),
             'attnwatch_vs_mlpwrite':medcos(AW,upM),
             'mlpwatch_vs_mlpwrite':medcos(MW,upM),
             'mlpwatch_vs_attnwrite':medcos(MW,upA)}
        R1=orth(Ch@torch.randn(D,8,device=DEV,generator=g))
        R2=orth(Ch@torch.randn(D,8,device=DEV,generator=g))
        row['null']=medcos(R1,R2)
        nullmax=max(nullmax,row['null'])
        res[lj]=row
        if row['attnwatch_vs_attnwrite']>row['attnwatch_vs_mlpwrite']: winsA+=1
        if row['mlpwatch_vs_mlpwrite']>row['mlpwatch_vs_attnwrite']: winsM+=1
        print(f"L{lj}: Awatch~Awrite {row['attnwatch_vs_attnwrite']:.2f} vs "
              f"Awatch~Mwrite {row['attnwatch_vs_mlpwrite']:.2f} | "
              f"Mwatch~Mwrite {row['mlpwatch_vs_mlpwrite']:.2f} vs "
              f"Mwatch~Awrite {row['mlpwatch_vs_attnwrite']:.2f} | "
              f"null {row['null']:.2f}",flush=True)
    pa=winsA>=2; pb=winsM>=2
    out={'per_layer':{str(k):v for k,v in res.items()},
         'pred_a_attn_channel':bool(pa),'pred_b_mlp_channel':bool(pb)}
    print(f"\n(a) attention reads attention (>=2/3): {'HELD' if pa else 'FAILED'} ({winsA}/3)")
    print(f"(b) mlp reads mlp (>=2/3): {'HELD' if pb else 'FAILED'} ({winsM}/3)")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
