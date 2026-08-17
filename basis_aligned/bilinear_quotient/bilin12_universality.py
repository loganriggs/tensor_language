"""UNIVERSALITY: do bilin18's structural laws hold in its sibling checkpoint
bilin12 (same bilinear-sqrd-attn family, 12 layers, 6 heads, 768 dims)? Three
load-bearing structures, cheapest instruments:

REGISTERED PREDICTIONS (grounded in bilin18's values):
(a) DILUTION: MLP write-to-stream ratios decline monotonically through the tail
    (layers 3-10, <= 1 inversion), as in bilin18 (0 inversions).
(b) FRONT-LOADED NONLINEARITY: linearization cost (consistent in-forward
    protocol) peaks in L0-L2 and the peak is >= 2x the median of mid layers
    (bilin18: L1 +0.28 vs mid ~0.03).
(c) MATCHED FILTERS: median on-distribution score eff-rank <= 12 of 128, and
    row-shuffled weights >= 2x the trained median (bilin18: 4.3 vs 18.7).
Also logged, no bar: the lambda table (bilin18 re-injects embeddings at 8.0).
Any failure = a bilin18-specific result, itself a finding."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import FW, DEV
from tier2_model import load_elriggs, rope_tables, apply_rot
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin12_universality_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    m2,cfg=load_elriggs('bilin12', device=DEV)
    NL=len(m2.transformer.h); D=m2.transformer.wte.weight.shape[1]
    NH=cfg.get('n_head',6); HD=D//NH
    print(f'bilin12 loaded: {NL} layers, {NH} heads, d={D}, hd={HD}',flush=True)
    lam_tab=[(float(b.lambdas[0]),float(b.lambdas[1])) for b in m2.transformer.h]
    print('lambdas (l0,l1):',' '.join(f'{a:.2f}/{b:.2f}' for a,b in lam_tab),flush=True)
    def fwd_all(idx):
        B,T=idx.shape
        x=F.rms_norm(m2.transformer.wte(idx),(D,)); x0=x; v1=None
        cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16')
        cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
        mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
        ins={};mos={};hcs={}
        for li in range(NL):
            blk=m2.transformer.h[li]; a=blk.attn
            x=blk.lambdas[0]*x+blk.lambdas[1]*x0
            ins[li]=x.detach().reshape(-1,D).float()
            hcur=F.rms_norm(x,(D,))
            hcs[li]=hcur.detach().reshape(-1,D).float()
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
            mo=mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
            mos[li]=mo.detach().reshape(-1,D).float()
            x=x+mo
        lg=m2.lm_head(F.rms_norm(x,(D,)))
        return ins,mos,hcs,(30*torch.tanh(lg/30)).float()
    # collect stats
    tri=[]
    for i in range(0,36,6): tri.append(fwd_all(FW[i:i+6,:257].to(DEV)))
    # (a) dilution ratios
    ratios={}
    for li in range(3,NL-1):
        Y=torch.cat([t[1][li] for t in tri]); X=torch.cat([t[0][li+1] for t in tri])
        ratios[li]=float((Y-Y.mean(0)).pow(2).sum(1).mean())/ \
                   float((X-X.mean(0)).pow(2).sum(1).mean())
    seq=[ratios[li] for li in sorted(ratios)]
    inv=sum(1 for i in range(len(seq)-1) if seq[i+1]>seq[i])
    pa=inv<=1
    print('dilution ratios: '+' '.join(f'{r:.3f}' for r in seq)
          +f' | inversions {inv} -> (a) {"HELD" if pa else "FAILED"}',flush=True)
    # (b) linearization costs L0,1,2,5,8 (in-forward stand-in)
    def fit(li):
        X=torch.cat([t[0][li] for t in tri]); Y=torch.cat([t[1][li] for t in tri])
        bx=X.mean(0); by=Y.mean(0)
        Xc=X-bx; Yc=Y-by
        lam=1e-2*float((Xc**2).mean())*Xc.shape[1]/Xc.shape[0]
        W=torch.linalg.solve(Xc.T@Xc/Xc.shape[0]+lam*torch.eye(D,device=DEV),
                             Xc.T@Yc/Xc.shape[0])
        return {'W':W,'bx':bx,'by':by}
    LINS={}
    def fwd_ce(idx,tg):
        B,T=idx.shape
        x=F.rms_norm(m2.transformer.wte(idx),(D,)); x0=x; v1=None
        cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16')
        cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
        mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
        for li in range(NL):
            blk=m2.transformer.h[li]; a=blk.attn
            x=blk.lambdas[0]*x+blk.lambdas[1]*x0
            xin=x
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
            if li in LINS:
                mp=LINS[li]
                xi=xin.reshape(-1,D).float()
                mo=((xi-mp['bx'])@mp['W']+mp['by']).to(x.dtype).view_as(x)
            else:
                mo=mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
            x=x+mo
        lg=(30*torch.tanh(m2.lm_head(F.rms_norm(x,(D,)))/30)).float()
        return float(F.cross_entropy(lg.view(-1,lg.size(-1)),tg))
    def ce():
        tot,n=0.0,0
        for i in range(300,348,4):
            b=FW[i:i+4,:257].to(DEV)
            c=fwd_ce(b[:,:-1].contiguous(), b[:,1:].reshape(-1))
            tot+=c*(b.shape[1]-1)*b.shape[0]; n+=(b.shape[1]-1)*b.shape[0]
        return tot/n
    base=ce()
    costs={}
    for li in (0,1,2,5,8):
        LINS={li:fit(li)}
        costs[li]=ce()-base
        LINS={}
        print(f'L{li}: linearization cost +{costs[li]:.4f}',flush=True)
    peak=max(costs[li] for li in (0,1,2))
    midm=sorted([costs[5],costs[8]])[0]
    midmed=(costs[5]+costs[8])/2
    pb=peak==max(costs.values()) and peak>=2*midmed
    print(f'(b) front peak {peak:.3f} vs mid {midmed:.3f} -> '
          f'{"HELD" if pb else "FAILED"}',flush=True)
    # (c) score ranks, trained vs shuffled
    g=torch.Generator(device=DEV).manual_seed(0)
    def rankset(shuffle):
        rs=[]
        for li in (1,5,9):
            X=torch.cat([t[2][li] for t in tri]); Xc=X-X.mean(0)
            C=Xc.T@Xc/Xc.shape[0]
            ev,U=torch.linalg.eigh(C.double())
            Ch=((U*ev.clamp_min(0).sqrt())@U.T).float()
            a=m2.transformer.h[li].attn
            for h_ in range(NH):
                for wq,wk in ((a.c_q,a.c_k),(a.c_q2,a.c_k2)):
                    Wq=wq.weight.detach().float().view(NH,HD,D)[h_]
                    Wk=wk.weight.detach().float().view(NH,HD,D)[h_]
                    if shuffle:
                        Wq=Wq[:,torch.randperm(D,generator=g,device=DEV)]
                        Wk=Wk[:,torch.randperm(D,generator=g,device=DEV)]
                    sv=torch.linalg.svdvals(Ch@Wq.T@Wk@Ch); e=sv**2
                    rs.append(float(e.sum()**2/(e**2).sum()))
        return sorted(rs)[len(rs)//2]
    mr=rankset(False); msh=rankset(True)
    pc=mr<=12 and msh>=2*mr
    print(f'(c) score-rank {mr:.1f} (shuffled {msh:.1f}) -> '
          f'{"HELD" if pc else "FAILED"}',flush=True)
    out={'lambdas':lam_tab,'dilution':ratios,'inversions':inv,'costs':costs,
         'score_rank':mr,'shuffled':msh,'pa':bool(pa),'pb':bool(pb),
         'pc':bool(pc),'base_ce':base}
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
