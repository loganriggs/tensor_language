"""MODE DICTIONARY -- the census feeds the benchmark (user loop): do the
data-dual's modes make BETTER dictionary classes than the hand
taxonomy? Fit the tail-MLP span dictionary (10-17, top-8 spans, joint
replacement -- the 241 instrument) with three label sets:
  BASE     -- the 10 hand classes;
  +MODES   -- 10 classes plus 16 root-mode override labels (a token
              gets mode-k's label when its causal damage profile
              projects onto mode k above the fit-window 85th
              percentile; oracle-conditioned, like the oracle
              dictionary arm);
  SHUFFLE  -- +MODES with mode labels permuted (control).
Mode scores on the eval window are computed causally: the same 108
probes' per-token damages, projected on the fit-window mode loadings.
REGISTERED PREDICTIONS:
  (a) +MODES joint recovery >= BASE + 5 points (absolute);
  (b) SHUFFLE gains <= 1 point over BASE;
  (c) per-mode contribution table reported."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from circuit_dictionary import classify, COMPS as TAILC
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'mode_dictionary_results.json'
CA,CB=300,512; R0,R1=120,300
MHL=list(range(2,10)); T=256

@torch.no_grad()
def main():
    t0=time.time()
    def ce_vec(rows,nb,hooks):
        ces=[]
        for i in range(0,nb*4,4):
            bb=rows[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h:
                x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        for h in hooks: h.remove()
        return torch.cat(ces)
    rowsA=FW[CA:CB]; nA=(CB-CA)//4
    rowsC=FW[R0:R1]; nC=(R1-R0)//4
    baseA=ce_vec(rowsA,nA,[]); baseC=ce_vec(rowsC,nC,[])
    # probes
    sums={}; hs=[]
    for li in range(18):
        for kind,mod in (('a',m.transformer.h[li].attn),
                         ('m',m.transformer.h[li].mlp)):
            key=f'{kind}{li}'; sums[key]=torch.zeros(D,device=DEV)
            def mk(key=key):
                def h(mo,i_,o_):
                    y=o_[0] if isinstance(o_,tuple) else o_
                    sums[key]+=y.detach().float().reshape(-1,D).sum(0)
                return h
            hs.append(mod.register_forward_hook(mk()))
    for i in range(0,nA*4,4):
        bb=rowsA[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
    for h in hs: h.remove()
    mus={k:v/(nA*4*256) for k,v in sums.items()}
    MODS={f'a{li}':m.transformer.h[li].attn for li in range(18)}
    MODS.update({f'm{li}':m.transformer.h[li].mlp for li in range(18)})
    def comp_probe(key):
        mu=mus[key]; mod=MODS[key]
        if key[0]=='a':
            def fh(mo,i_,o_,mu=mu):
                y,v1=o_
                return (mu.expand_as(y).to(y.dtype),v1)
        else:
            def fh(mo,i_,o_,mu=mu):
                return mu.expand_as(o_).to(o_.dtype)
        return [mod.register_forward_hook(fh)]
    mod2=sys.modules[type(m.transformer.h[0].attn).__module__]
    are=mod2.apply_rotary_emb
    def head_probe(li,hd):
        at=m.transformer.h[li].attn
        def fh(mo_,args,out,at=at,hd=hd):
            y,v1r=out
            X=args[0]; v1=args[1] if args[1] is not None else v1r
            B=X.shape[0]
            v=at.c_v(X).view(B,T,9,128)
            vm=(1-at.lamb)*v+at.lamb*v1.view_as(v)
            cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
            qf=F.rms_norm(at.c_q(X).view(B,T,9,128),(128,))
            kf=F.rms_norm(at.c_k(X).view(B,T,9,128),(128,))
            qf,kf=are(qf,cos,sin),are(kf,cos,sin)
            q2f=F.rms_norm(at.c_q2(X).view(B,T,9,128),(128,))
            k2f=F.rms_norm(at.c_k2(X).view(B,T,9,128),(128,))
            q2f,k2f=are(q2f,cos,sin),are(k2f,cos,sin)
            sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),kf.float())/128
            sc2=torch.einsum('bqhd,bkhd->bhqk',q2f.float(),
                             k2f.float())/128
            pat=(sc*sc2)*torch.tril(torch.ones(T,T,device=DEV))
            z=torch.einsum('bhqk,bkhd->bhqd',pat,vm.float())
            z[:,hd]=0
            ynew=at.c_proj(z.transpose(1,2).contiguous()
                           .view(B,T,-1).to(X.dtype))
            return (ynew,v1r)
        return [at.register_forward_hook(fh)]
    P0=[('comp',f'{k}{li}') for li in range(18) for k in ('a','m')]
    P0+=[('head',li,hd) for li in MHL for hd in range(9)]
    colsA=[]; colsC=[]
    for j,ps in enumerate(P0):
        mk_=lambda: (comp_probe(ps[1]) if ps[0]=='comp'
                     else head_probe(ps[1],ps[2]))
        colsA.append((ce_vec(rowsA,nA,mk_())-baseA).cpu())
        colsC.append((ce_vec(rowsC,nC,mk_())-baseC).cpu())
        if j%20==0: print(f'probe {j}/{len(P0)}',flush=True)
    MA=torch.stack(colsA,1); MC=torch.stack(colsC,1)
    muA=MA.mean(0,keepdim=True); sdA=MA.std(0,keepdim=True).clamp_min(1e-6)
    MA=torch.clamp((MA-muA)/sdA,-3,3)
    MC=torch.clamp((MC-muA)/sdA,-3,3)
    def safe_svd(X):
        try: return torch.linalg.svd(X,full_matrices=False)
        except Exception:
            U,S,Vh=torch.linalg.svd(X.double(),full_matrices=False)
            return U.float(),S.float(),Vh.float()
    U,Sg,Vh=safe_svd(MA)
    scA=U[:,:16]*Sg[:16]; scC=MC@Vh[:16].T
    thr=scA.abs().quantile(0.85,dim=0)
    def labels(sc,base_cls):
        lab=base_cls.clone()
        strength=sc.abs()/thr
        best=strength.argmax(1)
        hit=strength.max(1).values>=1.0
        lab[hit]=10+best[hit]
        return lab
    clsA=classify(CA,CB).reshape(-1)
    clsC=classify(R0,R1).reshape(-1)
    labA=labels(scA,clsA); labC=labels(scC,clsC)
    g=torch.Generator().manual_seed(0)
    perm=torch.randperm(16,generator=g)
    labAs=clsA.clone(); labCs=clsC.clone()
    hitA=labA>=10; hitC=labC>=10
    labAs[hitA]=10+perm[labA[hitA]-10]; labCs[hitC]=10+perm[labC[hitC]-10]
    print(f'mode-label coverage: fit {float(hitA.float().mean()):.2%} '
          f'eval {float(hitC.float().mean()):.2%}',flush=True)
    # tail span dictionary instrument (241-style)
    spans={}
    for li in TAILC:
        accs=[]
        for i in range(0,120,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc)
            accs.append(acc[0])
        Y=torch.cat(accs)
        _,_,Vh8=torch.linalg.svd((Y-Y.mean(0)).float(),
                                 full_matrices=False)
        spans[li]=orth(Vh8[:8].T)
    # capture tail comp outputs on both windows (clean)
    def cap_tail(rows,nb):
        caps={li:[] for li in TAILC}
        hs=[]
        for li in TAILC:
            def mk2(li=li):
                def h(mo,i_,o_):
                    caps[li].append(o_.detach().float().reshape(-1,D))
                return h
            hs.append(m.transformer.h[li].mlp.register_forward_hook(mk2()))
        for i in range(0,nb*4,4):
            bb=rows[i:i+4,:257].to(DEV)
            m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
        for h in hs: h.remove()
        return {li:torch.cat(v) for li,v in caps.items()}
    tailA=cap_tail(rowsA,nA)
    def run_dict(labA_,labC_,nlab):
        DICT={}
        for li in TAILC:
            C=tailA[li]@spans[li]
            DICT[li]=torch.stack([C[labA_==k].mean(0) if
                (labA_==k).sum()>2 else C.mean(0) for k in range(nlab)])
        cur={}
        hs=[]
        for li in TAILC:
            Q=spans[li]
            def h(mo,i_,o_,li=li,Q=Q):
                B,T2,_=o_.shape
                c=o_.float().reshape(-1,D)@Q
                tgt=DICT[li][cur['lab']]
                delta=((c-tgt)@Q.T).view(B,T2,D)
                return o_-delta.to(o_.dtype)
            hs.append(m.transformer.h[li].mlp.register_forward_hook(h))
        ces=[]
        for i in range(0,nC*4,4):
            bb=rowsC[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            cur['lab']=labC_[i*256:(i+4)*256].to(DEV)
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h:
                x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        for h in hs: h.remove()
        return float(torch.cat(ces).mean())
    # joint span ablation reference
    def run_abl():
        hs=[]
        for li in TAILC:
            Q=spans[li]
            mu=(tailA[li]@Q).mean(0)
            def h(mo,i_,o_,Q=Q,mu=mu):
                B,T2,_=o_.shape
                c=o_.float().reshape(-1,D)@Q
                delta=((c-mu)@Q.T).view(B,T2,D)
                return o_-delta.to(o_.dtype)
            hs.append(m.transformer.h[li].mlp.register_forward_hook(h))
        return float(ce_vec(rowsC,nC,hs).mean())
    b=float(baseC.mean())
    abl=run_abl()-b
    d10=run_dict(clsA,clsC,10)-b
    dm=run_dict(labA,labC,26)-b
    dsh=run_dict(labAs,labCs,26)-b
    rec=lambda d: 1-d/abl
    print(f'ablation {abl:+.4f} | 10-class {d10:+.4f} (rec {rec(d10):.1%}) '
          f'| +modes {dm:+.4f} (rec {rec(dm):.1%}) | shuffle {dsh:+.4f} '
          f'(rec {rec(dsh):.1%})')
    pa=rec(dm)>=rec(d10)+0.05
    pb=rec(dsh)<=rec(d10)+0.01
    out={'ablation':round(abl,4),'rec_10':round(rec(d10),4),
         'rec_modes':round(rec(dm),4),'rec_shuffle':round(rec(dsh),4),
         'coverage_eval':round(float(hitC.float().mean()),4),
         'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f"(a) +modes >= +5pts: {'HELD' if pa else 'FAILED'}")
    print(f"(b) shuffle <= +1pt: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
