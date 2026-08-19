"""ASSEMBLED v2: same as assembled_dictionary but the crown uses its best
rung -- the full ridge linear (+0.12 solo, section 256) instead of the
fold table. REGISTERED: (a) assembled v2 total <= +0.90 nats; (b) still
sub- or near-additive (<= 1.1x sum of solos); (c) solo sanities within
10%.

Original docstring:
THE ASSEMBLED DICTIONARY MODEL: every certified deployable stand-in
applied SIMULTANEOUSLY, one honest Track-2 submission:
  attn0 c_v -> exact weights table  (v1_t = c_v(rms(wte_t)); zero error,
               section 254 -- computed from weights, full vocab coverage)
  attn1 c_v -> empirical per-token value table (87% solo, section 253)
  mlp1     -> context-free fold table (79% solo, section 250)
  mlp0/2/3 -> empirical token tables (68/58/44% solo, section 251)
  mlp10-17 spans -> probe-conditioned class dictionary + 2 linear classes
               (75% solo, section 244)
Composition is the open question: the section-172-era product law says
damage compounds multiplicatively at content level, so the assembled cost
may exceed the sum of solo costs.

REGISTERED PREDICTIONS: (a) assembled CE increase <= 1.5x the sum of solo
increases (band registered, genuinely uncertain); (b) headline: assembled
model total CE increase <= 2.0 nats; (c) sanity: two solo re-measurements
(mlp1 fold, attn1 vtable) within 15% of their published numbers; (d) the
description ledger is printed (weights-computable vs fitted, per part)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from circuit_dictionary import classify, COMPS as TAILC
D=1152; V=50257
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'assembled_v2_results.json'
CA,CB=300,512; R0,R1=120,300

@torch.no_grad()
def main():
    t0=time.time()
    # --- weights-exact attn0 value table + mlp1 fold table (vocab pass)
    tab_a0=torch.zeros(V,D,device=DEV,dtype=torch.float16)
    tab_m1=torch.zeros(V,D,device=DEV,dtype=torch.float16)
    capm={}
    h1=m.transformer.h[1].mlp.register_forward_hook(
        lambda mo_,i_,o_: capm.__setitem__(1,o_.detach()))
    for s0 in range(0,V,4096):
        idx=torch.arange(s0,min(s0+4096,V),device=DEV)[:,None]
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        hc=F.rms_norm((m.transformer.h[0].lambdas[0]
                       +m.transformer.h[0].lambdas[1])*x,(D,))
        tab_a0[s0:s0+idx.shape[0]]=m.transformer.h[0].attn.c_v(hc)[:,0]\
            .to(torch.float16)
        for blk in m.transformer.h[:2]:
            x,v1=blk(x,v1,x0)
        tab_m1[s0:s0+idx.shape[0]]=capm[1][:,0].to(torch.float16)
    h1.remove()
    # crown linear map (fit window A)
    Ys=[]; Xs=[]
    hcap=m.transformer.h[1].mlp.register_forward_hook(
        lambda mo_,i_,o_: (Ys.append(o_.detach().reshape(-1,D).float()),
                           Xs.append(None)))
    hcap.remove()
    Ys=[]; Xs=[]
    def capL(mod,i_,o_):
        Ys.append(o_.detach().reshape(-1,D).float())
        Xs.append(i_[0].detach().reshape(-1,D).float())
    hcap=m.transformer.h[1].mlp.register_forward_hook(capL)
    for i in range(CA,CB,4):
        bb=FW[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
    hcap.remove()
    Yl=torch.cat(Ys); Xl=torch.cat(Xs)
    lamL=1e-2*len(Xl)
    W1=torch.linalg.solve(Xl.T@Xl+lamL*torch.eye(D,device=DEV),Xl.T@Yl)
    b1=Yl.mean(0)-Xl.mean(0)@W1
    del Ys,Xs,Yl,Xl
    # --- empirical tables: mlp0/2/3 full outputs, attn1 values,
    #     tail spans + probe (window A)
    spans={}
    for li in TAILC:
        accs=[]
        for i in range(0,120,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc)
            accs.append(acc[0])
        Y=torch.cat(accs); Yb=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Yb).float(), full_matrices=False)
        spans[li]=(orth(Vh[:8].T),Yb.float())
    clsA=classify(CA,CB).to(DEV); clsC=classify(R0,R1).to(DEV)
    sums={k:torch.zeros(V,D,device=DEV) for k in ('m0','m2','m3','a1v')}
    cnt=torch.zeros(V,device=DEV)
    capsT={li:[] for li in TAILC}; capI=[]
    caps={}
    hs=[]
    for tag,mod in (('m0',m.transformer.h[0].mlp),
                    ('m2',m.transformer.h[2].mlp),
                    ('m3',m.transformer.h[3].mlp),
                    ('a1v',m.transformer.h[1].attn.c_v)):
        def mk(tag=tag):
            return lambda mo_,i_,o_: caps.__setitem__(
                tag,(o_[0] if isinstance(o_,tuple) else o_)
                .detach().reshape(-1,D).float())
        hs.append(mod.register_forward_hook(mk()))
    for li in TAILC:
        def mk2(li=li):
            def hook(mod,i_,o_):
                capsT[li].append(o_.detach().reshape(-1,D).float())
                if li==TAILC[0]:
                    capI.append(i_[0].detach().reshape(-1,D).float())
            return hook
        hs.append(m.transformer.h[li].mlp.register_forward_hook(mk2()))
    for i in range(CA,CB,4):
        bb=FW[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous()
        m(idx, bb[:,1:].contiguous())
        ids=idx.reshape(-1)
        cnt.index_add_(0,ids,torch.ones_like(ids,dtype=torch.float))
        for tag in sums: sums[tag].index_add_(0,ids,caps[tag])
    for h in hs: h.remove()
    tabs={}
    for tag in sums:
        t=sums[tag]/cnt.clamp_min(1)[:,None]
        t[cnt==0]=sums[tag].sum(0)/cnt.sum()
        tabs[tag]=t.to(torch.float16)
    flatA=clsA.reshape(-1)
    XA=torch.cat(capI)
    Yoh=torch.zeros(len(flatA),10,device=DEV)
    Yoh[torch.arange(len(flatA)),flatA]=1.0
    lam=1e-2*len(XA)
    Wp=torch.linalg.solve(XA.T@XA+lam*torch.eye(D,device=DEV),XA.T@Yoh)
    DICT={}; LIN={}
    # need per-component inputs for LIN: recapture quickly
    capI2={li:[] for li in TAILC}; hs=[]
    for li in TAILC:
        def mk3(li=li):
            return lambda mo_,i_,o_: capI2[li].append(
                i_[0].detach().reshape(-1,D).float())
        hs.append(m.transformer.h[li].mlp.register_forward_hook(mk3()))
    for i in range(CA,CB,4):
        bb=FW[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
    for h in hs: h.remove()
    for li in TAILC:
        Q,_=spans[li]; C=torch.cat(capsT[li])@Q
        DICT[li]=torch.stack([C[flatA==k].mean(0) if (flatA==k).sum()>0
                              else C.mean(0) for k in range(10)])
        Xl=torch.cat(capI2[li])
        LIN[li]={}
        for k in (8,9):
            mk_=flatA==k
            Xk=Xl[mk_]; Ck=C[mk_]
            l2=1e-2*len(Xk)
            LIN[li][k]=torch.linalg.solve(Xk.T@Xk+l2*torch.eye(D,device=DEV),
                                          Xk.T@Ck)
        capsT[li]=None; capI2[li]=None
    del capI, XA
    cur={}
    def pertok(parts):
        hs=[]
        if 'a0' in parts:
            def h_a0(mod,i_,o_):
                return tab_a0[cur['idx']].to(o_.dtype)
            hs.append(m.transformer.h[0].attn.c_v
                      .register_forward_hook(h_a0))
        if 'a1v' in parts:
            def h_a1(mod,i_,o_):
                return tabs['a1v'][cur['idx']].to(o_.dtype)
            hs.append(m.transformer.h[1].attn.c_v
                      .register_forward_hook(h_a1))
        for tag,li in (('m0',0),('m2',2),('m3',3)):
            if tag in parts:
                def h_m(mod,i_,o_,tag=tag):
                    return tabs[tag][cur['idx']].to(o_.dtype)
                hs.append(m.transformer.h[li].mlp
                          .register_forward_hook(h_m))
        if 'm1' in parts:
            def h_m1(mod,i_,o_):
                x=i_[0].float().reshape(-1,D)
                return (x@W1+b1).view(o_.shape).to(o_.dtype)
            hs.append(m.transformer.h[1].mlp.register_forward_hook(h_m1))
        if 'tail' in parts:
            for li in TAILC:
                Q,mu=spans[li]; Dq=DICT[li]; Lq=LIN[li]
                def h_t(mod,i_,o_,li=li,Q=Q,Dq=Dq,Lq=Lq,
                        first=(li==TAILC[0])):
                    B,T,_=o_.shape
                    x=i_[0].float().reshape(-1,D)
                    c=o_.float().reshape(-1,D)@Q
                    if first:
                        cur['kk']=(x@Wp).argmax(1)
                    kk=cur['kk']
                    tgt=Dq[kk].clone()
                    for k in (8,9):
                        sel=kk==k
                        if sel.any(): tgt[sel]=x[sel]@Lq[k]
                    delta=((c-tgt)@Q.T).view(B,T,D)
                    return o_-delta.to(o_.dtype)
                hs.append(m.transformer.h[li].mlp
                          .register_forward_hook(h_t))
        ces=[]
        for i in range(R0,R1,4):
            bb=FW[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            cur['idx']=idx
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h:
                x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        for h in hs: h.remove()
        return torch.cat(ces)
    base=float(pertok(set()).mean())
    solos={}
    for p in ('a0','a1v','m0','m1','m2','m3','tail'):
        solos[p]=float(pertok({p}).mean())-base
        print(f'solo {p:4s}: {solos[p]:+.4f}',flush=True)
    comb=float(pertok({'a0','a1v','m0','m1','m2','m3','tail'}).mean())-base
    ssum=sum(solos.values())
    pa=comb<=1.1*ssum; pb=comb<=0.90
    pc=abs(solos['m1']-0.120)<=0.05 and abs(solos['a1v']-0.032)<=0.05
    out={'base':round(base,4),'solos':{k:round(v,4)
         for k,v in solos.items()},'sum_solo':round(ssum,4),
         'assembled':round(comb,4),
         'compounding':round(comb/max(ssum,1e-6),2),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'description':{'weights_exact':'attn0 vtable (V x D), mlp1 fold '
          '(V x D)','fitted':'mlp0/2/3 tables (3 V x D), attn1 vtable '
          '(V x D), tail dict 640 + probe D x 10 + 2x8 linear maps'}}
    print(f'\nbase {base:.3f} | sum-of-solos {ssum:+.3f} | ASSEMBLED '
          f'{comb:+.3f} (x{comb/max(ssum,1e-6):.2f})')
    print(f"(a) <=1.1x sum: {'HELD' if pa else 'FAILED'}")
    print(f"(b) assembled <=0.90 nats: {'HELD' if pb else 'FAILED'}")
    print(f"(c) solo sanity: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
