"""Dictionary rung 1: per-class LINEAR maps for exactly the two classes
where constants failed (induction -9%, other 26%), keeping constants for
the eight classes where they work. For each tail component (mlp10-17) and
each of {ind, other}: an 8-dim ridge map from the component's normalized
input to its span coefficients, fit on window A class sites, applied on
window C. Honest cost: constants 640 numbers + 2 classes x 8 comps x
(1152x8) = ~147K params for the linear part -- priced against what it buys.

REGISTERED PREDICTIONS: (a) total recovery rises from 50% to >= 65%;
(b) induction-class recovery rises from -9% to >= 40% (input-linear maps
can read the local stream, which carries the copied token's identity);
(c) the marginal-value ordering is reported: nats recovered per 100K
params for the linear rung vs the constant rung (the constant rung should
dominate -- semantics is the cheap information)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from circuit_dictionary import classify, COMPS, CLS
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'circuit_dictionary_rung1_results.json'
CA,CB=300,512; R0,R1=120,300
LINCLS=[8,9]   # ind, other

@torch.no_grad()
def main():
    t0=time.time()
    spans={}
    for li in COMPS:
        accs=[]
        for i in range(0,120,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc)
            accs.append(acc[0])
        Y=torch.cat(accs); Yb=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Yb).float(), full_matrices=False)
        spans[li]=(orth(Vh[:8].T),Yb.float())
    clsA=classify(CA,CB).to(DEV); clsC=classify(R0,R1).to(DEV)
    capsO={li:[] for li in COMPS}; capsI={li:[] for li in COMPS}
    hs=[]
    for li in COMPS:
        def mk(li=li):
            def hook(mod,i_,o_):
                capsO[li].append(o_.detach().reshape(-1,D).float())
                capsI[li].append(i_[0].detach().reshape(-1,D).float())
            return hook
        hs.append(m.transformer.h[li].mlp.register_forward_hook(mk()))
    for i in range(CA,CB,4):
        bb=FW[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
    for h in hs: h.remove()
    flatA=clsA.reshape(-1)
    DICT={}; LIN={}
    for li in COMPS:
        Y=torch.cat(capsO[li]); X=torch.cat(capsI[li])
        Q,_=spans[li]; C=Y@Q
        DICT[li]=torch.stack([C[flatA==k].mean(0) if (flatA==k).sum()>0
                              else C.mean(0) for k in range(10)])
        LIN[li]={}
        for k in LINCLS:
            mk_=flatA==k
            Xk=X[mk_]; Ck=C[mk_]
            lam=1e-2*len(Xk)
            LIN[li][k]=torch.linalg.solve(
                Xk.T@Xk+lam*torch.eye(D,device=DEV),Xk.T@Ck)
        capsO[li]=None; capsI[li]=None
    cur={'b0':0}
    def pertok(mode):
        hs=[]
        if mode!='clean':
            for li in COMPS:
                Q,mu=spans[li]; Dq=DICT[li]; Lq=LIN[li]
                def mk(li=li,Q=Q,mu=mu,Dq=Dq,Lq=Lq,mode=mode):
                    def hook(mod,i_,o_):
                        B,T,_=o_.shape
                        c=o_.float().reshape(-1,D)@Q
                        kk=clsC[cur['b0']:cur['b0']+B,:T].reshape(-1)
                        if mode=='ablate': tgt=(mu@Q).expand_as(c)
                        else:
                            tgt=Dq[kk].clone()
                            if mode=='rung1':
                                x=i_[0].float().reshape(-1,D)
                                for k in LINCLS:
                                    sel=kk==k
                                    if sel.any():
                                        tgt[sel]=x[sel]@Lq[k]
                        delta=((c-tgt)@Q.T).view(B,T,D)
                        return o_-delta.to(o_.dtype)
                    return hook
                hs.append(m.transformer.h[li].mlp
                          .register_forward_hook(mk()))
        ces=[]
        for i in range(R0,R1,4):
            cur['b0']=i-R0
            bb=FW[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h:
                x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        for h in hs: h.remove()
        return torch.cat(ces)
    base=pertok('clean')
    abl=pertok('ablate')-base
    d0=pertok('dict')-base
    d1=pertok('rung1')-base
    ta=float(abl.mean()); t0_=float(d0.mean()); t1=float(d1.mean())
    rec0=1-t0_/ta; rec1=1-t1/ta
    flatC=clsC.reshape(-1).cpu()
    ind=flatC==8
    ri0=1-float(d0[ind].mean())/max(float(abl[ind].mean()),1e-6)
    ri1=1-float(d1[ind].mean())/max(float(abl[ind].mean()),1e-6)
    nats_const=rec0*ta; nats_lin=(rec1-rec0)*ta
    v_const=nats_const/0.00064e2  # 640 params -> per 100K = *156
    v_const=nats_const/640*1e5; v_lin=nats_lin/147456*1e5
    pa=rec1>=0.65; pb=ri1>=0.40; pc=v_const>v_lin
    out={'ablate':round(ta,4),'dict':round(t0_,4),'rung1':round(t1,4),
         'recovery0':round(rec0,3),'recovery1':round(rec1,3),
         'ind_rec0':round(ri0,2),'ind_rec1':round(ri1,2),
         'nats_per_100K_const':round(v_const,3),
         'nats_per_100K_linear':round(v_lin,3),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f'ablate {ta:+.4f} | dict {t0_:+.4f} ({rec0:.0%}) | rung1 '
          f'{t1:+.4f} ({rec1:.0%}) | ind {ri0:.0%} -> {ri1:.0%}')
    print(f'value: const {v_const:.2f} vs linear {v_lin:.3f} nats/100K')
    print(f"(a) total >=65%: {'HELD' if pa else 'FAILED'}")
    print(f"(b) ind >=40%: {'HELD' if pb else 'FAILED'}")
    print(f"(c) constants dominate per-param: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
