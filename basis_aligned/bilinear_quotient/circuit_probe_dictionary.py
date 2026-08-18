"""INPUT-ONLY DICTIONARY via stream probe. Section 243: context rules
predict site class at only 31% -- but the model has plausibly already
decided the type by the time the stream reaches the tail. Two stages:
(1) linear probe: 10-class softmax-free ridge one-vs-all on the residual
stream ENTERING mlp10 (post-lambda-mix normalized input), fit window A,
accuracy on window C; (2) the probe-conditioned dictionary: section 241's
constants + section 242's two linear classes, but with class = probe
argmax instead of oracle. Fully input-only, deployable.

REGISTERED PREDICTIONS: (a) probe top-1 accuracy >= 65% (the tail's input
stream linearly encodes site type -- the front of the model has decided);
(b) probe-conditioned dictionary recovers >= 70% of joint ablation (vs 95%
oracle -- classifier errors cost <= 25 points); (c) the per-class accuracy
correlates with that class's dictionary recovery share (the model decides
best where content is most stereotyped); (d) shuffled-probe control
<= 10% recovery."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from circuit_dictionary import classify, COMPS, CLS
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'circuit_probe_dictionary_results.json'
CA,CB=300,512; R0,R1=120,300

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
    def capture(r0,r1,cls_):
        capsO={li:[] for li in COMPS}; capI=[]
        hs=[]
        for li in COMPS:
            def mk(li=li):
                def hook(mod,i_,o_):
                    capsO[li].append(o_.detach().reshape(-1,D).float())
                    if li==COMPS[0]:
                        capI.append(i_[0].detach().reshape(-1,D).float())
                return hook
            hs.append(m.transformer.h[li].mlp.register_forward_hook(mk()))
        for i in range(r0,r1,4):
            bb=FW[i:i+4,:257].to(DEV)
            m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
        for h in hs: h.remove()
        return {li:torch.cat(v) for li,v in capsO.items()}, torch.cat(capI)
    capA,XA=capture(CA,CB,clsA)
    flatA=clsA.reshape(-1)
    Yoh=torch.zeros(len(flatA),10,device=DEV)
    Yoh[torch.arange(len(flatA)),flatA]=1.0
    lam=1e-2*len(XA)
    W=torch.linalg.solve(XA.T@XA+lam*torch.eye(D,device=DEV),XA.T@Yoh)
    DICT={}; LIN={}
    for li in COMPS:
        Q,_=spans[li]; C=capA[li]@Q
        DICT[li]=torch.stack([C[flatA==k].mean(0) if (flatA==k).sum()>0
                              else C.mean(0) for k in range(10)])
    XAfull=XA
    for li in COMPS:
        Q,_=spans[li]; C=capA[li]@Q
        LIN[li]={}
        for k in (8,9):
            mk_=flatA==k
            Xk=XAfull[mk_]; Ck=C[mk_]
            l2=1e-2*len(Xk)
            LIN[li][k]=torch.linalg.solve(Xk.T@Xk+l2*torch.eye(D,device=DEV),
                                          Xk.T@Ck)
    del capA, XA
    g=torch.Generator(device=DEV).manual_seed(0)
    permW=W[:,torch.randperm(10,generator=g,device=DEV)]
    cur={'b0':0}
    predC={'probe':None,'shuffle':None}
    def pertok(mode):
        hs=[]; store=[]
        if mode!='clean' :
            for j,li in enumerate(COMPS):
                Q,mu=spans[li]; Dq=DICT[li]; Lq=LIN[li]
                def mk(li=li,Q=Q,mu=mu,Dq=Dq,Lq=Lq,mode=mode,first=(j==0)):
                    def hook(mod,i_,o_):
                        B,T,_=o_.shape
                        x=i_[0].float().reshape(-1,D)
                        c=o_.float().reshape(-1,D)@Q
                        if mode=='ablate': tgt=(mu@Q).expand_as(c)
                        else:
                            if mode=='oracle':
                                kk=clsC[cur['b0']:cur['b0']+B,:T].reshape(-1)
                            else:
                                Wp=permW if mode=='shuffle' else W
                                kk=(x@Wp).argmax(1)
                                if first and mode=='probe':
                                    store.append(kk)
                            tgt=Dq[kk].clone()
                            for k in (8,9):
                                sel=kk==k
                                if sel.any(): tgt[sel]=x[sel]@Lq[k]
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
        return torch.cat(ces), (torch.cat(store) if store else None)
    base,_=pertok('clean')
    abl,_=pertok('ablate'); abl=abl-base
    orc,_=pertok('oracle'); orc=orc-base
    prb,kk=pertok('probe'); prb=prb-base
    shf,_=pertok('shuffle'); shf=shf-base
    acc=float((kk==clsC.reshape(-1)).float().mean())
    ta=float(abl.mean())
    rec_o=1-float(orc.mean())/ta; rec_p=1-float(prb.mean())/ta
    rec_s=1-float(shf.mean())/ta
    pa=acc>=0.65; pb=rec_p>=0.70; pd=rec_s<=0.10
    out={'probe_acc':round(acc,3),'ablate':round(ta,4),
         'oracle_recovery':round(rec_o,3),'probe_recovery':round(rec_p,3),
         'shuffle_recovery':round(rec_s,3),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_d':bool(pd)}
    print(f'probe acc {acc:.0%} | recovery: oracle {rec_o:.0%} probe '
          f'{rec_p:.0%} shuffled-probe {rec_s:.0%}')
    print(f"(a) probe acc >=65%: {'HELD' if pa else 'FAILED'}")
    print(f"(b) probe-dict recovery >=70%: {'HELD' if pb else 'FAILED'}")
    print(f"(d) shuffled <=10%: {'HELD' if pd else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
