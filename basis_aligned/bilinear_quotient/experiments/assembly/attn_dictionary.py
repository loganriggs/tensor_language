"""ATTENTION DICTIONARY rung 0: all 18 attention components replaced
SIMULTANEOUSLY by per-class constant outputs (ten function classes, fit
window A, eval window C). Attention is the transport type (relay; section
239's antecedent finding), so this measures WHICH site classes receive
context-specific attention content vs a type-conditional constant.

REGISTERED PREDICTIONS: (a) joint recovery is markedly lower than the MLP
tail dictionary's 50% -- register 15-35% (attention is transport; constants
cannot carry context); (b) per-class dissociation: formatting classes
(newline, sentend, comma) recover >= 40% while induction recovers <= 10%
(copying is irreducibly contextual); (c) shuffled-label control <= 5%;
(d) per-class ablation floors reported, ratio suppressed where |ablate| <
0.01 (the section-241 floor lesson)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from circuit_dictionary import classify, CLS
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'attn_dictionary_results.json'
CA,CB=300,512; R0,R1=120,300
COMPS=list(range(18))

@torch.no_grad()
def main():
    t0=time.time()
    clsA=classify(CA,CB).to(DEV); clsC=classify(R0,R1).to(DEV)
    caps={li:[] for li in COMPS}
    hs=[]
    for li in COMPS:
        def mk(li=li):
            return lambda mo_,i_,o_: caps[li].append(
                (o_[0] if isinstance(o_,tuple) else o_)
                .detach().reshape(-1,D).float())
        hs.append(m.transformer.h[li].attn.register_forward_hook(mk()))
    for i in range(CA,CB,4):
        bb=FW[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
    for h in hs: h.remove()
    flatA=clsA.reshape(-1)
    DICT={}; GMEAN={}
    for li in COMPS:
        Y=torch.cat(caps[li])
        GMEAN[li]=Y.mean(0)
        DICT[li]=torch.stack([Y[flatA==k].mean(0) if (flatA==k).sum()>0
                              else Y.mean(0) for k in range(10)])
        caps[li]=None
    g=torch.Generator(device=DEV).manual_seed(0)
    perm=torch.randperm(10,generator=g,device=DEV)
    cur={'b0':0}
    def pertok(mode):
        hs=[]
        if mode!='clean':
            for li in COMPS:
                Dq=DICT[li]; mu=GMEAN[li]
                def mk(li=li,Dq=Dq,mu=mu,mode=mode):
                    def hook(mod,i_,o_):
                        out=o_[0] if isinstance(o_,tuple) else o_
                        B,T,_=out.shape
                        kk=clsC[cur['b0']:cur['b0']+B,:T].reshape(-1)
                        if mode=='ablate':
                            new=mu[None,:].expand(B*T,D)
                        elif mode=='dict': new=Dq[kk]
                        else: new=Dq[perm[kk]]
                        new=new.view(B,T,D).to(out.dtype)
                        if isinstance(o_,tuple): return (new,)+o_[1:]
                        return new
                    return hook
                hs.append(m.transformer.h[li].attn
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
    dic=pertok('dict')-base
    shf=pertok('shuffle')-base
    ta=float(abl.mean())
    rec=1-float(dic.mean())/ta; rec_s=1-float(shf.mean())/ta
    flatC=clsC.reshape(-1).cpu()
    percls={}
    for k,name in enumerate(CLS):
        mk_=flatC==k
        if mk_.sum()<50: continue
        ra=float(abl[mk_].mean()); rd=float(dic[mk_].mean())
        e={'n':int(mk_.sum()),'ablate':round(ra,4),'dict':round(rd,4)}
        if abs(ra)>=0.01: e['recovery']=round(1-rd/ra,2)
        percls[name]=e
    fmt=[percls.get(n,{}).get('recovery') for n in ('newline','sentend','comma')]
    fmt=[x for x in fmt if x is not None]
    indr=percls.get('ind',{}).get('recovery')
    pa=0.15<=rec<=0.35
    pb=(len(fmt)>0 and min(fmt)>=0.40 and indr is not None and indr<=0.10)
    pc=rec_s<=0.05
    out={'total':{'ablate':round(ta,4),'recovery':round(rec,3),
                  'shuffled':round(rec_s,3)},'per_class':percls,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f'joint attn ablate {ta:+.3f} | dict recovery {rec:.0%} | '
          f'shuffled {rec_s:.0%}')
    for n,e in percls.items():
        print(f"  {n:8s} n={e['n']:6d} abl {e['ablate']:+.3f} rec "
              f"{e.get('recovery','--')}")
    print(f"(a) recovery in 15-35%: {'HELD' if pa else 'FAILED'}")
    print(f"(b) formatting>=40%, ind<=10%: {'HELD' if pb else 'FAILED'}")
    print(f"(c) shuffled <=5%: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
