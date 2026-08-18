"""THE CONDITIONAL-CONSTANT DICTIONARY: whole-tail Track-2 submission built
from circuit semantics. Every tail MLP span component (mlp10-17, top-8
output span) is replaced SIMULTANEOUSLY by a per-class constant: sites are
classified by mutually-exclusive function class (priority: digit >
bracket-close > newline > sentence-end > comma > name > repetition >
subword > induction > other), and each component's span coefficients are
fixed to that class's mean (fit window A rows 300-512, eval window C rows
120-300). Description cost: 8 components x 10 classes x 8 numbers = 640
numbers + ten one-line predicates.

REGISTERED PREDICTIONS: (a) joint recovery >= 50%: dictionary damage <=
half of joint span-ablation damage on total CE; (b) subword-class sites
recover >= 70% (extends section 240's single-circuit 88% to the joint
setting); (c) control: label-shuffled dictionary (same constants, permuted
class assignment) recovers <= 10%; (d) class populations and per-class
recovery reported (floor rule)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
import tiktoken
enc=tiktoken.get_encoding('gpt2')
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'circuit_dictionary_results.json'
CA,CB=300,512; R0,R1=120,300
COMPS=list(range(10,18))
CLS=['digit','bclose','newline','sentend','comma','name','rep','subword',
     'ind','other']

def classify(r0,r1):
    Mid=torch.zeros(r1-r0,256,dtype=torch.long)
    for r in range(r0,r1):
        toks=FW[r,:257].tolist()
        for pos in range(256):
            t=toks[pos+1]; p=toks[pos]
            tg=enc.decode([t]); pv=enc.decode([p]); s=tg.strip()
            if s.isdigit() and not tg.startswith(' '): k=0
            elif s in (')',']') and any(b in enc.decode(
                toks[max(0,pos-60):pos+1]) for b in ('(','[')): k=1
            elif '\n' in tg: k=2
            elif tg in ('.','!','?'): k=3
            elif tg==',': k=4
            elif (tg.startswith(' ') and s[:1].isupper() and
                  (pv.strip()[:1].isupper() if pv.strip() else False)): k=5
            elif t==p: k=6
            elif (not tg.startswith(' ')) and s.isalpha(): k=7
            elif t in toks[:pos+1]: k=8
            else: k=9
            Mid[r-r0,pos]=k
    return Mid

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
        print(f'span mlp{li}',flush=True)
    clsA=classify(CA,CB).to(DEV); clsC=classify(R0,R1).to(DEV)
    caps={li:[] for li in COMPS}
    hs=[]
    for li in COMPS:
        def mk(li=li):
            return lambda mo_,i_,o_: caps[li].append(
                o_.detach().reshape(-1,D).float())
        hs.append(m.transformer.h[li].mlp.register_forward_hook(mk()))
    for i in range(CA,CB,4):
        bb=FW[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
    for h in hs: h.remove()
    flatA=clsA.reshape(-1)
    DICT={}
    for li in COMPS:
        Y=torch.cat(caps[li]); Q,_=spans[li]
        C=Y@Q
        DICT[li]=torch.stack([
            C[flatA==k].mean(0) if (flatA==k).sum()>0
            else C.mean(0) for k in range(10)])
    del caps
    g=torch.Generator(device=DEV).manual_seed(0)
    perm=torch.randperm(10,generator=g,device=DEV)
    cur={'b0':0}
    def pertok(mode):
        hs=[]
        if mode!='clean':
            for li in COMPS:
                Q,mu=spans[li]
                Dq=DICT[li]
                def mk(li=li,Q=Q,mu=mu,Dq=Dq,mode=mode):
                    def hook(mod,i_,o_):
                        B,T,_=o_.shape
                        c=o_.float().reshape(-1,D)@Q
                        kk=clsC[cur['b0']:cur['b0']+B,:T].reshape(-1)
                        if mode=='ablate': tgt=(mu@Q).expand_as(c)
                        elif mode=='dict': tgt=Dq[kk]
                        else: tgt=Dq[perm[kk]]
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
    dic=pertok('dict')-base
    shf=pertok('shuffle')-base
    tot_a=float(abl.mean()); tot_d=float(dic.mean()); tot_s=float(shf.mean())
    rec=1-tot_d/max(tot_a,1e-6); rec_s=1-tot_s/max(tot_a,1e-6)
    flatC=clsC.reshape(-1).cpu()
    percls={}
    for k,name in enumerate(CLS):
        mk_=flatC==k
        if mk_.sum()<50: continue
        ra=float(abl[mk_].mean()); rd=float(dic[mk_].mean())
        percls[name]={'n':int(mk_.sum()),'ablate':round(ra,4),
                      'dict':round(rd,4),
                      'recovery':round(1-rd/max(ra,1e-6),2)}
    pa=rec>=0.50
    pb=percls.get('subword',{}).get('recovery',0)>=0.70
    pc=rec_s<=0.10
    out={'total':{'ablate':round(tot_a,4),'dict':round(tot_d,4),
         'shuffled':round(tot_s,4),'recovery':round(rec,3),
         'shuffled_recovery':round(rec_s,3)},
         'per_class':percls,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc),
         'description_cost':'640 numbers + 10 predicates'}
    print(f"\nTOTAL: ablate {tot_a:+.4f} dict {tot_d:+.4f} "
          f"(recovery {rec:.0%}) shuffled rec {rec_s:.0%}")
    for k,v in percls.items(): print(f"  {k:8s} n={v['n']:6d} rec {v['recovery']}")
    print(f"(a) joint recovery >=50%: {'HELD' if pa else 'FAILED'}")
    print(f"(b) subword class >=70%: {'HELD' if pb else 'FAILED'}")
    print(f"(c) shuffled <=10%: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
