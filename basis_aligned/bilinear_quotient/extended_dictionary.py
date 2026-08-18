"""TAXONOMY EXTENSION: evaluate the 30 agent-proposed residual-class labels
mechanically, extend the ten-class taxonomy with the accepted ones, and
re-run the tail-MLP constants-only dictionary with the extended classes.
Acceptance (mechanical): a proposed label is accepted if its rule fires on
>= 100 window-A residual-class sites, after dedup (fire-set Jaccard > 0.7
keeps the first). New classes slot BELOW the existing ten in priority, so
they only ever reassign residual sites.

REGISTERED PREDICTIONS: (a) >= 8 of ~30 proposals accepted; (b) residual
('other') share of well-predicted sites drops from 36% to <= 25% on window
C; (c) the constants-only tail dictionary improves from 50.0% to >= 55%
total recovery with the extended taxonomy (each new class buys its sites a
tuned constant); (d) label-shuffled control <= 10% as before."""
import json, sys, glob, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from circuit_dictionary import classify as classify10, COMPS, CLS
import tiktoken
enc=tiktoken.get_encoding('gpt2')
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'extended_dictionary_results.json'
CA,CB=300,512; R0,R1=120,300

def tclass(tok):
    if '\n' in tok: return 'newline'
    s=tok.strip()
    if not s: return 'other'
    if s.isdigit(): return 'digit'
    if all(not c.isalnum() for c in s): return 'punct'
    if s[0].isupper(): return 'upper_start'
    if tok.startswith(' ') and s.isalpha(): return 'space_word'
    return 'other'

def feats_for(r0,r1):
    Fs=[]
    for row in range(r0,r1):
        toks=FW[row,:257].tolist()
        for pos in range(256):
            t=toks[pos+1]; p=toks[pos]
            p2=toks[pos-1] if pos>=1 else -1
            tg=enc.decode([t])
            Fs.append({'target':tg,'prev':enc.decode([p]),
                'prev2':enc.decode([p2]) if p2>=0 else '',
                'is_induction':t in toks[:pos+1],
                'target_equals_prev':t==p,
                'target_class':tclass(tg),'pos':pos})
    return Fs

def fires(rule,f):
    if 'target_in' in rule and f['target'] not in rule['target_in']: return False
    if 'prev_in' in rule and f['prev'] not in rule['prev_in']: return False
    if 'prev2_in' in rule and f['prev2'] not in rule['prev2_in']: return False
    if rule.get('is_induction') and not f['is_induction']: return False
    if rule.get('target_equals_prev') and not f['target_equals_prev']: return False
    if 'target_class_in' in rule and f['target_class'] not in rule['target_class_in']: return False
    if 'pos_min' in rule and f['pos']<rule['pos_min']: return False
    if 'pos_max' in rule and f['pos']>rule['pos_max']: return False
    return True

@torch.no_grad()
def main():
    t0=time.time()
    props=[]
    for f in sorted(glob.glob(PT+'labels/proposed_*.json')):
        props+=json.load(open(f))
    print(f'{len(props)} proposals',flush=True)
    FA=feats_for(CA,CB); FC=feats_for(R0,R1)
    base10A=classify10(CA,CB).reshape(-1)
    otherA=(base10A==9)
    accepted=[]; firesets=[]
    for pr in props:
        rule=pr.get('rule')
        if not rule: continue
        hit=torch.tensor([fires(rule,f) for f in FA])
        hitso=hit&otherA
        n=int(hitso.sum())
        if n<100: continue
        dup=False
        for fs in firesets:
            inter=int((hitso&fs).sum()); uni=int((hitso|fs).sum())
            if uni>0 and inter/uni>0.7: dup=True; break
        if dup: continue
        firesets.append(hitso)
        accepted.append(pr)
        print(f"  accept {pr['name']:28s} n_other={n}",flush=True)
    K=10+len(accepted)
    def classifyK(Fs, base10flat):
        kk=base10flat.clone()
        oth=(kk==9)
        for j,pr in enumerate(accepted):
            hit=torch.tensor([fires(pr['rule'],f) for f in Fs])
            sel=oth&hit&(kk==9)
            kk[sel]=10+j
        return kk
    clsA=classifyK(FA,base10A).to(DEV)
    base10C=classify10(R0,R1).reshape(-1)
    clsC=classifyK(FC,base10C).to(DEV)
    resid=float((clsC==9).float().mean())
    spans={}
    for li in COMPS:
        accs=[]
        for i in range(0,120,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc)
            accs.append(acc[0])
        Y=torch.cat(accs); Yb=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Yb).float(), full_matrices=False)
        spans[li]=(orth(Vh[:8].T),Yb.float())
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
        Y=torch.cat(caps[li]); Q,_=spans[li]; C=Y@Q
        DICT[li]=torch.stack([C[flatA==k].mean(0) if (flatA==k).sum()>0
                              else C.mean(0) for k in range(K)])
        caps[li]=None
    g=torch.Generator(device=DEV).manual_seed(0)
    perm=torch.randperm(K,generator=g,device=DEV)
    cur={'b0':0}
    clsC2=clsC.view(R1-R0,256)
    def pertok(mode):
        hs=[]
        if mode!='clean':
            for li in COMPS:
                Q,mu=spans[li]; Dq=DICT[li]
                def mk(li=li,Q=Q,mu=mu,Dq=Dq,mode=mode):
                    def hook(mod,i_,o_):
                        B,T,_=o_.shape
                        c=o_.float().reshape(-1,D)@Q
                        kk=clsC2[cur['b0']:cur['b0']+B,:T].reshape(-1)
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
    abl=float((pertok('ablate')-base).mean())
    dic=float((pertok('dict')-base).mean())
    shf=float((pertok('shuffle')-base).mean())
    rec=1-dic/abl; rec_s=1-shf/abl
    pa=len(accepted)>=8; pb=resid<=0.25; pc=rec>=0.55; pd=rec_s<=0.10
    json.dump([{'name':p['name'],'story':p['story'],'rule':p['rule']}
               for p in accepted],open(PT+'labels/accepted.json','w'),
              indent=1)
    out={'proposed':len(props),'accepted':len(accepted),
         'residual_share_C':round(resid,3),'K':K,
         'ablate':round(abl,4),'recovery':round(rec,3),
         'shuffled':round(rec_s,3),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'pred_d':bool(pd)}
    print(f"\naccepted {len(accepted)}/{len(props)} | residual {resid:.0%} "
          f"| K={K} | recovery {rec:.1%} (was 50.0%) | shuffled {rec_s:.0%}")
    print(f"(a) >=8 accepted: {'HELD' if pa else 'FAILED'}")
    print(f"(b) residual <=25%: {'HELD' if pb else 'FAILED'}")
    print(f"(c) recovery >=55%: {'HELD' if pc else 'FAILED'}")
    print(f"(d) shuffled <=10%: {'HELD' if pd else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
