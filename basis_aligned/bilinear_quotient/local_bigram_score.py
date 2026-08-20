"""LOCAL BIGRAM SCORE -- 416: head 16.8 is a LOCAL identity-code
comparator (414: 76% of top reads within 2 tokens, offset -1
dominant; score inputs m0|m0 per 411). Hypothesis: its (q, q-1)
score is a BIGRAM-PLAUSIBILITY signal -- comparing adjacent
tokens' identity codes to judge whether they belong together.
Test: correlate the head's offset -1 score with corpus bigram
log-frequency. Bigram counts from the 512-row FW token store
(disjoint from the 16 eval rows); eval restricted to bigrams with
count >= 3 (estimable). Controls: shuffled bigram table; head
16.2 (diffuse reader) expected weaker; head 1.4 (match head)
expected ~0.
REGISTERED PREDICTIONS:
  (a) 16.8: |spearman(score(q,q-1), log bigram count)| >= 0.3;
  (b) null: same corr against a SHUFFLED bigram table < 0.1 in
      absolute value;
  (c) specificity: |corr| for 16.8 exceeds both 16.2's and 1.4's
      by >= 0.15."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, FW, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'local_bigram_score_results.json'
HEADS=[(16,8),(16,2),(1,4)]
NR=16

def spearman(a,b):
    a=torch.tensor(a); b=torch.tensor(b)
    ra=a.argsort().argsort().float()
    rb=b.argsort().argsort().float()
    return float(torch.corrcoef(torch.stack([ra,rb]))[0,1])

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    # bigram counts from FW rows 0-511 EXCLUDING the eval rows
    # (cl.rows() on the old tree are FW-derived; exclude by prefix)
    seen={tuple(ROWS[r,:16].tolist()) for r in range(NR)}
    big={}
    for r in range(FW.shape[0]):
        row=FW[r,:257].tolist()
        if tuple(row[:16]) in seen: continue
        for a,b in zip(row[:-1],row[1:]):
            big[(a,b)]=big.get((a,b),0)+1
    print(f'{len(big)} bigrams counted',flush=True)
    g=torch.Generator().manual_seed(5)
    vals=torch.tensor(list(big.values()),dtype=torch.float)
    perm=torch.randperm(len(vals),generator=g)
    keys=list(big.keys())
    shuf={k:float(vals[perm[i]]) for i,k in enumerate(keys)}
    import math
    sc={f'{li}.{hd}':{'s':[],'lb':[],'lbs':[]} for li,hd in HEADS}
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        cap={}
        hs=[m.transformer.h[li].attn.register_forward_pre_hook(
            (lambda li: lambda mo_,args: cap.__setitem__(
                li,args[0]))(li)) for li,_ in HEADS]
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for h in hs: h.remove()
        for li,hd in HEADS:
            at=m.transformer.h[li].attn
            X=cap[li]
            cos,sin=at.rotary(at.c_q(X).view(4,T,9,128))
            qf=F.rms_norm(at.c_q(X).view(4,T,9,128),(128,))[:,:,hd]
            kf=F.rms_norm(at.c_k(X).view(4,T,9,128),(128,))[:,:,hd]
            q2=F.rms_norm(at.c_q2(X).view(4,T,9,128),(128,))[:,:,hd]
            k2=F.rms_norm(at.c_k2(X).view(4,T,9,128),(128,))[:,:,hd]
            qf=are(qf[:,:,None],cos,sin)[:,:,0]
            kf=are(kf[:,:,None],cos,sin)[:,:,0]
            q2=are(q2[:,:,None],cos,sin)[:,:,0]
            k2=are(k2[:,:,None],cos,sin)[:,:,0]
            f1=(qf.float()*torch.roll(kf.float(),1,dims=1)) \
                .sum(-1)
            f2=(q2.float()*torch.roll(k2.float(),1,dims=1)) \
                .sum(-1)
            s=(f1*f2)          # score at (q, q-1), per position
            st=sc[f'{li}.{hd}']
            for b in range(4):
                toks=ROWS[i+b,:T].tolist()
                for q in range(2,T):
                    key=(toks[q-1],toks[q])
                    c=big.get(key,0)
                    if c<3: continue
                    st['s'].append(float(s[b,q]))
                    st['lb'].append(math.log(c))
                    st['lbs'].append(math.log(max(shuf[key],1)))
        print(f'batch {i} done',flush=True)
    out={}
    for k,st in sc.items():
        r=spearman(st['s'],st['lb'])
        rs=spearman(st['s'],st['lbs'])
        out[k]={'corr':round(r,3),'corr_shuf':round(rs,3),
                'n':len(st['s'])}
        print(f"{k}: corr {r:.3f} shuf {rs:.3f} n {len(st['s'])}",
              flush=True)
    c68=abs(out['16.8']['corr'])
    pa=c68>=0.3
    pb=abs(out['16.8']['corr_shuf'])<0.1
    pc=(c68-abs(out['16.2']['corr'])>=0.15 and
        c68-abs(out['1.4']['corr'])>=0.15)
    out.update({'pred_a':bool(pa),'pred_b':bool(pb),
                'pred_c':bool(pc)})
    for nm,v in (('a','16.8 |corr| >= 0.3'),
                 ('b','shuffled-table null < 0.1'),
                 ('c','16.8 exceeds 16.2 and 1.4 by >=0.15')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
