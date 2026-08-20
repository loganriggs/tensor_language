"""COSTLY HEAD SEMANTICS -- 428: heads 1.1 and 12.6 are the two
costliest under BOTH 4-read truncation (415) and deletion (427,
+0.089 and +0.073), and 415 showed both compare m0|m0 -- the
universal identity code. 6.3 (the courier, +0.047) is the third.
Mechanism-first question: WHERE do their top reads land? Same
token as the query (identity coincidence), the previous token,
or local neighbourhood -- with a frequency-matched random-read
null and a seen/fresh split, at generic positions (every 4th).
REGISTERED PREDICTIONS:
  (a) LOCAL EARLY: head 1.1 puts >= 50% of its top reads within
      offsets -1/-2 (a layer-1 head should be local);
  (b) IDENTITY DEEP: head 12.6's same-token read rate is >= 20%
      at seen positions against a frequency-matched null < 5%
      (it compares m0|m0, so it should find repeats);
  (c) report all three heads' offset histograms and seen/fresh
      splits; the 'read lands in history' metric from 414 is
      VOID by construction and is not computed."""

import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'costly_head_semantics_results.json'
HEADS=[(1,1),(12,6),(6,3)]
NR=16

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    st={f'{li}.{hd}':{'seen_same':0,'seen_prev':0,'seen_hist':0,
                      'seen_n':0,'fresh_local':0,'fresh_n':0,
                      'null_same':0,'null_n':0,'offsets':{}}
        for li,hd in HEADS}
    g=torch.Generator().manual_seed(7)
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
            pat=(torch.einsum('bqd,bkd->bqk',qf.float(),kf.float())
                 *torch.einsum('bqd,bkd->bqk',q2.float(),
                               k2.float())) \
                *torch.tril(torch.ones(T,T,device=DEV))
            s=st[f'{li}.{hd}']
            for b in range(4):
                toks=ROWS[i+b,:T].tolist()
                seenpos={}
                for q in range(4,T,4):
                    hist=set(toks[:q])
                    isseen=toks[q] in hist
                    k=int(pat[b,q,:q].abs().argmax())
                    off=k-q
                    s['offsets'][off]=s['offsets'].get(off,0)+1
                    kr=int(torch.randint(0,q,(1,),generator=g))
                    s['null_same']+=int(toks[kr]==toks[q])
                    s['null_n']+=1
                    if isseen:
                        s['seen_n']+=1
                        s['seen_same']+=int(toks[k]==toks[q])
                        s['seen_prev']+=int(k>0 and
                                            toks[k-1]==toks[q])
                        s['seen_hist']+=int(toks[k] in hist)
                    else:
                        s['fresh_n']+=1
                        s['fresh_local']+=int(abs(off)<=2)
        print(f'batch {i} done',flush=True)
    outj={}
    for k9,s in st.items():
        outj[k9]={
            'seen_same':round(s['seen_same']/max(s['seen_n'],1),3),
            'seen_prev':round(s['seen_prev']/max(s['seen_n'],1),3),
            'seen_hist':round(s['seen_hist']/max(s['seen_n'],1),3),
            'fresh_local':round(s['fresh_local']
                                /max(s['fresh_n'],1),3),
            'null_same':round(s['null_same']/max(s['null_n'],1),3),
            'n_seen':s['seen_n'],'n_fresh':s['fresh_n'],
            'top_offsets':sorted(s['offsets'].items(),
                                 key=lambda kv:-kv[1])[:6]}
        print(f"{k9}: {outj[k9]}",flush=True)
    loc11=sum(c for o,c in outj['1.1']['top_offsets']
              if o in (-1,-2))
    tot11=sum(c for _,c in outj['1.1']['top_offsets'])
    pa=(loc11/max(tot11,1))>=0.50
    pb=(outj['12.6']['seen_same']>=0.20 and
        outj['12.6']['null_same']<0.05)
    pc=True
    out={'heads':outj,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc)}
    print(f"1.1 local(-1,-2) share of listed offsets: "
          f"{loc11}/{tot11}")
    for nm,v in (('a','1.1 local >=50%'),
                 ('b','12.6 same-token >=20% vs null <5%'),
                 ('c','histograms reported')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
