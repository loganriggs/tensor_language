"""CLOSE BRACKET -- the largest concentrated effect in the program.
The behaviour atlas (513) screened ten behaviour classes and the
biggest number by far was closing brackets: ablating attention
layer 13 costs +0.694 nats at positions where the next token is
) ] or }, against +0.015 elsewhere. Nothing else this program has
measured is that concentrated, and a bracket closer is exactly the
kind of computation that should have a dedicated mechanism --
predicting ")" well requires knowing an unclosed "(" is open,
which is state no bigram can carry.
This runs the newline pipeline (495-506) on it: localize to a
head, verify with the atlas, characterize causally, and then ask
the mechanism question that distinguishes the two hypotheses.
  DETECTOR: the head notices the local context looks bracket-ish
    and pushes ")" -- no memory, no matching.
  MATCHER: the head attends to the specific OPEN bracket its
    closer belongs to, which requires tracking nesting.
Those differ in where the score mass goes at a close-bracket
target: a matcher puts it on the MATCHING opener (computed with a
stack), a detector spreads it over any earlier opener or over
local context.
ADVANCE BET, weaker than the newline one and labelled as such. Of
a13's nine heads the atlas -- which knew nothing about brackets --
gives punctuation read-enrichment above 2 to exactly three: 13.3
(3.22), 13.5 (2.98), 13.8 (2.16), and 13.8/13.3 also carry the two
largest deletion costs in the layer. So the atlas narrows nine to
three, which can fail six ways. It does NOT single out one head
the way it did for newlines, and the prediction is registered at
that strength.
128 rows, because closing brackets are rare -- 48 rows gave only
30 targets, which is too few to price a head on.
REGISTERED PREDICTIONS:
  (a) CONCENTRATION: some single head carries >= 40% of a13's
      close-bracket damage;
  (b) ATLAS: the leading head is one of 13.3, 13.5, 13.8;
  (c) IT PUSHES THE CLOSER: deleting the leading head lowers the
      logit of the actual closing-bracket token by >= 0.10 more
      than it lowers the best non-bracket competitor at the same
      positions;
  (d) MATCHER vs DETECTOR: the leading head's signed share of
      score mass on the MATCHING opener exceeds its share on
      non-matching earlier openers by >= 1.5x. Failure means
      detector, not matcher, and that is a real answer.
  NULL 1: position-matched controls (targets jittered +-6) must
      show a smaller effect for the leading head; absolute pair
      reported, no quotient scored.
  NULL 2: a fully random target set of the same size must not
      produce any head with concentration >= 2.0 against the
      GLOBAL mean damage (the unbiased denominator of 513).
Reporting: pairs alongside ratios, bars through cl.score_bar."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=13; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'close_bracket_heads_results.json'
NFRESH=128
OPENS={'(':')','[':']','{':'}'}
CLOSES={v:k for k,v in OPENS.items()}

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    fresh=cl.fineweb_rows(NFRESH)
    cur=fresh[:,:256]; nxt=fresh[:,1:257]
    # target: next token IS a closing bracket. Also record, for each
    # target, the position of its MATCHING opener via a stack.
    tgt=torch.zeros(NFRESH,T,dtype=torch.bool)
    match={}; openers={}
    CLOSE_IDS=set()
    for r in range(NFRESH):
        stack=[]; opos=[]
        for q in range(T):
            s=cl.d1(int(cur[r,q])).strip()
            if s in OPENS: stack.append((q,s)); opos.append(q)
            elif s in CLOSES and stack: stack.pop()
            n=cl.d1(int(nxt[r,q])).strip()
            if n in CLOSES:
                tgt[r,q]=True; CLOSE_IDS.add(int(nxt[r,q]))
                mt=None
                for p,ch in reversed(stack):
                    if OPENS[ch]==n: mt=p; break
                match[(r,q)]=mt
                openers[(r,q)]=list(opos)
    CLOSE_IDS=sorted(CLOSE_IDS)
    g=torch.Generator().manual_seed(29)
    ctrl=torch.zeros_like(tgt); rnd=torch.zeros_like(tgt)
    for r in range(NFRESH):
        k=int(tgt[r].sum())
        if k==0: continue
        pos=tgt[r].nonzero().squeeze(1)
        ctrl[r,(torch.randint(-6,7,(k,),generator=g)+pos)
             .clamp(0,T-1)]=True
        rnd[r,torch.randint(0,T,(k,),generator=g)]=True
    nmatched=sum(1 for v in match.values() if v is not None)
    print(f'{int(tgt.sum())} close-bracket targets over {NFRESH} '
          f'rows | {nmatched} have a matching opener in window | '
          f'closer ids {CLOSE_IDS}',flush=True)
    if int(tgt.sum())<40:
        print('*** too few targets to price a head -- VOID ***')
        json.dump({'void':'too few targets',
                   'n':int(tgt.sum())},open(OUT,'w'),indent=1); return
    at=m.transformer.h[LJ].attn

    def mkhook(HD):
        def fh(mo,args,o_):
            y,v1r=o_; X=args[0]; B=X.shape[0]
            v1b=args[1] if args[1] is not None else v1r
            z,_=cl.head_parts(LJ,X,v1b)
            z=z.clone()
            z[:,HD]=z[:,HD].mean(dim=(0,1),keepdim=True)
            return (at.c_proj(z.transpose(1,2).contiguous()
                    .view(B,T,-1).to(X.dtype)),v1r)
        return fh

    def run(HD):
        ce=torch.zeros(NFRESH,T); lg_c=torch.zeros(NFRESH,T)
        lg_o=torch.zeros(NFRESH,T)
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=bb.shape[0]
            hs=[at.register_forward_hook(mkhook(HD))] \
               if HD is not None else []
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ce[i:i+B]=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                      reduction='none').view(B,T).cpu()
            L=lg.cpu()
            lg_c[i:i+B]=L[:,:,CLOSE_IDS].max(dim=-1).values
            o=L.clone(); o[:,:,CLOSE_IDS]=-1e9
            lg_o[i:i+B]=o.max(dim=-1).values
            for h in hs: h.remove()
        return ce,lg_c,lg_o

    base,bc,bo=run(None)
    gm=float((base*0).mean())   # placeholder; damage uses deltas
    res={}
    for HD in range(NH):
        ce,lc,lo=run(HD)
        d=ce-base
        row={'target':round(float(d[tgt].mean()),5),
             'ctrl':round(float(d[ctrl].mean()),5),
             'rand':round(float(d[rnd].mean()),5),
             'global':round(float(d.mean()),5),
             'dlogit_closer':round(float((bc-lc)[tgt].mean()),5),
             'dlogit_other':round(float((bo-lo)[tgt].mean()),5)}
        res[f'{LJ}.{HD}']=row
        print(f"{LJ}.{HD}: target {row['target']:+.5f} ctrl "
              f"{row['ctrl']:+.5f} global {row['global']:+.5f} | "
              f"dlogit closer {row['dlogit_closer']:+.4f} vs other "
              f"{row['dlogit_other']:+.4f}",flush=True)
        json.dump(res,open(OUT,'w'),indent=1)

    tot=sum(max(v['target'],0) for v in res.values())
    top=max(res,key=lambda k:res[k]['target'])
    share=res[top]['target']/tot if tot>0 else 0.0
    # (d) where the leading head looks at close-bracket targets
    HD=int(top.split('.')[1])
    prof={'match':[0.0,0],'other_open':[0.0,0],'prev':[0.0,0],
          'pos0':[0.0,0]}
    cap={}
    hh=at.register_forward_pre_hook(
        lambda mo_,a_: cap.__setitem__('X',a_[0]))
    for i in range(0,NFRESH,4):
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); B=bb.shape[0]
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        X=cap['X']
        cq,sq=at.rotary(at.c_q(X).view(B,T,NH,128))
        def r2(W):
            return are(F.rms_norm(W(X).view(B,T,NH,128),(128,)),
                       cq,sq)[:,:,HD].float()
        s1=torch.einsum('bqd,bkd->bqk',r2(at.c_q),r2(at.c_k))/128
        s2=torch.einsum('bqd,bkd->bqk',r2(at.c_q2),r2(at.c_k2))/128
        p2=((s1*s2)*torch.tril(torch.ones(T,T,device=DEV))).cpu()
        den=p2.abs().sum(-1).clamp_min(1e-6)
        for b in range(B):
            r=i+b
            for q in tgt[r].nonzero().squeeze(1).tolist():
                mt=match.get((r,q)); ops=openers.get((r,q),[])
                if mt is not None:
                    prof['match'][0]+=float(p2[b,q,mt]/den[b,q])
                    prof['match'][1]+=1
                oth=[p for p in ops if p!=mt and p<=q]
                if oth:
                    prof['other_open'][0]+=float(
                        sum(p2[b,q,p] for p in oth)/len(oth)/den[b,q])
                    prof['other_open'][1]+=1
                if q>0:
                    prof['prev'][0]+=float(p2[b,q,q-1]/den[b,q])
                    prof['prev'][1]+=1
                prof['pos0'][0]+=float(p2[b,q,0]/den[b,q])
                prof['pos0'][1]+=1
    hh.remove()
    P={k:round(v[0]/max(v[1],1),4) for k,v in prof.items()}
    va,_=cl.score_bar('a',share,0.40)
    vb='HELD' if top in ('13.3','13.5','13.8') else 'FAILED'
    vc,_=cl.score_bar('c',res[top]['dlogit_closer']
                      -res[top]['dlogit_other'],0.10)
    vd,_=cl.score_bar('d',P['match'],1.5*abs(P['other_open']),
                      denom=P['other_open'],ref=P['match'])
    n1=res[top]['target']>res[top]['ctrl']
    worst=max((v['rand']/v['global'] if abs(v['global'])>1e-4 else 0)
              for v in res.values())
    n2=worst<2.0
    print(f"\nleading head {top}: {share*100:.0f}% of a13's "
          f"close-bracket damage")
    print(f"(b) atlas narrowed 9 heads to 13.3/13.5/13.8; leader "
          f"is {top}: {vb}")
    print(f"(d) score mass: matching opener {P['match']}, other "
          f"openers {P['other_open']}, prev {P['prev']}, pos0 "
          f"{P['pos0']}")
    print(f"NULL 1 (target beats position-matched control, "
          f"{res[top]['target']:+.5f} vs {res[top]['ctrl']:+.5f}): "
          f"{'ok' if n1 else 'VIOLATED'}")
    print(f"NULL 2 (no head fires on random targets, worst "
          f"{worst:.2f} < 2.0): {'ok' if n2 else 'VIOLATED'}")
    out={'heads':res,'top':top,'top_share':round(share,3),
         'attn_profile':P,'n_targets':int(tgt.sum()),
         'n_matched':nmatched,'closer_ids':CLOSE_IDS,
         'pred_a':va=='HELD','pred_b':vb=='HELD',
         'pred_c':vc=='HELD','pred_d':vd=='HELD',
         'null1_ok':bool(n1),'null2_ok':bool(n2),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
