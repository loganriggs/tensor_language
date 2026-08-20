"""MODALITY BATCH -- is the account PREDICTIVE across more heads?
565 gave a predictive account: a structural head is POSITIONAL if
its task needs a specific positional referent, TOKEN-DETECTION if
it needs only class presence. Three heads fit (bracket, quote
positional; newline detection). This registers advance predictions
for three MORE behaviour-leading heads and tests them, so the
account is judged as a predictor, not fit after the fact.
For each, the target class, the referent it would need, and my
PRE-REGISTERED modality call:
  sentence_end -> a10 (head TBD)  needs: the current clause end.
     Predicting a period is about local clause completion, not a
     specific earlier referent -> PREDICT DETECTION (token).
  open_bracket -> a17             needs: to open a bracket is a
     local decision (a name, a citation) not a lookback ->
     PREDICT DETECTION (token).
  capitalized  -> a15             needs: sentence/proper-noun
     context; the most recent sentence-ender or newline sets it,
     a specific positional referent -> PREDICT POSITIONAL.
Each head is first localized (the atlas gives the leading head of
the layer for that class), then its most-recent-referent share is
measured against a matched distractor (second-most-recent of the
same class) with the rotary probe -- exactly the 565 protocol.
REGISTERED PREDICTIONS:
  (0) each head has >= 30 targets with two prior referents;
  (a) PREDICTIVE ACCURACY: at least 2 of the 3 pre-registered
      modality calls are correct (positional = ratio >= 2 and
      rotary-collapses; detection = ratio < 1.5 and
      rotary-insensitive);
  (b) report each head's ratio, rotary-off ratio, and modality;
  NULL: the rotary probe must move the ratio in the expected
      direction for positional heads (down by >= 40%); a positional
      head whose ratio survives rotary removal contradicts the
      mechanism."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'modality_batch_results.json'
NFRESH=256

def sent_end(t): return cl.d1(int(t)).strip() in ('.','!','?')
def openb(t): return cl.d1(int(t)).strip() in ('(','[','{')
def isnl(t): return chr(10) in cl.d1(int(t))
def iscap(t):
    z=cl.d1(int(t)).strip()
    return bool(z) and z[:1].isupper()

# (behaviour, key-class fn for the referent, target fn, predicted)
TASKS=[('sentence_end',sent_end,sent_end,'detection'),
       ('open_bracket',openb,openb,'detection'),
       ('capitalized',isnl,iscap,'positional')]  # cap referent=newline

@torch.no_grad()
def leading_head(cls_name):
    atlas=json.load(open(PT+'behaviour_atlas2_results.json'))
    top=atlas['summary'][cls_name]['top']       # e.g. 'a10'
    li=int(top[1:])
    # pick the head of that layer with the largest delete cost from
    # head_atlas
    ha=json.load(open(PT+'head_atlas_results.json'))['atlas']
    best=max(range(NH),
             key=lambda h:ha.get(f'{li}.{h}',{}).get('delete_cost',0))
    return li,best

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    fresh=cl.fineweb_rows(NFRESH)
    cur=fresh[:,:256]; nxt=fresh[:,1:257]
    res={}
    for beh,keyfn,tgtfn,pred in TASKS:
        try: li,hd=leading_head(beh)
        except Exception as e:
            print(f'{beh}: no atlas entry ({e})',flush=True); continue
        iskey=torch.zeros(NFRESH,T,dtype=torch.bool)
        for r in range(NFRESH):
            for q in range(T):
                if keyfn(cur[r,q]): iskey[r,q]=True
        cells={}
        for r in range(NFRESH):
            keys=iskey[r].nonzero().squeeze(1).tolist()
            for q in range(T):
                if tgtfn(nxt[r,q]):
                    prev=[k for k in keys if k<q]
                    if len(prev)>=2:
                        cells.setdefault(r,[]).append(
                            (q,prev[-1],prev[-2]))
        n=sum(len(v) for v in cells.values())
        at=m.transformer.h[li].attn; cap={}
        acc={'real':[0.0,0.0,0],'norot':[0.0,0.0,0]}
        for i in range(0,NFRESH,4):
            rows=[r for r in range(i,min(i+4,NFRESH)) if r in cells]
            if not rows: continue
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); B=bb.shape[0]
            hc=at.register_forward_pre_hook(
                lambda mo_,a_: cap.__setitem__('X',a_[0]))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            hc.remove()
            X=cap['X']
            cq,sq=at.rotary(at.c_q(X).view(B,T,NH,128))
            def rot(W): return are(F.rms_norm(W(X).view(B,T,NH,128),
                            (128,)),cq,sq)[:,:,hd].float()
            def raw(W): return F.rms_norm(W(X).view(B,T,NH,128),
                            (128,))[:,:,hd].float()
            for tag,fn in (('real',rot),('norot',raw)):
                s1=torch.einsum('bqd,bkd->bqk',fn(at.c_q),
                                fn(at.c_k))/128
                s2=torch.einsum('bqd,bkd->bqk',fn(at.c_q2),
                                fn(at.c_k2))/128
                p2=((s1*s2)*torch.tril(torch.ones(T,T,device=DEV))) \
                    .cpu()
                den=p2.abs().sum(-1).clamp_min(1e-6)
                for r in rows:
                    b=r-i
                    for (q,mt,ds) in cells[r]:
                        acc[tag][0]+=abs(float(p2[b,q,mt]/den[b,q]))
                        acc[tag][1]+=abs(float(p2[b,q,ds]/den[b,q]))
                        acc[tag][2]+=1
        if acc['real'][2]<30:
            res[beh]={'n':acc['real'][2],'status':'too few'}
            print(f'{beh} ({li}.{hd}): only {acc["real"][2]} '
                  f'targets, skipped',flush=True)
            continue
        rr=(acc['real'][0]/acc['real'][2])/max(
            acc['real'][1]/acc['real'][2],1e-9)
        nr=(acc['norot'][0]/acc['norot'][2])/max(
            acc['norot'][1]/acc['norot'][2],1e-9)
        modality=('positional' if (rr>=2 and (rr-nr)/max(rr,1e-9)>=0.4)
                  else 'detection' if rr<1.5 else 'mixed')
        correct=(modality==pred)
        res[beh]={'head':f'{li}.{hd}','n':acc['real'][2],
                  'ratio':round(rr,2),'ratio_norot':round(nr,2),
                  'modality':modality,'predicted':pred,
                  'correct':correct}
        print(f'{beh} ({li}.{hd}): ratio {rr:.2f} -> norot {nr:.2f}'
              f' | measured {modality} | predicted {pred} | '
              f"{'CORRECT' if correct else 'WRONG'}",flush=True)
        json.dump(res,open(OUT,'w'),indent=1)
    scored=[v for v in res.values() if 'correct' in v]
    ncorr=sum(1 for v in scored if v['correct'])
    pa=ncorr>=2
    print(f"\n(a) predictive accuracy {ncorr}/{len(scored)} "
          f"(>=2 of 3): {'HELD' if pa else 'FAILED'}")
    out={'heads':res,'n_correct':ncorr,'n_scored':len(scored),
         'pred_a':bool(pa),'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
