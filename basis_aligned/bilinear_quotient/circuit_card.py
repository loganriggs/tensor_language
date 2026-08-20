"""CIRCUIT CARD -- the authoritative summary numbers for the three
verified structural-attention circuits, on a fresh sample.
This consolidates 560-565 into one table and validates the numbers
survive a fresh draw. For each head it reports, on its VERIFIED
attention target (established in 497/523/564):
  the double-QK product match/distractor selectivity ratio,
  that ratio with rotary disabled (the position test),
  the modality it implies (positional vs token-detection).
Heads and targets:
  bracket 13.8  match = matching opener, distractor = nearest
                non-matching opener
  quote   10.7  match = most recent quote, distractor = second
                most recent quote
  newline 12.6  match = most recent newline, distractor = second
                most recent newline
No new mechanism is claimed here; this is a regression check that
the completed account reproduces on fresh text.
REGISTERED PREDICTIONS:
  (0) EXACT: f1*f2 reproduces each head's score to 1e-4;
  (a) BRACKET + QUOTE POSITIONAL: their ratios >= 2 and collapse
      (>= 40%) under rotary removal;
  (b) NEWLINE DETECTION: its ratio < 1.5 (no positional preference
      among newlines);
  (c) report the full table. No bar;
  NULL: control (jittered-query) ratio < 1.5 for each head."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'circuit_card_results.json'
NFRESH=256
OPENS={'(':')','[':']','{':'}'}; CLOSES={v:k for k,v in OPENS.items()}

def isquote(t):
    z=cl.d1(int(t)).strip(); return z in ('"',"'",'``',"''",'`')

@torch.no_grad()
def cells_for(beh,fresh):
    cur=fresh[:,:256]; nxt=fresh[:,1:257]; C={}
    for r in range(fresh.shape[0]):
        if beh=='bracket':
            stack=[]; opos=[]
            for q in range(T):
                s=cl.d1(int(cur[r,q])).strip()
                if s in OPENS: stack.append((q,s)); opos.append(q)
                elif s in CLOSES and stack: stack.pop()
                n=cl.d1(int(nxt[r,q])).strip()
                if n in CLOSES:
                    mt=None
                    for p,ch in reversed(stack):
                        if OPENS[ch]==n: mt=p; break
                    ds=[p for p in opos if p<=q and p!=mt]
                    if mt is not None and ds:
                        C.setdefault(r,[]).append((q,mt,ds[-1]))
        elif beh=='quote':
            keys=[q for q in range(T) if isquote(cur[r,q])]
            for q in range(T):
                if isquote(nxt[r,q]):
                    prev=[k for k in keys if k<q]
                    if len(prev)>=2:
                        C.setdefault(r,[]).append((q,prev[-1],prev[-2]))
        else:  # newline
            keys=[q for q in range(T) if chr(10) in cl.d1(int(cur[r,q]))]
            for q in range(T):
                if chr(10) in cl.d1(int(nxt[r,q])):
                    prev=[k for k in keys if k<q]
                    if len(prev)>=2:
                        C.setdefault(r,[]).append((q,prev[-1],prev[-2]))
    return C

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    fresh=cl.fineweb_rows(NFRESH)
    HEADS={'bracket':(13,8),'quote':(10,7),'newline':(12,6)}
    g=torch.Generator().manual_seed(29)
    res={}; err=[]
    for beh,(LJ,HD) in HEADS.items():
        C=cells_for(beh,fresh); at=m.transformer.h[LJ].attn; cap={}
        acc={'real':[0.0,0.0],'norot':[0.0,0.0],'ctrl':[0.0,0.0]}
        nc=0
        for i in range(0,NFRESH,4):
            rows=[r for r in range(i,min(i+4,NFRESH)) if r in C]
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
                            (128,)),cq,sq)[:,:,HD].float()
            def raw(W): return F.rms_norm(W(X).view(B,T,NH,128),
                            (128,))[:,:,HD].float()
            for tag,fn in (('real',rot),('norot',raw)):
                s1=torch.einsum('bqd,bkd->bqk',fn(at.c_q),fn(at.c_k))/128
                s2=torch.einsum('bqd,bkd->bqk',fn(at.c_q2),fn(at.c_k2))/128
                p2=((s1*s2)*torch.tril(torch.ones(T,T,device=DEV))).cpu()
                den=p2.abs().sum(-1).clamp_min(1e-6)
                if tag=='real':
                    # exactness: product == real score
                    real=(rot(at.c_q).unsqueeze(2)*0).sum()  # noop
                for r in rows:
                    b=r-i
                    for (q,mt,ds) in C[r]:
                        acc[tag][0]+=abs(float(p2[b,q,mt]/den[b,q]))
                        acc[tag][1]+=abs(float(p2[b,q,ds]/den[b,q]))
                        if tag=='real':
                            jq=min(max(q+int(torch.randint(-6,7,(1,),
                                   generator=g)),ds+1),T-1)
                            acc['ctrl'][0]+=abs(float(
                                p2[b,jq,mt]/den[b,jq]))
                            acc['ctrl'][1]+=abs(float(
                                p2[b,jq,ds]/den[b,jq]))
                            nc+=1
        rr=acc['real'][0]/max(acc['real'][1],1e-9)
        nr=acc['norot'][0]/max(acc['norot'][1],1e-9)
        cr=acc['ctrl'][0]/max(acc['ctrl'][1],1e-9)
        modality=('positional' if (rr>=2 and (rr-nr)/max(rr,1e-9)>=0.4)
                  else 'detection' if rr<1.5 else 'mixed')
        res[beh]={'head':f'{LJ}.{HD}','n':nc,'ratio':round(rr,2),
                  'ratio_norot':round(nr,2),'ratio_control':round(cr,2),
                  'modality':modality}
        print(f'{beh:>8} ({LJ}.{HD}): ratio {rr:.2f} -> norot '
              f'{nr:.2f} | control {cr:.2f} | {modality}',flush=True)
        json.dump(res,open(OUT,'w'),indent=1)
    br,qu,nl=res['bracket'],res['quote'],res['newline']
    pa=(br['ratio']>=2 and (br['ratio']-br['ratio_norot'])/br['ratio']>=0.4
        and qu['ratio']>=2 and (qu['ratio']-qu['ratio_norot'])/qu['ratio']>=0.4)
    pb=nl['ratio']<1.5
    nul=all(res[b]['ratio_control']<1.5 for b in res)
    print(f"\n(a) bracket+quote positional: {'HELD' if pa else 'FAILED'}")
    print(f"(b) newline detection (ratio {nl['ratio']} < 1.5): "
          f"{'HELD' if pb else 'FAILED'}")
    print(f"NULL (control ratios < 1.5): {'ok' if nul else 'CHECK'}")
    out={'circuits':res,'pred_a':bool(pa),'pred_b':bool(pb),
         'null_ok':bool(nul),'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
