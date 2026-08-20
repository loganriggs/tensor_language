"""MODALITY VERIFIED -- the corrected protocol: verify the target,
THEN test modality. (fixes the error of 563 and 566 in code)
566 established that testing a head's modality against an assumed
referent is unreliable -- I made that mistake twice. This builds
the destination step into the protocol: it FIRST measures where
head 10.5 (sentence_end) attends, and tests the recent-vs-second
positional discrimination ONLY if the head genuinely attends to
the referent class, reporting UNEVALUABLE otherwise.
Applied to sentence_end head 10.5, the largest unverified ratio
from 566 (4.64). If the head does attend to prior sentence-enders,
the modality test is valid and 566's positional reading is
confirmed on a verified target; if it does not, 566's ratio was an
artifact and is retracted.
PHASE 1 (destination): signed score-mass share at sentence-end
targets over key classes: recent_sentend, recent_newline,
prev, self, other.
PHASE 2 (modality), run only if recent_sentend share >= 2x its
control: recent vs second-most-recent sentence-ender, real and
rotary-off.
REGISTERED PREDICTIONS:
  (0) POPULATED: >= 30 targets;
  (a) TARGET VERIFIED: the head attends to the recent
      sentence-ender at >= 2x its position-matched control share.
      If not, the modality test is UNEVALUABLE and 566's 4.64 is
      retracted -- reported as the outcome;
  (b) IF VERIFIED, MODALITY: report the recent/second ratio real
      and rotary-off. Positional if ratio >= 2 and rotary-collapses
      >= 40%;
  (c) report both phases;
  NULL: at position-matched control targets the recent-sentend
      share is far below the target share (else the attention is
      not sentence-end-specific)."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=10; HD=5; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'modality_verified_results.json'
NFRESH=256

def sent_end(t): return cl.d1(int(t)).strip() in ('.','!','?')
def isnl(t): return chr(10) in cl.d1(int(t))

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    fresh=cl.fineweb_rows(NFRESH)
    cur=fresh[:,:256]; nxt=fresh[:,1:257]
    se=torch.zeros(NFRESH,T,dtype=torch.bool)
    nl=torch.zeros(NFRESH,T,dtype=torch.bool)
    for r in range(NFRESH):
        for q in range(T):
            if sent_end(cur[r,q]): se[r,q]=True
            if isnl(cur[r,q]): nl[r,q]=True
    tgt=torch.zeros(NFRESH,T,dtype=torch.bool); info={}
    for r in range(NFRESH):
        ses=se[r].nonzero().squeeze(1).tolist()
        nls=nl[r].nonzero().squeeze(1).tolist()
        for q in range(T):
            if sent_end(nxt[r,q]):
                prev=[k for k in ses if k<q]
                info[(r,q)]={'recent_se':prev[-1] if prev else None,
                             'second_se':prev[-2] if len(prev)>=2
                             else None,
                             'recent_nl':max([k for k in nls if k<q],
                                             default=None)}
                tgt[r,q]=True
    g=torch.Generator().manual_seed(29)
    ctrl=torch.zeros_like(tgt); cinfo={}
    for r in range(NFRESH):
        k=int(tgt[r].sum())
        if k==0: continue
        pos=tgt[r].nonzero().squeeze(1)
        j=(torch.randint(-6,7,(k,),generator=g)+pos).clamp(1,T-1)
        for a,q in zip(pos.tolist(),j.tolist()):
            if tgt[r,q]: continue
            d=info[(r,a)]
            cinfo[(r,q)]={kk:(vv if vv is not None and vv<q else None)
                          for kk,vv in d.items()}
            ctrl[r,q]=True
    n=int(tgt.sum())
    print(f'{n} sentence-end targets',flush=True)
    if n<30:
        json.dump({'void':'too few','n':n},open(OUT,'w'),indent=1)
        return
    at=m.transformer.h[LJ].attn; cap={}
    GROUPS=['recent_se','recent_nl','prev','self','other']
    acc={'t':{k:[0.0,0] for k in GROUPS},
         'c':{k:[0.0,0] for k in GROUPS}}
    # phase 2 accumulators
    mod={'real':[0.0,0.0,0],'norot':[0.0,0.0,0]}
    for i in range(0,NFRESH,4):
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
                for nm,mask,src in (('t',tgt,info),('c',ctrl,cinfo)):
                    for r in range(i,min(i+4,NFRESH)):
                        if not mask[r].any(): continue
                        b=r-i
                        for q in mask[r].nonzero().squeeze(1).tolist():
                            d=src.get((r,q))
                            if d is None: continue
                            sh=lambda k:float(p2[b,q,k]/den[b,q])
                            used=set()
                            for gk in ('recent_se','recent_nl'):
                                kk=d.get(gk)
                                if kk is not None:
                                    acc[nm][gk][0]+=sh(kk)
                                    acc[nm][gk][1]+=1; used.add(kk)
                            if q>0: acc[nm]['prev'][0]+=sh(q-1); acc[nm]['prev'][1]+=1
                            acc[nm]['self'][0]+=sh(q); acc[nm]['self'][1]+=1
                            oth=[k for k in range(q+1)
                                 if k not in used and k not in (q-1,q)]
                            if oth:
                                acc[nm]['other'][0]+=sum(sh(k) for k in oth)/len(oth)
                                acc[nm]['other'][1]+=1
            # phase 2: recent vs second sentence-ender
            for r in range(i,min(i+4,NFRESH)):
                if not tgt[r].any(): continue
                b=r-i
                for q in tgt[r].nonzero().squeeze(1).tolist():
                    d=info[(r,q)]
                    if d['recent_se'] is None or d['second_se'] is None:
                        continue
                    mod[tag][0]+=abs(sh_(p2,b,q,d['recent_se'],den))
                    mod[tag][1]+=abs(sh_(p2,b,q,d['second_se'],den))
                    mod[tag][2]+=1
    def _finish():
        S={nm:{k:round(acc[nm][k][0]/max(acc[nm][k][1],1),4)
               for k in GROUPS} for nm in ('t','c')}
        print('\nPHASE 1 destination at sentence-end targets:',
              flush=True)
        for k in GROUPS:
            print(f'  {k:>11}: {S["t"][k]:+.4f} (control '
                  f'{S["c"][k]:+.4f})',flush=True)
        verified=abs(S['t']['recent_se'])>=2*abs(S['c']['recent_se'])
        vtag='HELD' if verified else 'FAILED (566 ratio RETRACTED, modality UNEVALUABLE)'
        print(f"\n(a) target verified (recent_se "
              f"{S['t']['recent_se']:+.4f} >= 2x control "
              f"{S['c']['recent_se']:+.4f}): {vtag}")
        out={'n':n,'phase1':S,'target_verified':bool(verified)}
        if verified and mod['real'][2]>=10:
            rr=(mod['real'][0]/mod['real'][2])/max(
                mod['real'][1]/mod['real'][2],1e-9)
            nr=(mod['norot'][0]/mod['norot'][2])/max(
                mod['norot'][1]/mod['norot'][2],1e-9)
            modality=('positional' if (rr>=2 and (rr-nr)/max(rr,1e-9)>=0.4)
                      else 'detection' if rr<1.5 else 'mixed')
            print(f"(b) MODALITY on verified target: ratio {rr:.2f} "
                  f"-> norot {nr:.2f} = {modality}",flush=True)
            out.update({'ratio':round(rr,2),'ratio_norot':round(nr,2),
                        'modality':modality})
        else:
            print("(b) modality UNEVALUABLE (target not verified)")
            out['modality']='unevaluable'
        json.dump(out,open(OUT,'w'),indent=1)
        print(f'wrote {OUT} ({time.time()-t0:.0f}s)')
    _finish()

def sh_(p2,b,q,k,den):
    return float(p2[b,q,k]/den[b,q])

if __name__=='__main__': main()
