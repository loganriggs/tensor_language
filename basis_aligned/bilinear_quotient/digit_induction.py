"""DIGIT INDUCTION -- is head 8.3 copying digits? (verified target
first)
571 found head 8.3 attends to the most recent digit at 10.3x, with
detection modality. The natural hypothesis: it is doing digit
INDUCTION -- attend to a prior digit and predict the token that
FOLLOWED it, i.e. induction restricted to the digit class. That
would give the distributed digit behaviour (521, 569) a nameable
attention mechanism on top of its subspace.
Test, target verified first per the protocol lesson (563/566/567):
PHASE 1 confirms 8.3 attends to a prior digit (already 10.3x in
571, re-confirmed here on the same sample as phase 2).
PHASE 2, the induction test: at digit-target positions whose
current token is itself a digit that appeared before, does the
head attend to (prior-digit-occurrence + 1) -- the induction key,
what followed that digit last time -- more than to the prior digit
itself or to a control? Classic induction attends to occurrence+1,
not to the occurrence. If 8.3 attends to occurrence+1, it is doing
digit copying; if it attends to the digit occurrence itself, it is
a digit DETECTOR that gates the pool but does not copy, and the
copying (if any) is elsewhere.
REGISTERED PREDICTIONS:
  (0) POPULATED: >= 50 positions where the current token is a digit
      with a single clean prior occurrence;
  (a) TARGET: the head attends to prior digits at >= 2x control
      (re-confirms 571 on this sample);
  (b) INDUCTION vs DETECTION: compare the head's share on
      (prior-occurrence + 1) against its share on the prior
      occurrence itself. Induction if occ+1 share >= 1.5x the
      occurrence share; detection if the occurrence share
      dominates. Report the ratio -- this is the finding;
  (c) report both shares and a non-digit control;
  NULL: at positions whose current token is a NON-digit, the head's
      share on the analogous occ+1 key is far lower -- the
      mechanism is digit-specific."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=8; HD=3; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'digit_induction_results.json'
NFRESH=256

def isdig(t):
    z=cl.d1(int(t)).strip(); return bool(z) and z[0].isdigit()

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    fresh=cl.fineweb_rows(NFRESH)
    cur=fresh[:,:256]
    # targets: current token is a digit with exactly one clean prior
    # occurrence at least 3 back, and occ+1 exists and < q
    tgt=torch.zeros(NFRESH,T,dtype=torch.bool); info={}
    for r in range(NFRESH):
        last={}
        for q in range(T):
            t=int(cur[r,q])
            if isdig(cur[r,q]) and t in last and q-last[t]>=3 \
               and last[t]+1<q:
                info[(r,q)]={'occ':last[t],'occ1':last[t]+1,
                             'recent_dig':last[t]}
                tgt[r,q]=True
            last[t]=q
    n=int(tgt.sum())
    print(f'{n} digit induction targets',flush=True)
    if n<50:
        json.dump({'void':'too few','n':n},open(OUT,'w'),indent=1)
        return
    # control targets: current token NON-digit, with a prior occ
    ctl=torch.zeros(NFRESH,T,dtype=torch.bool); cinfo={}
    for r in range(NFRESH):
        last={}
        for q in range(T):
            t=int(cur[r,q])
            if (not isdig(cur[r,q])) and t in last and q-last[t]>=3 \
               and last[t]+1<q and not tgt[r,q]:
                cinfo[(r,q)]={'occ1':last[t]+1,'occ':last[t]}
                ctl[r,q]=True
            last[t]=q
    at=m.transformer.h[LJ].attn; cap={}
    acc={'occ1':[0.0,0],'occ':[0.0,0],'ctrl_occ1':[0.0,0]}
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
        def r2(W): return are(F.rms_norm(W(X).view(B,T,NH,128),
                        (128,)),cq,sq)[:,:,HD].float()
        s1=torch.einsum('bqd,bkd->bqk',r2(at.c_q),r2(at.c_k))/128
        s2=torch.einsum('bqd,bkd->bqk',r2(at.c_q2),r2(at.c_k2))/128
        p2=((s1*s2)*torch.tril(torch.ones(T,T,device=DEV))).cpu()
        den=p2.abs().sum(-1).clamp_min(1e-6)
        for r in range(i,min(i+4,NFRESH)):
            b=r-i
            for q in tgt[r].nonzero().squeeze(1).tolist():
                d=info[(r,q)]
                acc['occ1'][0]+=abs(float(p2[b,q,d['occ1']]/den[b,q]))
                acc['occ1'][1]+=1
                acc['occ'][0]+=abs(float(p2[b,q,d['occ']]/den[b,q]))
                acc['occ'][1]+=1
            for q in (ctl[r].nonzero().squeeze(1).tolist()
                      if ctl[r].any() else []):
                d=cinfo[(r,q)]
                acc['ctrl_occ1'][0]+=abs(float(p2[b,q,d['occ1']]/den[b,q]))
                acc['ctrl_occ1'][1]+=1
    o1=acc['occ1'][0]/max(acc['occ1'][1],1)
    oc=acc['occ'][0]/max(acc['occ'][1],1)
    co1=acc['ctrl_occ1'][0]/max(acc['ctrl_occ1'][1],1)
    print(f'\nshare on (prior digit + 1): {o1:.4f}',flush=True)
    print(f'share on prior digit itself: {oc:.4f}',flush=True)
    print(f'control (non-digit) occ+1 share: {co1:.4f}',flush=True)
    ratio=o1/max(oc,1e-9)
    induction=(ratio>=1.5)
    pa=oc>=2*co1 or o1>=2*co1
    print(f"\n(a) target attends digits (vs control): "
          f"{'HELD' if pa else 'FAILED'}")
    print(f"(b) occ+1 / occ ratio {ratio:.2f}: "
          f"{'INDUCTION (copies digits)' if induction else 'DETECTION (attends the digit, not what followed)'}")
    print(f"NULL (digit-specific: target occ+1 {o1:.4f} vs control "
          f"{co1:.4f}): {'ok' if o1>1.5*co1 else 'weak'}")
    out={'n':n,'share_occ_plus_1':round(o1,4),
         'share_occ':round(oc,4),'control_occ1':round(co1,4),
         'occ1_over_occ':round(ratio,3),
         'verdict':'induction' if induction else 'detection',
         'pred_a':bool(pa),'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
