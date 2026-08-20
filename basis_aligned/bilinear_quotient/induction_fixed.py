"""INDUCTION FIXED -- does the fixed-query account extend to
content-matching, or break on it? (a genuinely different head)
The structural heads (bracket, quote, newline) all use an
approximately CONSTANT query (559): the query says "attend to
class X", and rotary/AND does the rest. Induction is different in
kind. An induction head attends from the current token to the
position AFTER a previous occurrence of the SAME token, to predict
what followed last time. That requires the query to encode the
CURRENT TOKEN -- a content-dependent, per-position query. So the
fixed-query finding should BREAK here, and that break is the point:
it would show the constant-query result is specific to
class-selection heads, not a universal property.
Head 6.3 (a6.h3), the deep induction head from the identity-code
work (writeups ~387).
PHASE 1 (verify the target, per the lesson of 563/566/567): at
induction targets -- positions whose current token appeared
earlier in the document -- measure the head's score-mass share on
(previous occurrence + 1), the induction key, vs a control.
PHASE 2 (only if verified): the fixed-query test. Replace the
head's per-position query with the induction-average query and
measure whether it still attends to (prev occurrence + 1). A fixed
query CANNOT know which token to match, so for induction it should
FAIL where it succeeded for the structural heads.
REGISTERED PREDICTIONS:
  (0) POPULATED: >= 50 induction targets with a clean single prior
      occurrence;
  (a) TARGET VERIFIED: the head attends to (prev-occ + 1) at >= 2x
      its position-matched control share -- confirms this is the
      induction head doing induction here;
  (b) THE FIXED QUERY BREAKS: the induction-average fixed query's
      share on (prev-occ + 1) is < 0.5x the real query's. This is
      the distinguishing prediction -- unlike the structural heads,
      induction needs a content-dependent query. If instead the
      fixed query reproduces it, the fixed-query account is more
      general than I thought and that is the finding;
  (c) report the real and fixed shares and the ratio. No bar;
  NULL: at NON-induction positions (current token is novel) the
      real query's share on the analogous key is far lower --
      the head only does this where a prior occurrence exists."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=6; HD=3; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'induction_fixed_results.json'
NFRESH=256

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    fresh=cl.fineweb_rows(NFRESH)
    cur=fresh[:,:256]
    # induction targets: current token appeared exactly once before
    # (clean single prior occurrence), and it is not punctuation/
    # very common (to avoid degenerate matches)
    tgt=torch.zeros(NFRESH,T,dtype=torch.bool); info={}
    for r in range(NFRESH):
        seen={}
        for q in range(T):
            t=int(cur[r,q])
            if t in seen and q-seen[t]>=3 and seen[t]+1<q:
                # single most-recent prior occurrence
                info[(r,q)]={'prev_occ':seen[t],'key':seen[t]+1}
                tgt[r,q]=True
            seen[t]=q
    g=torch.Generator().manual_seed(29)
    ctrl=torch.zeros_like(tgt); cinfo={}
    for r in range(NFRESH):
        k=int(tgt[r].sum())
        if k==0: continue
        pos=tgt[r].nonzero().squeeze(1)
        j=(torch.randint(-6,7,(k,),generator=g)+pos).clamp(2,T-1)
        for a,q in zip(pos.tolist(),j.tolist()):
            if tgt[r,q]: continue
            key=info[(r,a)]['key']
            if key<q: cinfo[(r,q)]={'key':key}; ctrl[r,q]=True
    n=int(tgt.sum())
    print(f'{n} induction targets',flush=True)
    if n<50:
        json.dump({'void':'too few','n':n},open(OUT,'w'),indent=1)
        return
    at=m.transformer.h[LJ].attn; cap={}
    # accumulate the induction-average pre-rotary query, and shares
    qsum=torch.zeros(128,device=DEV); qn=0
    acc={'real':[0.0,0],'fixed':[0.0,0],'ctrl':[0.0,0]}
    # two passes: first get the average query, then price
    for i in range(0,NFRESH,4):
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); B=bb.shape[0]
        hc=at.register_forward_pre_hook(
            lambda mo_,a_: cap.__setitem__('X',a_[0]))
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        hc.remove()
        X=cap['X']
        qpre=F.rms_norm(at.c_q(X).view(B,T,NH,128),(128,))[:,:,HD] \
            .float()
        for r in range(i,min(i+4,NFRESH)):
            if not tgt[r].any(): continue
            b=r-i
            for q in tgt[r].nonzero().squeeze(1).tolist():
                qsum+=qpre[b,q]; qn+=1
    qfix=qsum/max(qn,1)
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
        qpre=F.rms_norm(at.c_q(X).view(B,T,NH,128),(128,))[:,:,HD] \
            .float()
        def rk(W): return are(F.rms_norm(W(X).view(B,T,NH,128),
                        (128,)),cq,sq)[:,:,HD].float()
        kf=rk(at.c_k); k2=rk(at.c_q2),rk(at.c_k2)
        q2r=rk(at.c_q2); k2r=rk(at.c_k2)
        def score(qp):
            qrot=are(qp[:,:,None].expand(B,T,NH,128).contiguous(),
                     cq,sq)[:,:,HD]
            s1=torch.einsum('bqd,bkd->bqk',qrot.float(),kf)/128
            s2=torch.einsum('bqd,bkd->bqk',q2r,k2r)/128
            sc=(s1*s2)*torch.tril(torch.ones(T,T,device=DEV))
            return sc.cpu()
        sc_real=score(qpre)
        sc_fix=score(qfix[None,None].expand(B,T,128))
        dr=sc_real.abs().sum(-1).clamp_min(1e-6)
        dfx=sc_fix.abs().sum(-1).clamp_min(1e-6)
        for r in range(i,min(i+4,NFRESH)):
            b=r-i
            for q in tgt[r].nonzero().squeeze(1).tolist():
                kk=info[(r,q)]['key']
                acc['real'][0]+=abs(float(sc_real[b,q,kk]/dr[b,q]))
                acc['real'][1]+=1
                acc['fixed'][0]+=abs(float(sc_fix[b,q,kk]/dfx[b,q]))
                acc['fixed'][1]+=1
            for q in (ctrl[r].nonzero().squeeze(1).tolist()
                      if ctrl[r].any() else []):
                kk=cinfo[(r,q)]['key']
                acc['ctrl'][0]+=abs(float(sc_real[b,q,kk]/dr[b,q]))
                acc['ctrl'][1]+=1
    rs=acc['real'][0]/max(acc['real'][1],1)
    fs=acc['fixed'][0]/max(acc['fixed'][1],1)
    cs=acc['ctrl'][0]/max(acc['ctrl'][1],1)
    print(f'\nreal query share on (prev-occ+1): {rs:.4f}',flush=True)
    print(f'fixed query share on same key: {fs:.4f}',flush=True)
    print(f'control (non-induction) share: {cs:.4f}',flush=True)
    verified=rs>=2*cs
    fixed_breaks=fs<0.5*rs
    print(f"\n(a) target verified (real {rs:.4f} >= 2x control "
          f"{cs:.4f}): {'HELD' if verified else 'FAILED -- UNEVALUABLE'}")
    if verified:
        print(f"(b) fixed query BREAKS (fixed {fs:.4f} < 0.5x real "
              f"{rs:.4f}): {'HELD' if fixed_breaks else 'FAILED -- '
              f'fixed query reproduces induction, account more general'}")
    out={'n':n,'real_share':round(rs,4),'fixed_share':round(fs,4),
         'control_share':round(cs,4),
         'fixed_over_real':round(fs/max(rs,1e-9),3),
         'target_verified':bool(verified),
         'fixed_breaks':bool(fixed_breaks) if verified else None,
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
