"""INDUCTION QUERY -- is the induction head's query content-
dependent? (the modality prediction from 572)
572 found head 8.3 does induction: it attends to (prior occurrence
+ 1) to copy what followed a repeated token. Induction is
content-matching -- the query must encode the CURRENT token to
find where it occurred before. So unlike the three structural
heads, whose query is approximately FIXED (559: a fixed query
reproduces them), head 8.3's query should be content-dependent,
and replacing it with a fixed (average) query should BREAK the
induction attention. This is the test that distinguishes
content-induction from fixed-query selection as a mechanism class.
Target verified in 572 (8.3 attends occ+1 at 0.103 for digits,
0.058 general). Here: measure the head's share on (prior
occurrence + 1) with the REAL per-position query and with the
INDUCTION-AVERAGE fixed query, on general repeated-token targets
(not just digits, since 8.3 does general induction).
REGISTERED PREDICTIONS:
  (0) POPULATED: >= 100 repeated-token targets with a clean prior
      occurrence;
  (a) REAL QUERY INDUCES: the real query's share on occ+1 exceeds
      its share on a control (non-induction) key by >= 1.5x;
  (b) THE FIXED QUERY BREAKS: the induction-average fixed query's
      occ+1 share is < 0.5x the real query's. This is the
      distinguishing prediction -- induction needs a content query.
      If the fixed query preserves the induction, the
      content-matching reading is wrong;
  (c) report real and fixed shares;
  NULL: the fixed query applied at NON-induction positions gives
      the same low share -- confirms the fixed query carries no
      position-specific induction signal."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=8; HD=3; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'induction_query_results.json'
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
            if t in last and q-last[t]>=3 and last[t]+1<q:
                info[(r,q)]={'occ':last[t],'occ1':last[t]+1}
                tgt[r,q]=True
            last[t]=q
    n=int(tgt.sum())
    print(f'{n} repeated-token induction targets',flush=True)
    if n<100:
        json.dump({'void':'too few','n':n},open(OUT,'w'),indent=1)
        return
    g=torch.Generator().manual_seed(29)
    at=m.transformer.h[LJ].attn; cap={}
    # PASS 1: induction-average pre-rotary query
    qsum=torch.zeros(128,device=DEV); qn=0
    for i in range(0,NFRESH,4):
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); B=bb.shape[0]
        hc=at.register_forward_pre_hook(
            lambda mo_,a_: cap.__setitem__('X',a_[0]))
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        hc.remove()
        qp=F.rms_norm(at.c_q(cap['X']).view(B,T,NH,128),(128,))[:,:,HD].float()
        for r in range(i,min(i+4,NFRESH)):
            if not tgt[r].any(): continue
            b=r-i
            for q in tgt[r].nonzero().squeeze(1).tolist():
                qsum+=qp[b,q]; qn+=1
    qfix=qsum/max(qn,1)
    acc={'real_occ1':[0.0,0],'real_ctrl':[0.0,0],
         'fix_occ1':[0.0,0],'fix_ctrl':[0.0,0]}
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
        qpre=F.rms_norm(at.c_q(X).view(B,T,NH,128),(128,))[:,:,HD].float()
        def rk(W): return are(F.rms_norm(W(X).view(B,T,NH,128),
                        (128,)),cq,sq)[:,:,HD].float()
        kf=rk(at.c_k); q2r=rk(at.c_q2); k2r=rk(at.c_k2)
        def scores(qp):
            qr=are(qp[:,:,None].expand(B,T,NH,128).contiguous(),
                   cq,sq)[:,:,HD]
            s1=torch.einsum('bqd,bkd->bqk',qr.float(),kf)/128
            s2=torch.einsum('bqd,bkd->bqk',q2r,k2r)/128
            return ((s1*s2)*torch.tril(torch.ones(T,T,device=DEV))).cpu()
        scr=scores(qpre); scf=scores(qfix[None,None].expand(B,T,128))
        dr=scr.abs().sum(-1).clamp_min(1e-6); df=scf.abs().sum(-1).clamp_min(1e-6)
        for r in range(i,min(i+4,NFRESH)):
            b=r-i
            for q in tgt[r].nonzero().squeeze(1).tolist():
                k1=info[(r,q)]['occ1']
                # control key: a random earlier position
                kc=int(torch.randint(0,q,(1,),generator=g)) if q>0 else 0
                acc['real_occ1'][0]+=abs(float(scr[b,q,k1]/dr[b,q])); acc['real_occ1'][1]+=1
                acc['real_ctrl'][0]+=abs(float(scr[b,q,kc]/dr[b,q])); acc['real_ctrl'][1]+=1
                acc['fix_occ1'][0]+=abs(float(scf[b,q,k1]/df[b,q])); acc['fix_occ1'][1]+=1
                acc['fix_ctrl'][0]+=abs(float(scf[b,q,kc]/df[b,q])); acc['fix_ctrl'][1]+=1
    ro1=acc['real_occ1'][0]/max(acc['real_occ1'][1],1)
    rc=acc['real_ctrl'][0]/max(acc['real_ctrl'][1],1)
    fo1=acc['fix_occ1'][0]/max(acc['fix_occ1'][1],1)
    fc=acc['fix_ctrl'][0]/max(acc['fix_ctrl'][1],1)
    print(f'\nreal query: occ+1 {ro1:.4f} | control {rc:.4f}',flush=True)
    print(f'fixed query: occ+1 {fo1:.4f} | control {fc:.4f}',flush=True)
    pa=ro1>=1.5*rc
    pb=fo1<0.5*ro1
    print(f"(a) real query induces (occ+1 {ro1:.4f} >= 1.5x control "
          f"{rc:.4f}): {'HELD' if pa else 'FAILED'}")
    print(f"(b) fixed query BREAKS (fixed {fo1:.4f} < 0.5x real "
          f"{ro1:.4f}): {'HELD -- induction needs content query' if pb else 'FAILED -- fixed query preserves induction'}")
    out={'n':n,'real_occ1':round(ro1,4),'real_control':round(rc,4),
         'fixed_occ1':round(fo1,4),'fixed_control':round(fc,4),
         'fixed_over_real':round(fo1/max(ro1,1e-9),3),
         'pred_a':bool(pa),'pred_b':bool(pb),'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
