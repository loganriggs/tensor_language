"""COVERAGE, NOT CHOICE (rung 28): does the tail-span gain track how much output variance the span covers?

§2121: any 8 of the top-32 PCA directions beat the tail's top-8 span by ~0.2 nat, random as well as metric-picked.
The reading: the tail program replaces the projection of the real MLP output on its span with a class-dictionary
target and passes the rest through, so a span covering less variance intervenes less and costs less. This tests
that reading: six random 8-of-32 spans per tail MLP (a different draw per arm, the same draw across the eight
MLPs), plus the 8 LOWEST-variance of the top-32 (the limit of doing less), plus plain top-8; for each arm record the
variance share the span covers (mean over tail MLPs, from the PCA spectrum) and the observable-energy share it
covers (u^T G u summed, normalised by the top-32 total), and the CE gap on window 1 and the eight fresh windows.

REGISTERED PREDICTIONS (8 arms: plain, lowest-8, six random):
  (a) COVERAGE PRICES THE SPAN: Spearman(gain over plain, covered variance share) <= -0.7 across the 8 arms.
  (b) DOING LESS IS THE LIMIT: the lowest-8 span gains >= the median random gain on window 1.
  (c) THE METRIC DOES NOT SEE MORE: |Spearman(gain, observable-energy share)| is within 0.1 of |Spearman(gain,
      variance share)| -- the two coverages predict equally (the metric adds nothing, §2121).
  (d) REPRODUCTION GATE: cfgE block-6 rel-MSE 1.7415 within 0.10.

ORIGINAL HEADERS FOLLOW.


§2120: constructing each tail MLP's rank-8 span in the metric-whitened space hurts on every window (-0.05 to -0.14);
§2105: constructing residual bases under the metric at blocks 1-4 was worthless (-0.009). Selecting mlp4/mlp5 units
under the same metric is certified (+0.08 median, 8/8 and 7/8 windows). Hypothesis: the metric is a good SELECTOR
among fixed directions and a bad CONSTRUCTOR of new ones. Test: keep the plain PCA's top-32 directions of each tail
MLP's output (data-defined, estimated from the same energy as before) and select the 8 with the largest metric
weight u^T G u. Control: a random 8 of the 32.

ARMS: cfgE (plain top-8) | select-8-of-32 by metric | random-8-of-32.  Window 1 + eight fresh pile windows.

REGISTERED PREDICTIONS:
  (a) NO HARM: selected-8 gap on window 1 <= plain gap + 0.01.
  (b) GAIN: median gain of selected-8 over cfgE across the eight fresh windows >= +0.02.
  (c) THE SELECTION IS NOT FREE: random-8-of-32 is worse than plain top-8 by >= 0.02 on window 1.
  (d) REPRODUCTION GATE: cfgE block-6 rel-MSE 1.7415 within 0.10.
If (a) fails the distinction is model-defined vs data-defined directions, not select vs construct.

ORIGINAL HEADERS FOLLOW.


§2117: the tail spans tie for the largest single lever of the certified arm (0.158 of the gap when made real). The
spans are the rank-8 PCA basis of each tail MLP's real output (blocks 10-17), fitted on FW rows 0:120; the tail program
replaces the output's projection on that span with a class-dictionary target. §2105 found metric-chosen residual
bases at blocks 1-4 worthless (-0.009) where the loss-gradient metrics are tiny (top eigenvalue 5e-9 .. 4e-8); at
blocks 11-18 the metrics are 100-1000x larger (§2104: 7e-6 at block 6 and rising; depth profile). This chooses each
span as the top-8 directions of the metric-whitened output covariance, mapped back and orthonormalised (the tail hook
needs an orthonormal Q), with a random-metric control; mlp4/mlp5 units stay NORM-selected so the span knob is isolated.

ARMS (each a full matched-context rebuild): cfgE (plain spans) | metric-spans | random-spans.  CE on window 1 (R0:R1)
and on §2083's eight fresh pile-10k windows (FW rows 0:120 are NOT a test window here: they fit the spans).

REGISTERED PREDICTIONS:
  (a) metric spans beat cfgE by >= 0.05 nat on window 1 -- a third of the tail's 0.158 oracle recovery.
  (b) random-metric spans gain <= 0.02 on window 1.
  (c) TRANSFER: median gain of metric spans over the eight fresh windows >= 0.025 (half of (a)'s bar).
  (d) REPRODUCTION GATE: cfgE block-6 rel-MSE 1.7415 within 0.10.

ORIGINAL HEADER OF THE PARENT SCRIPT FOLLOWS.


BENCHMARK_BACKLOG rung 12. S2101: the certified arm's block-6 error is anti-random and its cost sits in the ~600-dim
first-order observable subspace. S2102: no single front piece is the lever and block-6 rel-MSE does not price CE
(rho 0.07) -- direction does. S2103: oracle-correcting ONLY the observable projection of the block-6 stream recovers
94.5% of what full correction recovers. So the front should spend its constrained capacity on what the loss reads.
A metric does not move an unconstrained least-squares solution (tables, the m1 linear map), so only the constrained
pieces change: the 64-dim residual bases of the m0/m2/m3 table+residual programs, and the 2,304-unit selections of
mlp4/mlp5. Stored values are IDENTICAL to cfgE's by construction.

METRIC. For a piece writing into the stream entering block k, M_k = first-order observability Gramian at block k's
input on the FIT rows (CA:CB), floored at 1e-3 x its top eigenvalue so its square root is invertible (m0 -> k=1,
m2 -> 3, m3 -> 4, mlp4 -> 5, mlp5 -> 6). Residual basis: PCA of the residual in the M_k-whitened space (top 64), the
deployed map = tb + (features @ A) @ U^T @ M_k^{-1/2}. Unit selection: importance ||M_k^{1/2} Down[:,u]|| x ||L_u|| x
||R_u|| instead of ||Down[:,u]|| x ... . Control: the same construction with a RANDOM orthogonal rotation of M_k's
eigenbasis (same spectrum, unrelated directions).

ARMS (full matched-context sequential rebuild each; CE on the S2086 evaluation rows R0:R1; rel-MSE profile; block-6
error energy inside the eval-row observable subspace):  cfgE (plain) | metric | random-metric

REGISTERED PREDICTIONS:
  (a) THE METRIC ARM BEATS cfgE BY >= 0.15 nat -- half of S2102's best single-piece-real gain (0.30, m2-real), at ZERO
      extra stored values. If FALSE, first-order observability is the wrong weight for fitting even though it is the
      right subspace for oracle correction (S2103) -- i.e. the fits cannot reach the subspace with this capacity.
  (b) IT MOVES THE RIGHT ENERGY: the metric arm's block-6 error energy inside the observable subspace drops by >= 20%
      relative to cfgE, whatever happens to total rel-MSE. If (a) holds and (b) fails, the gain came from somewhere
      other than the registered mechanism and is reported as unexplained.
  (c) THE CONTROL DOES NOT: the random-metric arm gains <= 0.05 nat over cfgE. If FALSE, any re-weighting helps and the
      observability content of the metric is not what matters.
  (d) REPRODUCTION GATE (LESSON 42): cfgE's block-6 rel-MSE reproduces S2086's 1.7415 within 0.10.

Self-reviewed. Writes metric_front_refit_results.json.
"""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import os

# PLAN PRE-FLIGHT (LESSON 109): bilin18_joint_removal loads the model at import.
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['stream_error_profile_results.json','metric_tail_select_results.json']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    _p=json.load(open(_bq+'stream_error_profile_results.json'))['profile']
    print(f"DRYRUN OK: S2086 present (block 6 {_p['6']}); eight arms (plain / lowest-8 / six random 8-of-32) x 9 windows, with coverage")
    raise SystemExit(0)
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import os

# PLAN PRE-FLIGHT (LESSON 109): bilin18_joint_removal loads the model at import.
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    if not os.path.exists(_bq+'stream_fidelity_results.json'):
        print('DRYRUN FAIL: S309 stream_fidelity_results.json absent'); raise SystemExit(1)
    _p=json.load(open(_bq+'stream_fidelity_results.json'))['per_layer']
    print(f"DRYRUN OK: S309 present (block7 emp {_p['7']['emp']}, block14 emp "
          f"{_p['14']['emp']}); profiling all 18 block inputs on the empirical arm")
    raise SystemExit(0)

import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from circuit_dictionary import classify, COMPS as TAILC, CLS
D=1152; V=50257
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'tail_span_coverage_results.json'
CA,CB=300,512; R0,R1=120,300
CONSTN={'digit','bclose','sentend','comma','name','rep'}
CONSTK=[k for k,nm in enumerate(CLS) if nm in CONSTN]
LINK=[k for k in range(10) if k not in CONSTK]
ATTM=[2,3,4,5,6,7,8,9]; ATTT=[10,11,12,13,14,15,16,17]
MIDL=(4,5,6,7,8,9)
TRI=torch.triu_indices(32,32)

@torch.no_grad()
def main():
    t0=time.time()
    S={}; cur={}
    vdir={}
    for li in (0,2,3):
        mlp=m.transformer.h[li].mlp
        _,_,Vh=torch.linalg.svd(
            torch.cat([mlp.Left.weight.detach().float(),
                       mlp.Right.weight.detach().float()]),
            full_matrices=False)
        vdir[li]=Vh[:32].T.contiguous()
    def quadfeat(X,li):
        Z=X@vdir[li]
        iu,il=TRI
        return torch.cat([X,Z[:,iu]*Z[:,il]],1)
    tab=torch.zeros(V,D,device=DEV,dtype=torch.float16)
    for s0 in range(0,V,4096):
        idx=torch.arange(s0,min(s0+4096,V),device=DEV)[:,None]
        x=F.rms_norm(m.transformer.wte(idx),(D,))
        hc=F.rms_norm((m.transformer.h[0].lambdas[0]
                       +m.transformer.h[0].lambdas[1])*x,(D,))
        tab[s0:s0+idx.shape[0]]=m.transformer.h[0].attn.c_v(hc)[:,0]\
            .to(torch.float16)
    S['a0']=('cv',0,tab)
    spans={}
    for li in TAILC:
        accs=[]
        for i in range(0,120,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc)
            accs.append(acc[0])
        Y=torch.cat(accs); Yb=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Yb).float(), full_matrices=False)
        spans[li]=(orth(Vh[:8].T),Yb.float())
    clsA=classify(CA,CB).to(DEV).reshape(CB-CA,256)
    clsC=classify(R0,R1).to(DEV).reshape(R1-R0,256)
    flatA=clsA.reshape(-1)
    for li in MIDL:
        mlp=m.transformer.h[li].mlp
        L=mlp.Left.weight.detach().float()
        Rw=mlp.Right.weight.detach().float()
        Dw=mlp.Down.weight.detach().float()
        db=mlp.Down_bias.detach().float()
        imp=Dw.norm(dim=0)*L.norm(dim=1)*Rw.norm(dim=1)
        keep=imp.argsort(descending=True)[:2304]
        S[f'c{li}']=('cp',li,L[keep].contiguous(),Rw[keep].contiguous(),
                     Dw[:,keep].contiguous(),db)
    def install(active):
        alist=[nm for nm in active if S[nm][0]=='attnd']
        hs=[]
        for nm in active:
            kind=S[nm][0]
            if kind=='cv':
                _,li,tb=S[nm]
                def h(mod,i_,o_,tb=tb):
                    return tb[cur['idx']].to(o_.dtype)
                hs.append(m.transformer.h[li].attn.c_v
                          .register_forward_hook(h))
            elif kind=='tableres':
                _,li,tb,A,P=S[nm]
                def h(mod,i_,o_,tb=tb,A=A,P=P,li=li):
                    x=i_[0].float().reshape(-1,D)
                    ft=quadfeat(x,li)
                    new=tb[cur['idx']].float().reshape(-1,D)+(ft@A)@P.T
                    return new.view(o_.shape).to(o_.dtype)
                hs.append(m.transformer.h[li].mlp
                          .register_forward_hook(h))
            elif kind=='linear':
                _,li,W,b=S[nm]
                def h(mod,i_,o_,W=W,b=b):
                    x=i_[0].float().reshape(-1,D)
                    return (x@W+b).view(o_.shape).to(o_.dtype)
                hs.append(m.transformer.h[li].mlp
                          .register_forward_hook(h))
            elif kind=='cp':
                _,li,Lk,Rk,Dk,db=S[nm]
                def h(mod,i_,o_,Lk=Lk,Rk=Rk,Dk=Dk,db=db):
                    x=i_[0].float()
                    return (((x@Lk.T)*(x@Rk.T))@Dk.T+db).to(o_.dtype)
                hs.append(m.transformer.h[li].mlp
                          .register_forward_hook(h))
            elif kind=='attnd':
                _,li,CV,LW,Wp2=S[nm]
                first=(len(alist)>0 and nm==alist[0])
                def h(mod,i_,o_,CV=CV,LW=LW,Wp2=Wp2,first=first):
                    y,v1=o_
                    x=i_[0].float().reshape(-1,D)
                    if first and cur['mode']=='probe':
                        cur['lab']=(x@Wp2).argmax(1)
                    c=cur['lab']
                    new=CV[c].clone()
                    for k in LINK:
                        sel=c==k
                        if sel.any(): new[sel]=x[sel]@LW[k]
                    return (new.view(y.shape).to(y.dtype),v1)
                hs.append(m.transformer.h[li].attn
                          .register_forward_hook(h))
            elif kind=='attnz':
                _,li=S[nm]
                def h(mod,i_,o_):
                    y,v1=o_
                    return (torch.zeros_like(y),v1)
                hs.append(m.transformer.h[li].attn
                          .register_forward_hook(h))
            elif kind=='attnm':
                _,li,mu=S[nm]
                def h(mod,i_,o_,mu=mu):
                    y,v1=o_
                    return (mu.expand_as(y).to(y.dtype),v1)
                hs.append(m.transformer.h[li].attn
                          .register_forward_hook(h))
            elif kind=='tail':
                _,Wp,DICT,LIN=S[nm]
                for li in TAILC:
                    Q,_=spans[li]
                    def h(mod,i_,o_,li=li,Q=Q,first=(li==TAILC[0]),
                          Wp=Wp,DICT=DICT,LIN=LIN):
                        B,T,_=o_.shape
                        x=i_[0].float().reshape(-1,D)
                        c=o_.float().reshape(-1,D)@Q
                        if first: cur['kk']=(x@Wp).argmax(1)
                        kk=cur['kk']
                        tgt=DICT[li][kk].clone()
                        for k in (8,9):
                            sel=kk==k
                            if sel.any(): tgt[sel]=x[sel]@LIN[li][k]
                        delta=((c-tgt)@Q.T).view(B,T,D)
                        return o_-delta.to(o_.dtype)
                    hs.append(m.transformer.h[li].mlp
                              .register_forward_hook(h))
        return hs
    def runA(active, cap_mod):
        Ys=[]; Xs=[]; Ids=[]
        hs=install(active)
        def capf(mod,i_,o_):
            Ys.append((o_[0] if isinstance(o_,tuple) else o_)
                      .detach().reshape(-1,D).float())
            Xs.append(i_[0].detach().reshape(-1,D).float())
        hs.append(cap_mod.register_forward_hook(capf))
        for i in range(CA,CB,4):
            bb=FW[i:i+4,:257].to(DEV)
            cur['idx']=bb[:,:-1].contiguous()
            cur['mode']='oracle'
            cur['lab']=clsA[i-CA:i-CA+4].reshape(-1)
            m(cur['idx'], bb[:,1:].contiguous())
            Ids.append(cur['idx'].reshape(-1))
        for h in hs: h.remove()
        return torch.cat(Ys),torch.cat(Xs),torch.cat(Ids)
    def fit_table(Y,ids):
        sums=torch.zeros(V,D,device=DEV); cnt=torch.zeros(V,device=DEV)
        cnt.index_add_(0,ids,torch.ones_like(ids,dtype=torch.float))
        sums.index_add_(0,ids,Y)
        t=sums/cnt.clamp_min(1)[:,None]; t[cnt==0]=Y.mean(0)
        return t.to(torch.float16)
    def fit_tableres(Y,X,ids,li):
        tb=fit_table(Y,ids)
        Rr=Y-tb[ids].float()
        _,_,Vh=torch.linalg.svd(Rr[:30000], full_matrices=False)
        P=orth(Vh[:64].T)
        ft=quadfeat(X,li)
        lam=1e-2*len(X)
        A=torch.linalg.solve(ft.T@ft+lam*torch.eye(ft.shape[1],device=DEV),
                             ft.T@(Rr@P))
        return ('tableres',li,tb,A,P)
    Yoh=torch.zeros(len(flatA),10,device=DEV)
    Yoh[torch.arange(len(flatA)),flatA]=1.0
    Wp2=None
    def fit_attnd(li,active):
        nonlocal Wp2
        Y,X,ids=runA(active,m.transformer.h[li].attn)
        if Wp2 is None:
            lam=1e-2*len(X)
            Wp2=torch.linalg.solve(X.T@X+lam*torch.eye(D,device=DEV),
                                   X.T@Yoh)
            acc=float(((X@Wp2).argmax(1)==flatA).float().mean())
            print(f'a{li}-input probe acc {acc:.2f}',flush=True)
        CV=torch.stack([Y[flatA==k].mean(0) if (flatA==k).sum()>0
                        else Y.mean(0) for k in range(10)])
        LW={}
        for k in LINK:
            mk=flatA==k
            Xk=X[mk]; Yk=Y[mk]
            l2=1e-2*max(len(Xk),1)
            LW[k]=torch.linalg.solve(Xk.T@Xk+l2*torch.eye(D,device=DEV),
                                     Xk.T@Yk)
        print(f'fit a{li}',flush=True)
        return ('attnd',li,CV,LW,Wp2)
    order=['a0']
    Y,X,ids=runA(order,m.transformer.h[0].mlp)
    S['m0']=fit_tableres(Y,X,ids,0); order.append('m0')
    Y,X,ids=runA(order,m.transformer.h[1].attn.c_v)
    S['a1v']=('cv',1,fit_table(Y,ids)); order.append('a1v')
    Y,X,ids=runA(order,m.transformer.h[1].mlp)
    lam=1e-2*len(X)
    W1=torch.linalg.solve(X.T@X+lam*torch.eye(D,device=DEV),X.T@Y)
    b1=Y.mean(0)-X.mean(0)@W1
    S['m1']=('linear',1,W1,b1); order.append('m1')
    S['a2']=fit_attnd(2,order); order.append('a2')
    Y,X,ids=runA(order,m.transformer.h[2].mlp)
    S['m2']=fit_tableres(Y,X,ids,2); order.append('m2')
    S['a3']=fit_attnd(3,order); order.append('a3')
    Y,X,ids=runA(order,m.transformer.h[3].mlp)
    S['m3']=fit_tableres(Y,X,ids,3); order.append('m3')
    for li in MIDL:
        S[f'a{li}']=fit_attnd(li,order)
        order.append(f'a{li}'); order.append(f'c{li}')
    for li in ATTT:
        S[f'a{li}']=fit_attnd(li,order); order.append(f'a{li}')
    capsT={li:[] for li in TAILC}; capsI={li:[] for li in TAILC}
    hs=install(order)
    for li in TAILC:
        def mk(li=li):
            def h(mod,i_,o_):
                capsT[li].append(o_.detach().reshape(-1,D).float())
                capsI[li].append(i_[0].detach().reshape(-1,D).float())
            return h
        hs.append(m.transformer.h[li].mlp.register_forward_hook(mk()))
    for i in range(CA,CB,4):
        bb=FW[i:i+4,:257].to(DEV)
        cur['idx']=bb[:,:-1].contiguous()
        cur['mode']='oracle'; cur['lab']=clsA[i-CA:i-CA+4].reshape(-1)
        m(cur['idx'], bb[:,1:].contiguous())
    for h in hs: h.remove()
    X10=torch.cat(capsI[TAILC[0]])
    lam=1e-2*len(X10)
    Wp=torch.linalg.solve(X10.T@X10+lam*torch.eye(D,device=DEV),X10.T@Yoh)
    DICT={}; LIN={}
    for li in TAILC:
        Q,_=spans[li]; C=torch.cat(capsT[li])@Q
        DICT[li]=torch.stack([C[flatA==k].mean(0) if (flatA==k).sum()>0
                              else C.mean(0) for k in range(10)])
        Xl=torch.cat(capsI[li]); LIN[li]={}
        for k in (8,9):
            mk_=flatA==k
            Xk=Xl[mk_]; Ck=C[mk_]
            l2=1e-2*len(Xk)
            LIN[li][k]=torch.linalg.solve(Xk.T@Xk+l2*torch.eye(D,device=DEV),
                                          Xk.T@Ck)
        capsT[li]=None; capsI[li]=None
    S['tail']=('tail',Wp,DICT,LIN); order.append('tail')
    def evalC(active, mode):
        hs=install(active)
        ces=[]
        for i in range(R0,R1,4):
            bb=FW[i:i+4,:257].to(DEV)
            cur['idx']=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            cur['mode']=mode
            if mode=='oracle':
                cur['lab']=clsC[i-R0:i-R0+4].reshape(-1)
            x=F.rms_norm(m.transformer.wte(cur['idx']),(D,)); x0=x; v1=None
            for blk in m.transformer.h:
                x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        for h in hs: h.remove()
        return float(torch.cat(ces).mean())
    # ---- fold tables (weights-only) ----
    FOLD={}
    capsF={0:[],2:[],3:[]}
    hsF=[]
    for li in (0,2,3):
        def mkf(li=li):
            def h(mo,i_,o_):
                capsF[li].append(o_.detach()[:,0].float())
            return h
        hsF.append(m.transformer.h[li].mlp.register_forward_hook(mkf()))
    for s0 in range(0,V,2048):
        idx=torch.arange(s0,min(s0+2048,V),device=DEV)[:,None]
        xF=F.rms_norm(m.transformer.wte(idx),(D,)); x0F=xF; v1F=None
        for blk in m.transformer.h[:4]:
            xF,v1F=blk(xF,v1F,x0F)
    for h in hsF: h.remove()
    for li in (0,2,3):
        FOLD[li]=torch.cat(capsF[li]).to(torch.float16); capsF[li]=None
    # ---- CP middles (weights-only) ----
    for li in (4,5,6,7,8,9):
        mlp=m.transformer.h[li].mlp
        L=mlp.Left.weight.detach().float()
        Rw=mlp.Right.weight.detach().float()
        Dw=mlp.Down.weight.detach().float()
        db=mlp.Down_bias.detach().float()
        imp=Dw.norm(dim=0)*L.norm(dim=1)*Rw.norm(dim=1)
        keep=imp.argsort(descending=True)[:2304]
        S[f'c{li}']=('cp',li,L[keep].contiguous(),Rw[keep].contiguous(),
                     Dw[:,keep].contiguous(),db)
    # ---- matched-context sequential fits, both arms ----

    # ---- site-local observability metrics on the FIT rows ----
    NRF=len(range(CA,CB,4))*4; TT=256; SKIP=64
    TOKF=torch.cat([FW[i:i+4,:257] for i in range(CA,CB,4)]).to(DEV)
    TOKE=torch.cat([FW[i:i+4,:257] for i in range(R0,R1,8)]).to(DEV)
    def gramian(TOKS,site):
        G=torch.zeros(D,D,device=DEV,dtype=torch.float64); n=0
        for b0 in range(0,TOKS.shape[0],4):
            idx=TOKS[b0:b0+4,:-1]; tg=TOKS[b0:b0+4,1:].reshape(-1)
            with torch.enable_grad():
                x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None; leaf=None
                for li,blk in enumerate(m.transformer.h):
                    if li==site:
                        x=x.detach().requires_grad_(True); leaf=x
                    x,v1=blk(x,v1,x0)
                if site==len(m.transformer.h):
                    x=x.detach().requires_grad_(True); leaf=x
                lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
                ce=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,reduction='none').view(idx.shape[0],TT)
                ce[:,SKIP:].sum().backward()
            g=leaf.grad[:,SKIP:].reshape(-1,D).double(); G+=g.T@g; n+=g.shape[0]
        return G/n
    SITE_OF={f't{li}':li+1 for li in TAILC}
    gen=torch.Generator(device='cpu').manual_seed(12)
    MET={}
    for piece,site in SITE_OF.items():
        G=gramian(TOKF,site); e,Q=torch.linalg.eigh(G); e=e.clamp_min(1e-3*float(e.max()))
        Qr=torch.linalg.qr(torch.randn(D,D,generator=gen,dtype=torch.float64))[0].to(DEV)
        for mode,QQ in (('metric',Q),('random-metric',Qr)):
            half=(QQ*e.sqrt()[None,:])@QQ.T; inv=(QQ*(1/e.sqrt())[None,:])@QQ.T
            MET[(mode,piece)]=(half.float(),inv.float())
        print(f'metric for {piece} at block {site}: top eig {float(e.max()):.3e}, floor {float(e.min()):.3e}',flush=True)
    SPANS={'plain':{li:spans[li] for li in TAILC}}
    genS=torch.Generator(device='cpu').manual_seed(28)
    COVER={'plain':{'var':[],'obs':[]}}
    MODES=['lowest8']+[f'rand{k}' for k in range(6)]
    PICKS={m_:torch.randperm(32,generator=genS)[:8] for m_ in MODES if m_.startswith('rand')}
    for mode in MODES:
        SPANS[mode]={}; COVER[mode]={'var':[],'obs':[]}
        for li in TAILC:
            accs=[]
            for i in range(0,120,6):
                acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc); accs.append(acc[0])
            Y=torch.cat(accs).float(); Yb=Y.mean(0)
            _,_,Vh=torch.linalg.svd((Y-Yb), full_matrices=False)
            U_,S_,Vh=torch.linalg.svd((Y-Yb), full_matrices=False)
            V32=Vh[:32].T.contiguous()
            half,_=MET[('metric',f't{li}')]
            w=((half@V32)**2).sum(0)
            var=(S_[:32]**2); tot=float((S_**2).sum())
            pick=(torch.arange(24,32) if mode=='lowest8' else PICKS[mode]).to(DEV)
            Q=orth(V32[:,pick].contiguous())
            SPANS[mode][li]=(Q,Yb)
            COVER[mode]['var'].append(float(var[pick].sum()/tot)); COVER[mode]['obs'].append(float(w[pick].sum()/w.sum()))
            if mode=='lowest8':
                COVER['plain']['var'].append(float(var[:8].sum()/tot)); COVER['plain']['obs'].append(float(w[:8].sum()/w.sum()))
            ov=float(((SPANS['plain'][li][0].T@Q)**2).sum()/8)
            print(f'  tail span mlp{li} ({mode}): overlap with plain span {ov:.3f}',flush=True)
    def set_spans(mode):
        for li in TAILC: spans[li]=SPANS[mode][li]
    # ---- arms ----
    def build_arm(mode):
        def fit_res(li,piece,active):
            Y,X,ids=runA(active,m.transformer.h[li].mlp)
            tb=fit_table(Y,ids)
            Rr=Y-tb[ids].float()
            ft=quadfeat(X,li); lam=1e-2*len(X)
            if True:
                _,_,Vh2=torch.linalg.svd(Rr[:30000],full_matrices=False)
                P=orth(Vh2[:64].T)
                A=torch.linalg.solve(ft.T@ft+lam*torch.eye(ft.shape[1],device=DEV),ft.T@(Rr@P))
                return ('tableres',li,tb,A,P)
            half,inv=MET[(mode,piece)]
            Z=Rr@half
            _,_,Vh2=torch.linalg.svd(Z[:30000],full_matrices=False)
            U=orth(Vh2[:64].T)
            A=torch.linalg.solve(ft.T@ft+lam*torch.eye(ft.shape[1],device=DEV),ft.T@(Z@U))
            # deployed: tb + (ft@A)@U^T@inv   == the same shape as P.T with P := (U^T inv)^T, i.e. 64 x D stored
            Pd=(U.T@inv).T.contiguous()
            return ('tableres',li,tb,A,Pd)
        def cp_piece(li,piece):
            mlp=m.transformer.h[li].mlp
            L=mlp.Left.weight.detach().float(); Rw=mlp.Right.weight.detach().float()
            Dw=mlp.Down.weight.detach().float(); db=mlp.Down_bias.detach().float()
            if True:
                imp=Dw.norm(dim=0)*L.norm(dim=1)*Rw.norm(dim=1)
            else:
                half,_=MET[(mode,piece)]
                imp=(half@Dw).norm(dim=0)*L.norm(dim=1)*Rw.norm(dim=1)
            keep=imp.argsort(descending=True)[:2304]
            return ('cp',li,L[keep].contiguous(),Rw[keep].contiguous(),Dw[:,keep].contiguous(),db)
        tag={'plain':'E','lowest8':'L','rand0':'0','rand1':'1','rand2':'2','rand3':'3','rand4':'4','rand5':'5'}[mode]
        act=['a0']
        S[f'm0{tag}']=fit_res(0,'m0',act); act.append(f'm0{tag}')
        act.append('a1v'); act.append('m1')
        S[f'm2{tag}']=fit_res(2,'m2',act); act.append(f'm2{tag}')
        S[f'm3{tag}']=fit_res(3,'m3',act); act.append(f'm3{tag}')
        S[f'c4{tag}']=cp_piece(4,'c4'); S[f'c5{tag}']=cp_piece(5,'c5')
        stack=act+[f'c4{tag}',f'c5{tag}']+[f'c{li}' for li in (6,7,8,9)]
        capsT={li:[] for li in TAILC}; capsI={li:[] for li in TAILC}
        hs=install(stack)
        for li in TAILC:
            def mk2(li=li):
                def h(mo,i_,o_):
                    capsT[li].append(o_.detach().reshape(-1,D).float())
                    capsI[li].append(i_[0].detach().reshape(-1,D).float())
                return h
            hs.append(m.transformer.h[li].mlp.register_forward_hook(mk2()))
        for i in range(CA,CB,4):
            bb=FW[i:i+4,:257].to(DEV)
            cur['idx']=bb[:,:-1].contiguous(); cur['mode']='oracle'
            cur['lab']=clsA.reshape(CB-CA,256)[i-CA:i-CA+4].reshape(-1)
            m(cur['idx'], bb[:,1:].contiguous())
        for h in hs: h.remove()
        X10=torch.cat(capsI[TAILC[0]])
        Yoh2=torch.zeros(len(flatA),10,device=DEV); Yoh2[torch.arange(len(flatA)),flatA]=1.0
        lam=1e-2*len(X10)
        Wp=torch.linalg.solve(X10.T@X10+lam*torch.eye(D,device=DEV),X10.T@Yoh2)
        DICT={}; LIN={}
        for li in TAILC:
            Q,_=spans[li]; C=torch.cat(capsT[li])@Q
            DICT[li]=torch.stack([C[flatA==k].mean(0) if (flatA==k).sum()>0 else C.mean(0) for k in range(10)])
            Xl=torch.cat(capsI[li]); LIN[li]={}
            for k in (8,9):
                mk_=flatA==k; Xk=Xl[mk_]; Ck=C[mk_]; l2=1e-2*len(Xk)
                LIN[li][k]=torch.linalg.solve(Xk.T@Xk+l2*torch.eye(D,device=DEV),Xk.T@Ck)
            capsT[li]=None; capsI[li]=None
        S[f'tail{tag}']=('tail',Wp,DICT,LIN)
        return stack+[f'tail{tag}']
    def evalT(TOK,N,active):
        hs=install(active)
        ces=[]
        for i in range(0,N,4):
            bb=TOK[i:i+4,:257].to(DEV)
            cur['idx']=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            cur['mode']='oracle'
            x=F.rms_norm(m.transformer.wte(cur['idx']),(D,))
            x0=x; v1=None
            for blk in m.transformer.h:
                x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        for h in hs: h.remove()
        return float(torch.cat(ces).mean())
    # capture real streams at probe layers
    PROBES=tuple(range(18))
    def collect(active):
        caps={li:[] for li in PROBES}
        hs=install(active)
        for li in PROBES:
            def mk3(li=li):
                def h(mo,args):
                    caps[li].append(args[0].detach().float().reshape(-1,D))
                return h
            hs.append(m.transformer.h[li].register_forward_pre_hook(mk3()))
        for i in range(R0,R1,8):
            bb=FW[i:i+4,:257].to(DEV)
            cur['idx']=bb[:,:-1].contiguous()
            cur['mode']='oracle'
            m(cur['idx'], bb[:,1:].contiguous())
        for h in hs: h.remove()
        return {li:torch.cat(v) for li,v in caps.items()}

    real=collect([])
    # ---- the eight document-disjoint pile-10k windows, built exactly as ops/probe_gate7.py builds them ----
    import tiktoken
    from datasets import load_dataset
    enc3=tiktoken.get_encoding('gpt2')
    dsf=load_dataset('NeelNanda/pile-10k',split='train')
    seen={tuple(FW[r,:32].tolist()) for r in range(FW.shape[0])}
    def take_window(start_di, banned):
        rws=[]; used=set(); last=start_di
        for di in range(start_di,10000):
            if di in banned: continue
            tk=enc3.encode_ordinary(dsf[di]['text'])
            for st0 in range(0,len(tk)-513,513):
                row=tk[st0:st0+513]
                if tuple(row[:32]) in seen: continue
                rws.append(row); used.add(di)
                if len(rws)>=120: break
            last=di
            if len(rws)>=120: break
        assert len(rws)==120, f'window short at di {start_di}: {len(rws)}'
        return rws, used, last
    NWIN=8; WIN=[]; banned=set(); nxt=3000
    for wi in range(NWIN):
        rws,used,last=take_window(nxt,banned)
        WIN.append({'tok':torch.tensor(rws,dtype=torch.long),'docs':len(used),'keys':{tuple(r[:32]) for r in rws}})
        banned|=used; nxt=last+1
    for i in range(NWIN):
        for j in range(i+1,NWIN):
            assert not (WIN[i]['keys'] & WIN[j]['keys']), f'windows {i},{j} share a row'
        del WIN[i]['keys']
    print(f'{NWIN} windows x 120 rows, document-disjoint; docs per window {[w["docs"] for w in WIN]}',flush=True)
    W1=FW[R0:R1]
    base1=evalT(W1,R1-R0,[]); baseW=[evalT(w['tok'],120,[]) for w in WIN]
    res={}
    for mode in ['plain']+MODES:
        ts=time.time(); set_spans(mode)
        cfg=build_arm(mode)
        ce1=evalT(W1,R1-R0,cfg)
        gw=[evalT(w['tok'],120,cfg)-b for w,b in zip(WIN,baseW)]
        st=collect(cfg)
        prof=[float(((st[li]-real[li])**2).mean(0).sum())/max(float(real[li].var(0).sum()),1e-12) for li in PROBES]
        del st; torch.cuda.empty_cache()
        res[mode]={'gap_w1':round(ce1-base1,5),'gaps_fresh':[round(g,5) for g in gw],'profile':[round(p,4) for p in prof]}
        print(f"{mode:13s}: gap w1 {ce1-base1:+.4f} | fresh {[round(g,3) for g in gw]} | b6 {prof[6]:.3f} b17 {prof[17]:.3f} | {time.time()-ts:.0f}s",flush=True)
    set_spans('plain')
    import statistics as stt
    E=res['plain']
    arms=['plain']+MODES
    gain={k:E['gap_w1']-res[k]['gap_w1'] for k in arms}
    cov_var={k:stt.mean(COVER[k]['var']) for k in arms}; cov_obs={k:stt.mean(COVER[k]['obs']) for k in arms}
    def spearman(x,y):
        rx=torch.tensor(x).argsort().argsort().double(); ry=torch.tensor(y).argsort().argsort().double()
        rx,ry=rx-rx.mean(),ry-ry.mean(); return float((rx*ry).sum()/(rx.norm()*ry.norm()).clamp_min(1e-12))
    r_var=spearman([gain[k] for k in arms],[cov_var[k] for k in arms]); r_obs=spearman([gain[k] for k in arms],[cov_obs[k] for k in arms])
    rand_med=stt.median([gain[k] for k in MODES if k.startswith('rand')])
    pa=r_var<=-0.7; pb=gain['lowest8']>=rand_med; pc=abs(abs(r_obs)-abs(r_var))<=0.1; pd=abs(E['profile'][6]-1.7415)<=0.10
    for k in arms: print(f"{k:8s}: gain w1 {gain[k]:+.4f} | var share {cov_var[k]:.4f} | obs share {cov_obs[k]:.4f} | fresh median gap {stt.median(res[k]['gaps_fresh']):.4f}")
    out={'base_ce_w1':round(base1,5),'arms':res,'gain_w1':{k:round(v,4) for k,v in gain.items()},'variance_share':{k:round(v,4) for k,v in cov_var.items()},
         'observable_share':{k:round(v,4) for k,v in cov_obs.items()},'spearman_gain_vs_variance_share':round(r_var,4),
         'spearman_gain_vs_observable_share':round(r_obs,4),'random_gain_median':round(rand_med,4),
         'pred_a_coverage_prices_the_span':bool(pa),'pred_b_lowest8_is_the_limit':bool(pb),'pred_c_metric_sees_no_more':bool(pc),
         'pred_d_reproduces_S2086':bool(pd),'self_reviewed':True,'runtime_s':round(time.time()-t0,1)}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f"(d) cfgE b6 {E['profile'][6]:.4f}: {'HELD' if pd else 'FAILED'}")
    print(f"(a) rho(gain, variance share) {r_var:+.3f} <= -0.7: {'HELD' if pa else 'FAILED'}")
    print(f"(b) lowest-8 gain {gain['lowest8']:+.4f} >= random median {rand_med:+.4f}: {'HELD' if pb else 'FAILED'}")
    print(f"(c) |rho obs| {abs(r_obs):.3f} within 0.1 of |rho var| {abs(r_var):.3f}: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-t0:.0f}s)')

if __name__=='__main__': main()
