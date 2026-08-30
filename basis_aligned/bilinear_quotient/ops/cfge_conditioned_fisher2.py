"""RNG-MATCHED cfgE CONDITIONING + LABEL BOUND (rung 38; licensing rung 36 properly).

S2130: rung 36 saw conditioning buy +0.0123 median on cfgE (7/8) vs +0.047 on the frontier - the S2128
reader-mechanism direction - but its reproduction gate missed by 0.0001 over tol because restricting SITE_OF
moved the shared genF RNG stream (instrument fact: ~0.01 MC-sample sensitivity at 2 samples/position). This rerun
restores the FULL rung-29 SITE_OF so the c4/c5 Gramians consume the stream exactly as S2124 did, and adds a
random-label arm to the window-0 instrument check (S2130 found stale vs classified labels identical to 4dp).

REGISTERED PREDICTIONS (arm-named):
  (a) CONDITIONING INSTALLS IN cfgE TOO: median over the 8 windows of [gap(top8) - gap(cond8)] >= +0.02.
      If FAILED with (c) HELD, the licensed conclusion is: conditioning matters in proportion to how much the
      deployed readers differ from the real ones - cfgE (real readers) sits at ~1/4 of the frontier gain.
  (b) NOT ONE WINDOW: the conditioning gain is >= 0 on >= 6 of 8 windows.
  (c) REPRODUCTION (stream matched): the top8 arm reproduces S2124 median +0.0857 within 0.01.
  (d) LABELS MATTER: RANDOM attnd class labels move the window-0 plain gap by >= 0.05 nat vs classified labels.
      If FAILED, cfgE attnd class-label input is certified inert at window grain (<= 0.05).

Writes cfge_conditioned_fisher2_results.json. Self-reviewed. RUNG-36 HEADER FOLLOWS.

CONDITION cfgE ON ITSELF (rung 36). S2128 showed conditioning the Fisher on the deployed S312 assembly adds
+0.047 over the real-model Fisher there. cfgE's certified +0.086 (S2124) still uses the REAL-model metric; this
tests whether conditioning is general or frontier-specific: recompute the true-Fisher top-8 at blocks 5/6 with
cfgE's own stand-ins installed (front tables, attnd pieces, CP middles, tail programs; labels sampled from cfgE's
own predictions), select mlp4/mlp5 units under it, and score all three arms (plain / real-model top8 / conditioned
top8) on the eight document-disjoint pile-10k windows. Note: cfgE's READERS after each site include the real
attention and lm_head, so the S2128 mechanism reading predicts conditioning matters LITTLE here.

REGISTERED PREDICTIONS (arm-named, per the S2128 process rule):
  (a) CONDITIONING INSTALLS IN cfgE TOO: median over the 8 windows of [gap(top8) - gap(cond8)] >= +0.02.
      If FAILED with (c) HELD, the conditioning gain is a property of assemblies whose readers are replaced by
      dictionaries, not of selection metrics generally - the S2128 mechanism reading survives sharpened.
  (b) NOT ONE WINDOW: the conditioning gain is >= 0 on >= 6 of 8 windows.
  (c) REPRODUCTION: the real-model top8 arm reproduces S2124's median +0.0857 within 0.01.

Descriptive: subspace overlap of the conditioned eight vs the real-model eight at both sites; an instrument check
on the window evals' attnd oracle labels (evalT inherits cur[lab] from the last fit batch - measured on window 0,
inherited vs properly classified labels). Writes cfge_conditioned_fisher_results.json. Self-reviewed.

OLDER STACKED HEADERS FOLLOW.

LABEL-FREE SELECTOR (rung 29): does the TRUE-Fisher top-8 (labels sampled from the model) still buy the gain?

§2123: the certified eight are only half shared with the label-free true Fisher (top-8 overlap 0.55/0.51). The
operational question is whether the label-dependent half carries the CE gain. This ranks mlp4/mlp5 units by the
top-8 of the MC TRUE Fisher (y ~ p, 2 samples per position, same FIT rows as every Gramian in this arc) and
certifies at K = 2304 on window 1 and the eight fresh pile-10k windows.

REGISTERED PREDICTIONS:
  (a) window-1 gain >= 0.6 x 0.1285 (§2110's empirical top-8 gain).
  (b) fresh median gain >= 0.6 x 0.082 (§2116).
  (c) positive on >= 6 of 8 fresh windows.
  (d) cfgE block-6 rel-MSE 1.7415 within 0.10.
If (a)/(b) fail, the gain needs the label-dependent half of the eight and the selector is a fitted object.

ORIGINAL HEADERS FOLLOW.


§2110: a selector scoring each mlp4/mlp5 unit by its write into the top-8 loss-gradient eigen-directions reproduces
the full metric's gain on two windows (+0.1285 / +0.0648). §2116 certified the full metric on §2083's eight fresh
pile-10k windows (8/8, median +0.082). §2118 withdrew the half-price claim on the same windows. This certifies the
named eight-direction object itself at K = 2304 on the eight windows.

REGISTERED PREDICTIONS:
  (a) top-8 gain > 0 on >= 7 of 8 windows.
  (b) median top-8 gain >= 0.6 x 0.082 (§2116's full-metric median).
  (c) REPRODUCTION: window-1 top-8 gain reproduces §2110's 0.1285 within 0.02.
  (d) cfgE block-6 rel-MSE 1.7415 within 0.10.

ORIGINAL HEADERS FOLLOW.


§2109: units the observability metric swaps in load 2.5x more on the top-8 loss-gradient directions at blocks 5/6,
1.3x on the top-r50, 1.02x on the r90 span. If the gain is carried by the head, a selector that scores units by
||P_k^T Down_u|| x ||L_u|| ||R_u|| with P_k the top-k eigenvectors (k = 8 or r50) should reproduce it at a stored
price of k x 1152 per site instead of a full Gramian, and it names the object: k directions the front must serve.

ARMS (mlp4/mlp5 at K = 2304, everything else cfgE; two windows): norm (cfgE) | full metric | top-8 | top-r50.

REGISTERED PREDICTIONS:
  (a) top-8 selector gains >= 0.8 x 0.124 on window 1 (§2106's certified full-metric gain).
  (b) top-8 selector gains >= 0.5 x 0.075 on window 2.
  (c) top-r50 >= top-8 on window 1 (more head directions, no worse). If FALSE the top 8 are the object and the
      next 80 directions add noise to the ranking.
  (d) REPRODUCTION GATE: cfgE block-6 rel-MSE 1.7415 within 0.10; full metric reproduces 0.124 within 0.02.

ORIGINAL HEADERS OF THE PARENT SCRIPTS FOLLOW.


§2106 certified +0.124 / +0.075 nat (two windows) from selecting mlp4/mlp5's 2,304 kept units by the block-5/6
observability metric instead of output norm, at identical stored values. A stand-in's worth is its CE at a price;
this sweeps K for both selectors so the metric's gain can be stated as "metric at K matches norm at K'".

ARMS: norm-selected and metric-selected mlp4/mlp5 units at K in {1152, 2304, 4608}; c6-c9 plain top-2304 in every
arm (§2106: metric selection there hurts); tables/residual bases plain; full matched-context rebuild per arm;
CE on window 1 (R0:R1) and window 2 (FW rows 0:120).

REGISTERED PREDICTIONS:
  (a) TWO-FOR-ONE: metric-2304 CE gap <= norm-4608 CE gap on window 1 -- half the units at no worse CE. If FALSE the
      gain is worth less than a doubling of the unit budget and is priced as such.
  (b) IT IS A SELECTION EFFECT, NOT CAPACITY: metric gain at K=4608 <= 0.5 x metric gain at K=2304 (window 1). With
      more units kept, which ones are chosen matters less. If FALSE the metric keeps paying at high K and the gain is
      about ordering the tail of the importance ranking, not the head.
  (c) CROSS-RUN ANCHOR: metric gain at 2304 reproduces §2106's 0.1240 within 0.02.
  (d) REPRODUCTION GATE: cfgE (norm-2304) block-6 rel-MSE reproduces 1.7415 within 0.10.

ORIGINAL HEADERS OF THE PARENT SCRIPTS FOLLOW.


§2105: the whole 0.125-nat gain of the metric-weighted front is mlp4/mlp5 unit selection under the block-5/6
observability metric (units-only +0.124, bases-only -0.009). Two things before it is a frontier number: a second
document-disjoint window (rung 6's lesson: every "Nx" in the gating arc collapsed on a second window) and the
extension to the plain CP middles c6-c9, which use the same top-2304 construction. Window 2 = FW rows 0:120, never
used by any fit (CA:CB = 300:512) or evaluation (R0:R1 = 120:300) in this arc.

REGISTERED PREDICTIONS:
  (a) IT TRANSFERS: on window 2, gain(metric-units) >= 0.062 = half of §2105's window-1 gain (0.124). If FALSE the
      0.124 is a window-1 fit and is not quotable, like §342's 3.8x.
  (b) c6-c9 ADD MORE: on window 1, gain(metric-units-all) >= gain(metric-units) + 0.03. Blocks 6-9 sit ON the price
      cliff (1.5-1.8 nat per half-norm), so metric selection there should pay at least as much per layer as at
      mlp4/mlp5. If FALSE, the gain is specific to the pieces attn5 amplifies and does not scale down the stack.
  (c) CROSS-RUN ANCHOR (LESSON 42): window-1 gain(metric-units) reproduces §2105's 0.1241 within 0.02.
  (d) REPRODUCTION GATE: cfgE's block-6 rel-MSE reproduces 1.7415 within 0.10.

ORIGINAL HEADERS OF THE PARENT SCRIPTS FOLLOW.


BENCHMARK_BACKLOG rung 12b. §2104 refit two kinds of constrained piece under site-local observability metrics at equal
stored values and gained 0.125 nat over cfgE (random-metric control 0.017). This splits it: bases-only (m0/m2/m3
residual bases under the metric, mlp4/mlp5 units plain) and units-only (the reverse), same rows, same grammar, same
control structure.

REGISTERED PREDICTIONS (gains are CE-gap reductions vs cfgE on the R0:R1 rows; §2104 joint gain = 0.1253):
  (a) THE UNITS CARRY IT: gain(units-only) >= 0.5 x 0.1253. mlp4 is the piece attn5 amplifies 8.6x (§2102) and the
      block-6 metric has 1300x the block-1 metric's scale, so re-selecting mlp4/mlp5 units by what the loss reads
      should be most of the effect. If FALSE the bases (the tables' residual directions) carry it.
  (b) ADDITIVE: gain(bases-only) + gain(units-only) is within 30% of the joint 0.1253. If FALSE the two knobs
      interact and neither can be priced alone.
  (c) CONTROL: the random-metric arm reproduces §2104's +0.0168 within 0.02 (LESSON 42 cross-run anchor).
  (d) REPRODUCTION GATE: cfgE's block-6 rel-MSE reproduces 1.7415 within 0.10.

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
_bq0='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
if os.environ.get('BQLIB_DRYRUN')=='1':
    if not os.path.exists(_bq0+'cfge_conditioned_fisher_results.json'):
        print('DRYRUN FAIL: S2130 artifact absent'); raise SystemExit(1)
    _p0=json.load(open(_bq0+'cfge_conditioned_fisher_results.json'))
    print(f"DRYRUN OK: S2130 cond median {_p0['gain_cond_median']}; RNG-matched rerun + random-label arm")
    raise SystemExit(0)
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['stream_error_profile_results.json','../polynomial_causal/fisher_metric_v1_results.json']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    _p=json.load(open(_bq+'stream_error_profile_results.json'))['profile']
    print(f"DRYRUN OK: S2086 present (block 6 {_p['6']}); norm vs TRUE-Fisher top-8 at K=2304 x eight fresh windows")
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
OUT=PT+'cfge_conditioned_fisher2_results.json'
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
    genF=torch.Generator(device=DEV).manual_seed(29)
    def gramian(TOKS,site):
        G=torch.zeros(D,D,device=DEV,dtype=torch.float64); n=0
        for b0 in range(0,TOKS.shape[0],4):
            idx=TOKS[b0:b0+4,:-1]
            with torch.no_grad():
                x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
                for li,blk in enumerate(m.transformer.h): x,v1=blk(x,v1,x0)
                p=torch.softmax((30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30))[:,:-1].float(),-1)
            for _s in range(2):
                y=torch.multinomial(p.reshape(-1,p.shape[-1]),1,generator=genF).view(p.shape[0],p.shape[1])
                with torch.enable_grad():
                    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None; leaf=None
                    for li,blk in enumerate(m.transformer.h):
                        if li==site:
                            x=x.detach().requires_grad_(True); leaf=x
                        x,v1=blk(x,v1,x0)
                    lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
                    lp=F.log_softmax(lg[:,:-1],-1)
                    nll=-lp.gather(-1,y[...,None]).squeeze(-1)
                    nll[:,SKIP:].sum().backward()
                g=leaf.grad[:,SKIP:-1].reshape(-1,D).double(); G+=g.T@g; n+=g.shape[0]
                m.zero_grad(set_to_none=True)
        return G/n
    SITE_OF={'m0':1,'m2':3,'m3':4,'c4':5,'c5':6,'c6':7,'c7':8,'c8':9,'c9':10}
    gen=torch.Generator(device='cpu').manual_seed(12)
    MET={}; HEAD={}
    for piece,site in SITE_OF.items():
        G=gramian(TOKF,site); e,Q=torch.linalg.eigh(G)
        ed,Qd=e.flip(0).clamp_min(0),Q.flip(1); cc=torch.cumsum(ed,0)/ed.sum(); r50=int((cc<0.5).sum())+1
        HEAD[piece]={'top8':Qd[:,:8].float().contiguous(),'topr50':Qd[:,:r50].float().contiguous(),'r50':r50}
        e=e.clamp_min(1e-3*float(e.max()))
        Qr=torch.linalg.qr(torch.randn(D,D,generator=gen,dtype=torch.float64))[0].to(DEV)
        for mode,QQ in (('metric',Q),('random-metric',Qr)):
            half=(QQ*e.sqrt()[None,:])@QQ.T; inv=(QQ*(1/e.sqrt())[None,:])@QQ.T
            MET[(mode,piece)]=(half.float(),inv.float())
        print(f'metric for {piece} at block {site}: top eig {float(e.max()):.3e}, floor {float(e.min()):.3e}',flush=True)
    Ge=gramian(TOKE,6); ee,Qe=torch.linalg.eigh(Ge); ee,Qe=ee.flip(0).clamp_min(0),Qe.flip(1)
    ce_=torch.cumsum(ee,0)/ee.sum(); r90e=int((ce_<0.9).sum())+1; P6=Qe[:,:r90e].float()
    print(f'eval-row block-6 observable r90 {r90e}',flush=True)
    def cgramian(active,site):
        genC=torch.Generator(device=DEV).manual_seed(29)
        G=torch.zeros(D,D,device=DEV,dtype=torch.float64); n=0
        for b0 in range(0,TOKF.shape[0],4):
            idx=TOKF[b0:b0+4,:-1]
            cur['idx']=idx.contiguous(); cur['mode']='oracle'
            cur['lab']=clsA[b0:b0+4].reshape(-1)
            hs=install(active)
            try:
                with torch.no_grad():
                    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
                    for _li,blk in enumerate(m.transformer.h): x,v1=blk(x,v1,x0)
                    p=torch.softmax((30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30))[:,:-1].float(),-1)
                for _s in range(2):
                    y=torch.multinomial(p.reshape(-1,p.shape[-1]),1,generator=genC).view(p.shape[0],p.shape[1])
                    with torch.enable_grad():
                        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None; leaf=None
                        for _li,blk in enumerate(m.transformer.h):
                            if _li==site:
                                x=x.detach().requires_grad_(True); leaf=x
                            x,v1=blk(x,v1,x0)
                        lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
                        lp=F.log_softmax(lg[:,:-1],-1)
                        nll=-lp.gather(-1,y[...,None]).squeeze(-1)
                        nll[:,SKIP:].sum().backward()
                    g=leaf.grad[:,SKIP:-1].reshape(-1,D).double(); G+=g.T@g; n+=g.shape[0]
                    m.zero_grad(set_to_none=True)
            finally:
                for h in hs: h.remove()
        return G/n
    # ---- arms ----
    def build_arm(mode,K=2304):
        def fit_res(li,piece,active):
            Y,X,ids=runA(active,m.transformer.h[li].mlp)
            tb=fit_table(Y,ids)
            Rr=Y-tb[ids].float()
            ft=quadfeat(X,li); lam=1e-2*len(X)
            if mode in ('plain','metric-units','metric-units-all','top8','topr50','cond8'):
                _,_,Vh2=torch.linalg.svd(Rr[:30000],full_matrices=False)
                P=orth(Vh2[:64].T)
                A=torch.linalg.solve(ft.T@ft+lam*torch.eye(ft.shape[1],device=DEV),ft.T@(Rr@P))
                return ('tableres',li,tb,A,P)
            half,inv=MET[('random-metric' if mode=='random-metric' else 'metric',piece)]
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
            if mode in ('plain','metric-bases') or (mode!='plain' and li>=6):
                imp=Dw.norm(dim=0)*L.norm(dim=1)*Rw.norm(dim=1)
            elif mode=='cond8':
                Pk=CONDH[piece]
                imp=(Pk.T@Dw).norm(dim=0)*L.norm(dim=1)*Rw.norm(dim=1)
            elif mode in ('top8','topr50'):
                Pk=HEAD[piece][mode]; imp=(Pk.T@Dw).norm(dim=0)*L.norm(dim=1)*Rw.norm(dim=1)
            else:
                half,_=MET[('random-metric' if mode=='random-metric' else 'metric',piece)]
                imp=(half@Dw).norm(dim=0)*L.norm(dim=1)*Rw.norm(dim=1)
            keep=imp.argsort(descending=True)[:(K if li in (4,5) else 2304)]
            return ('cp',li,L[keep].contiguous(),Rw[keep].contiguous(),Dw[:,keep].contiguous(),db)
        tag={'plain':'E','metric':'M','metric-bases':'B','metric-units':'U','metric-units-all':'A','random-metric':'R','top8':'T','topr50':'F','cond8':'Q'}[mode]
        act=['a0']
        S[f'm0{tag}']=fit_res(0,'m0',act); act.append(f'm0{tag}')
        act.append('a1v'); act.append('m1')
        S[f'm2{tag}']=fit_res(2,'m2',act); act.append(f'm2{tag}')
        S[f'm3{tag}']=fit_res(3,'m3',act); act.append(f'm3{tag}')
        for li in (4,5,6,7,8,9): S[f'c{li}{tag}']=cp_piece(li,f'c{li}')
        stack=act+[f'c{li}{tag}' for li in (4,5,6,7,8,9)]
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
    arms={}; OVL={}; CONDH={}
    for mode in ('plain','top8','cond8'):
        if mode=='cond8':
            for piece,site in (('c4',5),('c5',6)):
                Gc=cgramian(arms['plain']['cfg'],site)
                ec,Qc=torch.linalg.eigh(Gc); Qc=Qc.flip(1)
                CONDH[piece]=Qc[:,:8].float().contiguous()
                ov=float(((HEAD[piece]['top8'].double().T@Qc[:,:8])**2).sum()/8)
                OVL[piece]=round(ov,4)
                print(f'conditioned top-8 for {piece} (site {site}): overlap with real-model eight {ov:.3f}',flush=True)
        cfg=build_arm(mode,2304)
        st=collect(cfg)
        prof=[float(((st[li]-real[li])**2).mean(0).sum())/max(float(real[li].var(0).sum()),1e-12) for li in PROBES]
        del st; torch.cuda.empty_cache()
        arms[mode]={'cfg':cfg,'profile':[round(p,4) for p in prof]}
    base1=evalT(W1,R1-R0,[])
    gap1={mode:evalT(W1,R1-R0,arms[mode]['cfg'])-base1 for mode in arms}
    rows=[]
    for wi,w in enumerate(WIN):
        T=w['tok']; b=evalT(T,120,[])
        g={mode:evalT(T,120,arms[mode]['cfg'])-b for mode in arms}
        rows.append({'window':wi,'docs':w['docs'],'base_ce':round(b,5),'gap_plain':round(g['plain'],5),
                     'gap_top8':round(g['top8'],5),'gap_cond8':round(g['cond8'],5),
                     'gain_top8':round(g['plain']-g['top8'],5),'gain_cond':round(g['top8']-g['cond8'],5)})
        print(f"window {wi}: docs {w['docs']:3d} base {b:.4f} | gap plain {g['plain']:+.4f} top8 {g['top8']:+.4f} "
              f"cond8 {g['cond8']:+.4f} | gain(top8) {g['plain']-g['top8']:+.4f} gain(cond) {g['top8']-g['cond8']:+.4f}",flush=True)
    import statistics as stt
    g45=[r['gain_top8'] for r in rows]
    gcd=[r['gain_cond'] for r in rows]
    pa=stt.median(gcd)>=0.02
    pb=sum(g>=0 for g in gcd)>=6
    pc=abs(stt.median(g45)-0.0857)<=0.01
    def cls_tokens(Tk):
        n=Tk.shape[0]; Mid=torch.zeros(n,256,dtype=torch.long)
        for r in range(n):
            toks=Tk[r,:257].tolist()
            for pos in range(256):
                t=toks[pos+1]; pv=enc3.decode([toks[pos]]); tg=enc3.decode([t]); st=tg.strip()
                if st.isdigit() and not tg.startswith(' '): k=0
                elif st in (')',']') and any(bch in enc3.decode(toks[max(0,pos-60):pos+1]) for bch in ('(','[')): k=1
                elif chr(10) in tg: k=2
                elif tg in ('.','!','?'): k=3
                elif tg==',': k=4
                elif (tg.startswith(' ') and st[:1].isupper() and (pv.strip()[:1].isupper() if pv.strip() else False)): k=5
                elif t==toks[pos]: k=6
                elif (not tg.startswith(' ')) and st.isalpha(): k=7
                elif t in toks[:pos+1]: k=8
                else: k=9
                Mid[r,pos]=k
        return Mid
    labs=cls_tokens(WIN[0]['tok'])
    def evalL(TOK,N,active,LB):
        hs=install(active); ces=[]
        for i2 in range(0,N,4):
            bb=TOK[i2:i2+4,:257].to(DEV)
            cur['idx']=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            cur['mode']='oracle'; cur['lab']=LB[i2:i2+4].reshape(-1).to(DEV)
            x=F.rms_norm(m.transformer.wte(cur['idx']),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,reduction='none'))
        for h in hs: h.remove()
        return float(torch.cat(ces).mean())
    prop=evalL(WIN[0]['tok'],120,arms['plain']['cfg'],labs)-rows[0]['base_ce']
    inh=rows[0]['gap_plain']
    genL=torch.Generator().manual_seed(38)
    rnd_labs=torch.randint(0,10,labs.shape,generator=genL)
    rndg=evalL(WIN[0]['tok'],120,arms['plain']['cfg'],rnd_labs)-rows[0]['base_ce']
    print(f'instrument: window-0 plain gap, inherited {inh:+.4f} | classified {prop:+.4f} | RANDOM labels {rndg:+.4f}',flush=True)
    pd=abs(rndg-prop)>=0.05
    out={'windows':rows,'gain_cond_median':round(stt.median(gcd),4),'gain_top8_median':round(stt.median(g45),4),
         'cond_windows_nonneg':int(sum(g>=0 for g in gcd)),'overlap_cond_vs_real_top8':OVL,
         'instrument_label_check_w0':{'inherited_gap':round(inh,4),'classified_gap':round(prop,4),
                                      'random_gap':round(rndg,4)},
         'profiles':{k:v['profile'] for k,v in arms.items()},
         'pred_a_conditioning_installs':bool(pa),'pred_b_windows_nonneg':bool(pb),'pred_c_reproduces_top8':bool(pc),
         'pred_d_labels_matter':bool(pd),
         'self_reviewed':True,'runtime_s':round(time.time()-t0,1)}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'per-window gain(top8 over plain): {g45}')
    print(f'per-window gain(cond8 over top8): {gcd}')
    print(f"(a) conditioning median {stt.median(gcd):+.4f} >= 0.02: {'HELD' if pa else 'FAILED'}")
    print(f"(b) non-negative on {sum(g>=0 for g in gcd)}/8 (bar 6): {'HELD' if pb else 'FAILED'}")
    print(f"(c) top8 median {stt.median(g45):+.4f} vs 0.0857 (tol 0.01): {'HELD' if pc else 'FAILED'}")
    print(f"(d) random-label shift {abs(rndg-prop):.4f} >= 0.05: {'HELD' if pd else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-t0:.0f}s)')

if __name__=='__main__': main()
