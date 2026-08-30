"""PRUNE c6-c9 UNDER THE CONDITIONED RANKING (rung 42).

S2131 showed REORDERING c6-c9 at K = 2304 adds nothing. S2134 reframed the mlp4/5 win: it came from DROPPING the
harmful tail of the conditioned ranking, not from reordering the top. So the open question at c6-c9 is pruning:
with mlp4/5 fixed at cond-576 (the S2134 best), cut c6-c9 from 2304 to 576 each - ranked by the conditioned
Fisher at their sites vs by norm. Saves 4 x 1728 units (~23.9M values) if free. The S2131 null (conditioned
ranking is worthless at c6-c9) predicts pred_b ~ 0.

ARMS: norm-2304 (Fisher collection, sites 5-10) | cond576@45 (S2134 reproduction; c6-c9 norm-2304) |
c69norm576 (c6-c9 pruned by NORM rank) | c69cond576 (c6-c9 pruned by conditioned rank).

REGISTERED PREDICTIONS (arm-named):
  (a) PRUNING c6-c9 IS AT LEAST FREE: median over the 8 windows of [L2_F_w(c69cond576) - L2_F_w(cond576@45)] >= 0.
  (b) THE CONDITIONED RANK BEATS NORM FOR WHAT SURVIVES: median [L2_F_w(c69cond576) - L2_F_w(c69norm576)] >= 0.
  (c) REPRODUCTION: the cond576@45 arm reproduces S2134 (median over norm +0.1277 within 0.015).

Writes frontier_c69_prune_results.json. Self-reviewed. RUNG-41 HEADER FOLLOWS.

THE PRICE FLOOR (rung 41): K = 144, and K = 0 as the "do the units matter at all" null.

S2134: the conditioned curve peaks near K = 576 (+2.8372 fresh, quarter price) and cond-288 barely turns down.
Arms: norm-2304 (Fisher collection) | cond-576 (S2134 reproduction) | cond-144 | K-0 (Down-bias only).

REGISTERED PREDICTIONS (arm-named):
  (a) SIXTEENTH PRICE AT NO LOSS: median over the 8 windows of [L2_F_w(cond144) - L2_F_w(cond576)] >= -0.01.
  (b) THE UNITS CARRY REAL CE: median [L2_F_w(K0) - L2_F_w(cond576)] <= -0.05. If FAILED, mlp4/mlp5 CP content
      is nearly all harmful-or-inert in the deployed assembly - a major finding on its own.
  (c) REPRODUCTION: the cond-576 arm reproduces S2134 (median over norm +0.1277 within 0.015).

Writes frontier_cond_floor_results.json. Self-reviewed. RUNG-40 HEADER FOLLOWS.

THE CONDITIONED PRICE CURVE, DOWNWARD (rung 40): does the half-price win extend to quarter and eighth price?

S2133: cond-1152 beats norm-2304 by +0.0861 median and cond-2304 by ~+0.04 on every window - the bottom half of
the conditioned ranking is net harmful when deployed. S2107 (capacity vs selection) says the curve must turn
somewhere below; this finds where. Arms: norm-2304 (Fisher collection) | cond-1152 (S2133 reproduction) |
cond-576 | cond-288, each on the eight document-disjoint windows.

REGISTERED PREDICTIONS (arm-named):
  (a) QUARTER PRICE AT NO LOSS: median over the 8 windows of [L2_F_w(cond576) - L2_F_w(cond1152)] >= -0.01.
  (b) NOT ONE WINDOW: that difference is >= -0.02 on >= 6 of 8 windows.
  (c) REPRODUCTION: the cond-1152 arm reproduces S2133 (median over norm +0.0861 within 0.015).
  cond-288 is descriptive (the curve's next point), no bar.

Writes frontier_cond_ksweep_results.json. Self-reviewed. RUNG-39 HEADER FOLLOWS.

HALF-PRICE UNDER THE CONDITIONED METRIC (rung 39): does the assembly-conditioned selector let mlp4/mlp5 keep
HALF the units at no CE loss in the deployed frontier?

S2107 found metric-1152 ~ norm-2304 on two windows; S2118 WITHDREW half-price on the eight fresh windows under
the real-model metric on cfgE (median -0.028, worse on six of eight) - that is the stated null. But S2129 showed
the conditioned metric is the right one for the frontier (+0.0481 certified), so the price question reopens
exactly once, here: arms norm-2304 | conditioned-2304 (S2129 reproduction) | conditioned-1152 at mlp4/mlp5, each
on the eight document-disjoint windows. LITERAL PRICE: conditioned-1152 stores 2 x 1152 fewer CP units
(~7.96M fewer values) than either 2304 arm, plus the same 8 x 1152 selector per site.

REGISTERED PREDICTIONS (arm-named):
  (a) HALF PRICE AT NO LOSS: median over the 8 windows of [L2_F_w(cond1152) - L2_F_w(norm2304)] >= 0.
      If FAILED the S2118 withdrawal extends to the conditioned metric and the price axis is closed at mlp4/5.
  (b) NOT ONE WINDOW: that difference is >= -0.01 on >= 6 of 8 windows.
  (c) REPRODUCTION: the conditioned-2304 arm reproduces S2129 (median gain over norm +0.0481 within 0.015).

Writes frontier_cond_halfprice_results.json. Self-reviewed. RUNG-37 HEADER FOLLOWS.

EXTEND THE CONDITIONED SELECTION TO ALL SIX CP MIDDLES (rung 37).

S2129 certified the assembly-conditioned selector at mlp4/mlp5 (+0.0481 median over eight windows). The other four
CP middles (c6-c9) still use norm selection - and S2106/S2107 found that REAL-model metric selection there HURTS
on cfgE, so the null expectation is genuine. This collects the assembly-conditioned Fisher top-8 at sites 5-10
during the norm arm and runs THREE arms: norm | conditioned selection at mlp4/mlp5 (S2129 reproduction) |
conditioned selection at mlp4-mlp9, each scored on the eight document-disjoint windows.

REGISTERED PREDICTIONS (arm-named, per the S2128 process rule):
  (a) THE OTHER FOUR LAYERS ADD: median over the 8 windows of [L2_F_w(all) - L2_F_w(mlp45)] >= +0.02.
      If FAILED, conditioned selection is an mlp4/mlp5 story like the real-model metric was (S2106) and the
      frontier keeps the S2129 config.
  (b) NOT ONE WINDOW: that increment is >= 0 on >= 6 of 8 windows.
  (c) REPRODUCTION: the mlp45 arm reproduces S2129 (median gain over norm +0.0481 within 0.015; the Fisher
      collection has ~0.003 CUDA-atomics wobble, S2129).

Writes frontier_asm_alllayers_results.json. Self-reviewed. RUNG-35 HEADER FOLLOWS.

CERTIFY THE CONDITIONED FRONTIER AT THE RUNG-6 STANDARD (rung 35).

S2128 (correcting the sign-inverted S2125): assembly-conditioned true-Fisher top-8 selection at mlp4/mlp5 lifts
the S312 frontier from +2.6735 to +2.7682 fresh — but that fresh number is ONE 120-row window set (FR). Before
+2.7682 is called the frontier best it must meet the certification standard every other frontier claim met: the
eight document-disjoint pile-10k windows (built exactly as ops/metric_units_fresh8.py builds them). This reruns
both arms and evaluates each L2 config on all eight windows; fits touch FW rows only, so every window is held out
for both arms.

REGISTERED PREDICTIONS (formulas name the arms; rung 30's sign error is the reason):
  (a) THE GAIN CERTIFIES: median over the 8 windows of [L2_F_w(asm) - L2_F_w(norm)] >= 0.04 (S2128 saw +0.095 on
      FR; half survives averaging is the bar, matching rung 22's standard).
  (b) NOT ONE WINDOW'S STORY: the gain is positive on >= 7 of 8 windows.
  (c) REPRODUCTION: the norm arm's FR L2_F reproduces 2.6735 within 0.01 (pipeline unchanged).

Writes frontier_asm_fresh8_results.json. Self-reviewed. RUNG-30 HEADER FOLLOWS.

INSTALL THE CERTIFIED SELECTOR INTO THE FRONTIER (rung 30). The observability arc's certified, label-free
gain (§2116/§2119/§2124: mlp4/mlp5 CP units ranked by the true-Fisher top-8 at blocks 5/6, +0.082-0.086 median on
eight fresh windows) was shown on cfgE, the all-attention-real arm. The quotable frontier is §312's empirical-L2
config (+2.6735 fresh: empirical base + 38 motif heads + tail-attention dictionaries), whose middle MLPs use the
SAME norm-selected top-2304 CP construction. This reruns §312's pipeline twice — norm selection vs true-Fisher
top-8 selection at mlp4/mlp5 (c6-c9 stay norm, §2106) — and re-measures the frontier.

REGISTERED PREDICTIONS:
  (a) REPRODUCTION GATE: the norm arm's L2_F reproduces the published 2.6735 within 0.05.
  (b) THE GAIN INSTALLS: L2_F(norm) - L2_F(fisher8) >= 0.04 — half of cfgE's fresh median 0.086, since the
      frontier config carries motif-head and tail-dictionary error the selector does not touch. If FAILED, the
      certified gain does not survive composition with the head hybrids and tail refits, and the frontier keeps
      norm selection.
  (c) NO IN-DISTRIBUTION HARM: L2_C(norm) - L2_C(fisher8) >= -0.01.

Writes frontier_fisher8_results.json. Self-reviewed. ORIGINAL §312 HEADER FOLLOWS.

HIGH-COVERAGE VIA LOCAL PRICING -- the user's question sharpened the
objective: coverage is the goal; the pricing principle is the tool for
the HIGH-coverage end. Prediction from 311: at full-band coverage the
EMPIRICAL base wins (its block 2-9 streams are faithful where the
motif heads read), even though the fold base wins at low coverage.
Config: empirical matched-context base (20 comps) + 38 motif heads
(+2.287 fresh, measured) + tail-attention dictionaries a10-17 refit
under that exact stack -> 28 components + 38 heads.
REGISTERED PREDICTIONS:
  (a) total <= +2.75 fresh (beats fold-L2's +2.84 AND the old 34-comp
      +2.93 at comparable coverage);
  (b) the tail-attn increment lands in [0.30, 0.55] (pricing: empirical
      late streams are mid-faithful; fold's increment was 0.34 on
      better late streams);
  (c) window-C total reported."""
import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['frontier_cond_floor_results.json','empirical_L2_results.json']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    _p=json.load(open(_bq+'empirical_L2_results.json'))
    print(f"DRYRUN OK: published L2_F {_p['L2_F']}; c6-c9 pruning arms x eight windows")
    raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from circuit_dictionary import classify, COMPS as TAILC, CLS
D=1152; V=50257
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'frontier_c69_prune_results.json'
CA,CB=300,512; R0,R1=120,300
CONSTN={'digit','bclose','sentend','comma','name','rep'}
CONSTK=[k for k,nm in enumerate(CLS) if nm in CONSTN]
LINK=[k for k in range(10) if k not in CONSTK]
ATTM=[2,3,4,5,6,7,8,9]; ATTT=[10,11,12,13,14,15,16,17]
MIDL=(4,5,6,7,8,9)
TRI=torch.triu_indices(32,32)
SEL={'mode':'norm','P8':{}}
SKIP8=64


def select_units(li,L,Rw,Dw):
    if SEL['mode']=='fisher8' and li in (4,5):
        Pk=SEL['P8'][li+1]
        imp=(Pk.T@Dw).norm(dim=0)*L.norm(dim=1)*Rw.norm(dim=1)
    elif SEL['mode']=='fisher8all' and li in (4,5,6,7,8,9):
        Pk=SEL['P8'][li+1]
        imp=(Pk.T@Dw).norm(dim=0)*L.norm(dim=1)*Rw.norm(dim=1)
    else:
        imp=Dw.norm(dim=0)*L.norm(dim=1)*Rw.norm(dim=1)
    return imp.argsort(descending=True)[:(SEL.get('K',2304) if li in (4,5) else SEL.get('K69',2304))]


def fisher_top8(site):
    """true-Fisher (y~p, 2 samples) top-8 at a block input, on the FIT rows; §2124's construction, seed 29."""
    genF=torch.Generator(device=DEV).manual_seed(29)
    TOKS=torch.cat([FW[i:i+4,:257] for i in range(CA,CB,4)]).to(DEV)
    G=torch.zeros(D,D,device=DEV,dtype=torch.float64)
    for b0 in range(0,TOKS.shape[0],4):
        idx=TOKS[b0:b0+4,:-1]
        with torch.no_grad():
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for _li,blk in enumerate(m.transformer.h): x,v1=blk(x,v1,x0)
            p=torch.softmax((30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30))[:,:-1].float(),-1)
        for _s in range(2):
            y=torch.multinomial(p.reshape(-1,p.shape[-1]),1,generator=genF).view(p.shape[0],p.shape[1])
            with torch.enable_grad():
                x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None; leaf=None
                for _li,blk in enumerate(m.transformer.h):
                    if _li==site:
                        x=x.detach().requires_grad_(True); leaf=x
                    x,v1=blk(x,v1,x0)
                lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
                lp=F.log_softmax(lg[:,:-1],-1)
                (-lp.gather(-1,y[...,None]).squeeze(-1))[:,SKIP8:].sum().backward()
            g=leaf.grad[:,SKIP8:-1].reshape(-1,D).double(); G+=g.T@g
            m.zero_grad(set_to_none=True)
    _e,Q=torch.linalg.eigh(G)
    return Q.flip(1)[:,:8].float().contiguous()

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
        keep=select_units(li,L,Rw,Dw)
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
        keep=select_units(li,L,Rw,Dw)
        S[f'c{li}']=('cp',li,L[keep].contiguous(),Rw[keep].contiguous(),
                     Dw[:,keep].contiguous(),db)
    # ---- matched-context sequential fits, both arms ----
    def build_arm(fold):
        S3={}
        def fit_res(li,tb,active):
            Y,X,ids=runA(active,m.transformer.h[li].mlp)
            if tb is None:
                tb=fit_table(Y,ids)
            Rr=Y-tb[ids].float()
            _,_,Vh2=torch.linalg.svd(Rr[:30000],full_matrices=False)
            P=orth(Vh2[:64].T)
            ft=quadfeat(X,li)
            lam=1e-2*len(X)
            A=torch.linalg.solve(ft.T@ft+lam*torch.eye(ft.shape[1],
                                                       device=DEV),
                                 ft.T@(Rr@P))
            return ('tableres',li,tb,A,P)
        pre='F' if fold else 'E'
        S[f'm0{pre}']=fit_res(0,FOLD[0] if fold else None,['a0'])
        act=['a0',f'm0{pre}','a1v','m1']
        S[f'm2{pre}']=fit_res(2,FOLD[2] if fold else None,act)
        act=act+[f'm2{pre}']
        S[f'm3{pre}']=fit_res(3,FOLD[3] if fold else None,act)
        front=['a0',f'm0{pre}','a1v','m1',f'm2{pre}',f'm3{pre}']
        stack=front+[f'c{li}' for li in (4,5,6,7,8,9)]
        # tail refit under this exact stack (attention real)
        capsT={li:[] for li in TAILC}; capsI={li:[] for li in TAILC}
        hs=install(stack)
        for li in TAILC:
            def mk2(li=li):
                def h(mo,i_,o_):
                    capsT[li].append(o_.detach().reshape(-1,D).float())
                    capsI[li].append(i_[0].detach().reshape(-1,D)
                                     .float())
                return h
            hs.append(m.transformer.h[li].mlp
                      .register_forward_hook(mk2()))
        for i in range(CA,CB,4):
            bb=FW[i:i+4,:257].to(DEV)
            cur['idx']=bb[:,:-1].contiguous()
            cur['mode']='oracle'
            cur['lab']=clsA.reshape(CB-CA,256)[i-CA:i-CA+4].reshape(-1)
            m(cur['idx'], bb[:,1:].contiguous())
        for h in hs: h.remove()
        X10=torch.cat(capsI[TAILC[0]])
        Yoh=torch.zeros(len(flatA),10,device=DEV)
        Yoh[torch.arange(len(flatA)),flatA]=1.0
        lam=1e-2*len(X10)
        Wp=torch.linalg.solve(X10.T@X10+lam*torch.eye(D,device=DEV),
                              X10.T@Yoh)
        DICT={}; LIN={}
        for li in TAILC:
            Q,_=spans[li]; C=torch.cat(capsT[li])@Q
            DICT[li]=torch.stack([C[flatA==k].mean(0)
                                  if (flatA==k).sum()>0 else C.mean(0)
                                  for k in range(10)])
            Xl=torch.cat(capsI[li]); LIN[li]={}
            for k in (8,9):
                mk_=flatA==k
                Xk=Xl[mk_]; Ck=C[mk_]
                l2=1e-2*len(Xk)
                LIN[li][k]=torch.linalg.solve(
                    Xk.T@Xk+l2*torch.eye(D,device=DEV),Xk.T@Ck)
            capsT[li]=None; capsI[li]=None
        S[f'tail{pre}']=('tail',Wp,DICT,LIN)
        return stack+[f'tail{pre}']
    cfgF=build_arm(False)  # EMPIRICAL twin base
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
    import tiktoken
    from datasets import load_dataset
    enc3=tiktoken.get_encoding('gpt2')
    dsf=load_dataset('NeelNanda/pile-10k',split='train')
    seen={tuple(FW[r,:32].tolist()) for r in range(FW.shape[0])}
    rows=[]
    for di in range(3000,10000):
        tk=enc3.encode_ordinary(dsf[di]['text'])
        for st0 in range(0,len(tk)-513,513):
            row=tk[st0:st0+513]
            if tuple(row[:32]) in seen: continue
            rows.append(row)
            if len(rows)>=120: break
        if len(rows)>=120: break
    FR=torch.tensor(rows,dtype=torch.long)
    baseC=evalT(FW[R0:R1],R1-R0,[])
    baseF=evalT(FR,120,[])
    # ---- L1: motif gains fit under cfgF ----
    mt=json.load(open(PT+'attn_motifs3_results.json'))['motif_table']
    prevh={}; selfh={}
    for li,hd,mo,fr in mt:
        if 2<=li<=9:
            if mo=='prev': prevh.setdefault(li,[]).append(hd)
            if mo=='self': selfh.setdefault(li,[]).append(hd)
    mod2=sys.modules[type(m.transformer.h[0].attn).__module__]
    are=mod2.apply_rotary_emb
    T=256
    def head_z(at,X2,v1):
        B=X2.shape[0]
        q=at.c_q(X2).view(B,T,9,128); k=at.c_k(X2).view(B,T,9,128)
        q2=at.c_q2(X2).view(B,T,9,128); k2=at.c_k2(X2).view(B,T,9,128)
        v=at.c_v(X2).view(B,T,9,128)
        if v1 is None: v1=v
        vm=(1-at.lamb)*v+at.lamb*v1.view_as(v)
        cos,sin=at.rotary(q)
        qn=F.rms_norm(q,(128,)); kn=F.rms_norm(k,(128,))
        qn,kn=are(qn,cos,sin),are(kn,cos,sin)
        q2n=F.rms_norm(q2,(128,)); k2n=F.rms_norm(k2,(128,))
        q2n,k2n=are(q2n,cos,sin),are(k2n,cos,sin)
        sc=torch.einsum('bqhd,bkhd->bhqk',qn.float(),kn.float())/128
        sc2=torch.einsum('bqhd,bkhd->bhqk',q2n.float(),k2n.float())/128
        pat=(sc*sc2)*torch.tril(torch.ones(T,T,device=DEV))
        z=torch.einsum('bhqk,bkhd->bhqd',pat,vm.float())
        return z,vm.float()
    caps={li:{'x':[],'v1':[]} for li in range(2,10)}
    hs=install(cfgF)
    for li in range(2,10):
        def mkc(li=li):
            def h(mo_,args):
                caps[li]['x'].append(args[0].detach())
                caps[li]['v1'].append(args[1].detach()
                                      if args[1] is not None else None)
            return h
        hs.append(m.transformer.h[li].attn
                  .register_forward_pre_hook(mkc()))
    for i in range(CA,CA+32,4):
        bb=FW[i:i+4,:257].to(DEV)
        cur['idx']=bb[:,:-1].contiguous()
        m(cur['idx'], bb[:,1:].contiguous())
    for h in hs: h.remove()
    ALPHA={}
    for li in range(2,10):
        at=m.transformer.h[li].attn
        num=torch.zeros(9,device=DEV); den=torch.zeros(9,device=DEV)
        nums=torch.zeros(9,device=DEV); dens=torch.zeros(9,device=DEV)
        for X2,v1 in zip(caps[li]['x'],caps[li]['v1']):
            z,vm=head_z(at,X2,v1)
            vp=torch.zeros_like(vm); vp[:,1:]=vm[:,:-1]
            vp=vp.permute(0,2,1,3); vs=vm.permute(0,2,1,3)
            num+=(z*vp).sum((0,2,3)); den+=(vp*vp).sum((0,2,3))
            nums+=(z*vs).sum((0,2,3)); dens+=(vs*vs).sum((0,2,3))
        ALPHA[li]=(num/den.clamp_min(1e-9),nums/dens.clamp_min(1e-9))
        caps[li]=None
    def motif_hooks(layers):
        hs2=[]
        for li in layers:
            if li not in set(list(prevh)+list(selfh)): continue
            at=m.transformer.h[li].attn
            ap,asf=ALPHA[li]
            ph=prevh.get(li,[]); sh=selfh.get(li,[])
            def h(mo_,args,out,at=at,ph=ph,sh=sh,ap=ap,asf=asf):
                y,v1r=out
                X2=args[0]; v1=args[1] if args[1] is not None else v1r
                z,vm=head_z(at,X2,v1)
                vp=torch.zeros_like(vm); vp[:,1:]=vm[:,:-1]
                vp=vp.permute(0,2,1,3); vs=vm.permute(0,2,1,3)
                for hd in ph: z[:,hd]=ap[hd]*vp[:,hd]
                for hd in sh: z[:,hd]=asf[hd]*vs[:,hd]
                B=X2.shape[0]
                ynew=at.c_proj(z.transpose(1,2).contiguous()
                               .view(B,T,-1).to(X2.dtype))
                return (ynew,v1r)
            hs2.append(at.register_forward_hook(h))
        return hs2
    def evalM(TOK,N,active,mlayers):
        hs=install(active)+motif_hooks(mlayers)
        ces=[]
        for i in range(0,N,4):
            bb=TOK[i:i+4,:257].to(DEV)
            cur['idx']=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            cur['mode']='oracle'
            if 'clsmap' in cur:
                cur['lab']=cur['clsmap'][i:i+4].reshape(-1)
            x=F.rms_norm(m.transformer.wte(cur['idx']),(D,))
            x0=x; v1=None
            for blk in m.transformer.h:
                x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        for h in hs: h.remove()
        return float(torch.cat(ces).mean())
    ML=list(range(2,10))
    L1C=evalM(FW[R0:R1],R1-R0,cfgF,ML)-baseC
    L1F=evalM(FR,120,cfgF,ML)-baseF
    print(f'L1 (+38 heads): C {L1C:+.4f} | fresh {L1F:+.4f}',flush=True)
    # tail-attention dicts refit under (empirical base + motifs)
    Yoh=torch.zeros(len(flatA),10,device=DEV)
    Yoh[torch.arange(len(flatA)),flatA]=1.0
    order2=list(cfgF)
    Wp2=None
    ML=list(range(2,10))
    for li in range(10,18):
        Ys=[]; Xs=[]
        hs=install(order2)+motif_hooks(ML)
        def cap2(mo_,i_,o_):
            Ys.append((o_[0]).detach().reshape(-1,D).float())
            Xs.append(i_[0].detach().reshape(-1,D).float())
        hs.append(m.transformer.h[li].attn.register_forward_hook(cap2))
        for i in range(CA,CB,4):
            bb=FW[i:i+4,:257].to(DEV)
            cur['idx']=bb[:,:-1].contiguous()
            cur['mode']='oracle'
            cur['lab']=clsA.reshape(CB-CA,256)[i-CA:i-CA+4].reshape(-1)
            m(cur['idx'], bb[:,1:].contiguous())
        for h in hs: h.remove()
        Y=torch.cat(Ys); X2=torch.cat(Xs)
        if Wp2 is None:
            lam=1e-2*len(X2)
            Wp2=torch.linalg.solve(X2.T@X2+lam*torch.eye(D,device=DEV),
                                   X2.T@Yoh)
        CV=torch.stack([Y[flatA==k].mean(0) if (flatA==k).sum()>0
                        else Y.mean(0) for k in range(10)])
        LW={}
        for k in LINK:
            mk_=flatA==k
            Xk=X2[mk_]; Yk=Y[mk_]
            l2=1e-2*max(len(Xk),1)
            LW[k]=torch.linalg.solve(Xk.T@Xk+l2*torch.eye(D,device=DEV),
                                     Xk.T@Yk)
        S[f'a{li}L']=('attnd',li,CV,LW,Wp2)
        order2.append(f'a{li}L')
        print(f'fit a{li}L',flush=True)
    cur['clsmap']=clsC.reshape(R1-R0,256)
    L2C=evalM(FW[R0:R1],R1-R0,order2,ML)-baseC
    del cur['clsmap']
    import tiktoken as tk2
    enc4=tk2.get_encoding('gpt2')
    def classify2(Tk):
        n=Tk.shape[0]
        Mid=torch.zeros(n,256,dtype=torch.long)
        for r in range(n):
            toks=Tk[r,:257].tolist()
            for pos in range(256):
                t=toks[pos+1]; p=toks[pos]
                tg=enc4.decode([t]); pv=enc4.decode([p]); st=tg.strip()
                if st.isdigit() and not tg.startswith(' '): k=0
                elif st in (')',']') and any(b in enc4.decode(
                    toks[max(0,pos-60):pos+1]) for b in ('(','[')): k=1
                elif chr(10) in tg: k=2
                elif tg in ('.','!','?'): k=3
                elif tg==',': k=4
                elif (tg.startswith(' ') and st[:1].isupper() and
                      (pv.strip()[:1].isupper() if pv.strip()
                       else False)): k=5
                elif t==p: k=6
                elif (not tg.startswith(' ')) and st.isalpha(): k=7
                elif t in toks[:pos+1]: k=8
                else: k=9
                Mid[r,pos]=k
        return Mid
    cur['clsmap']=classify2(FR).to(DEV)
    L2F=evalM(FR,120,order2,ML)-baseF
    del cur['clsmap']
    W8banned=set(); W8RES=[]
    for wi in range(8):
        rws=[]; used=set()
        for di in range(3000,10000):
            if di in W8banned: continue
            tkr=enc3.encode_ordinary(dsf[di]['text'])
            for st0 in range(0,len(tkr)-513,513):
                row=tkr[st0:st0+513]
                if tuple(row[:32]) in seen: continue
                rws.append(row); used.add(di)
                if len(rws)>=120: break
            if len(rws)>=120: break
        assert len(rws)==120, f'window {wi} short: {len(rws)}'
        W8banned|=used
        for row in rws: seen.add(tuple(row[:32]))
        Wt=torch.tensor(rws,dtype=torch.long)
        bW=evalT(Wt,120,[])
        cur['clsmap']=classify2(Wt).to(DEV)
        fW=evalM(Wt,120,order2,ML)-bW
        del cur['clsmap']
        W8RES.append(round(fW,4))
        print(f'  window {wi} ({len(used)} docs): L2 fresh {fW:+.4f}',flush=True)
    inc=L2F-L1F
    print(f'L2 empirical: C {L2C:+.4f} | fresh {L2F:+.4f} | tail-attn '
          f'increment {inc:+.4f}',flush=True)
    if SEL.get('collect_asm'):
        genA=torch.Generator(device=DEV).manual_seed(32)
        TOKS=torch.cat([FW[i:i+4,:257] for i in range(CA,CB,4)]).to(DEV)
        for site in (5,6,7,8,9,10):
            Gm=torch.zeros(D,D,device=DEV,dtype=torch.float64)
            for b0 in range(0,TOKS.shape[0],4):
                bb=TOKS[b0:b0+4]
                cur['idx']=bb[:,:-1].contiguous(); cur['mode']='oracle'
                cur['lab']=clsA.reshape(CB-CA,256)[b0:b0+4].reshape(-1)
                hs=install(order2)+motif_hooks(ML)
                try:
                    with torch.no_grad():
                        x=F.rms_norm(m.transformer.wte(cur['idx']),(D,)); x0=x; v1=None
                        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
                        pdist=torch.softmax((30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30))[:,:-1].float(),-1)
                    for _sm in range(2):
                        y=torch.multinomial(pdist.reshape(-1,pdist.shape[-1]),1,generator=genA).view(pdist.shape[0],pdist.shape[1])
                        with torch.enable_grad():
                            x=F.rms_norm(m.transformer.wte(cur['idx']),(D,)); x0=x; v1=None; leaf=None
                            for _li,blk in enumerate(m.transformer.h):
                                if _li==site:
                                    x=x.detach().requires_grad_(True); leaf=x
                                x,v1=blk(x,v1,x0)
                            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
                            lp=F.log_softmax(lg[:,:-1],-1)
                            (-lp.gather(-1,y[...,None]).squeeze(-1))[:,SKIP8:].sum().backward()
                        g=leaf.grad[:,SKIP8:-1].reshape(-1,D).double(); Gm+=g.T@g
                        m.zero_grad(set_to_none=True)
                finally:
                    for h in hs: h.remove()
            _e,Qm=torch.linalg.eigh(Gm)
            SEL['P8'][site]=Qm.flip(1)[:,:8].float().contiguous()
            print(f'assembly-conditioned Fisher top-8 collected at site {site}',flush=True)
    pa=L2F<=2.75; pb=0.30<=inc<=0.55
    out={'L1_F':round(L1F,4),'L2_C':round(L2C,4),'L2_F':round(L2F,4),
         'increment':round(inc,4),
         'orig_s312_a':bool(pa),'orig_s312_b':bool(pb)}
    print(f"(a) L2 <= +2.75 fresh: {'HELD' if pa else 'FAILED'}")
    print(f"(b) increment in [0.30,0.55]: {'HELD' if pb else 'FAILED'}")
    out['fresh8']=W8RES
    out['runtime_s']=time.time()-t0
    return out

if __name__=='__main__':
    T00=time.time()
    SEL['mode']='norm'; SEL['collect_asm']=True; SEL['K']=2304; SEL['K69']=2304
    print('ARM 1/4: norm-2304 + assembly-conditioned Fisher at sites 5-10',flush=True)
    r_norm=main()
    SEL['collect_asm']=False
    SEL['mode']='fisher8'; SEL['K']=576; SEL['K69']=2304
    print('ARM 2/4: cond576@45, c6-c9 norm-2304 (S2134 reproduction)',flush=True)
    r_b=main()
    SEL['mode']='fisher8'; SEL['K']=576; SEL['K69']=576
    print('ARM 3/4: c6-c9 pruned to 576 by NORM rank',flush=True)
    r_n=main()
    SEL['mode']='fisher8all'; SEL['K']=576; SEL['K69']=576
    print('ARM 4/4: c6-c9 pruned to 576 by CONDITIONED rank',flush=True)
    r_c=main()
    import statistics as stt
    gb=[round(a-b,4) for a,b in zip(r_b['fresh8'],r_norm['fresh8'])]
    gp=[round(a-b,4) for a,b in zip(r_c['fresh8'],r_b['fresh8'])]
    gr=[round(a-b,4) for a,b in zip(r_c['fresh8'],r_n['fresh8'])]
    pa=stt.median(gp)>=0.0
    pb=stt.median(gr)>=0.0
    pc=abs(stt.median(gb)-0.1277)<=0.015
    res={'norm2304':r_norm,'cond576at45':r_b,'c69norm576':r_n,'c69cond576':r_c,
         'per_window_base_minus_norm':gb,'per_window_c69cond_minus_base':gp,
         'per_window_c69cond_minus_c69norm':gr,
         'pred_a_pruning_free':bool(pa),'pred_b_cond_beats_norm_rank':bool(pb),'pred_c_reproduces_cond576':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(res,open(OUT,'w'),indent=1)
    print(f'cond576@45 - norm per window: {gb}')
    print(f'c69cond576 - base per window: {gp}')
    print(f'c69cond576 - c69norm576 per window: {gr}')
    print(f"(a) c6-c9 pruning median {stt.median(gp):+.4f} >= 0: {'HELD' if pa else 'FAILED'}")
    print(f"(b) cond-vs-norm rank median {stt.median(gr):+.4f} >= 0: {'HELD' if pb else 'FAILED'}")
    print(f"(c) base median {stt.median(gb):+.4f} vs 0.1277 (tol 0.015): {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({res["runtime_s"]:.0f}s)')
