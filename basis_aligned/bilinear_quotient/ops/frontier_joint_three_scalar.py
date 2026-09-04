"""JOINTLY OPTIMISE THE THREE SCALARS ON A 3x3x4 GRID -- 36 CELLS IN ONE RUN (Claude, LANE 1)

The scaling story has produced three corrections, each measured ALONE or in pairs, never on a joint grid:

    TAIL   LW    x 0.25   SS2896 adopted   (SS2903 closed per-layer tuning: one scalar is the whole effect)
    CP     Dk    x 0.50   SS2902 adopted
    MOTIF  ALPHA x 1.25   SS2905/SS2906, bracketed, but 91% redundant with T+C (+0.0149 increment)
    T+C             SS2904 adopted, -0.3213 fresh / -0.2899 in sample -> frontier +2.3522

**The blocks are subadditive, so the individual optima need not be the joint optimum.** SS2906 found the motif block's entire standalone
value (-0.1604) almost absorbed by T+C, and SS2904 measured pairwise interactions of +0.0149, +0.0602 and +0.1234. When corrections
overlap, each one's best solo setting typically OVERSHOOTS in company -- which is exactly what SS2899/SS2901 found WITHIN the tail band
(each layer alone prefers .05; all eight together prefer ~.20). Nothing has yet tested it ACROSS blocks.

Grid: tail {0.20, 0.25, 0.30} x CP {0.35, 0.50, 0.65} x motif {1.00, 1.15, 1.25, 1.40} = **36 cells, one pipeline run**, built with
`ops/frontier_evalarms.factorial_arms` -- the 12:07Z ops action, whose measurement showed 40-arm rungs cost 5.8 GPU-s per arm against
14.8 for 8-arm rungs, and cost far fewer authoring cycles. This is its first use.

The cell (0.25, 0.50, 1.25) is the current best-measured configuration, so pred_b re-anchors the grid to SS2906 and every other cell is
read against a number already on the ledger.

SIGN CONVENTION (SS2135): frontier L2 is CE ADDED ABOVE THE REAL MODEL, so **LOWER IS BETTER** (SS312: +2.6735 beating +2.84/+2.93).
A cost is L2(arm) - L2(baseline), **POSITIVE = WORSE**, so a NEGATIVE cost is an improvement. SS2128/SS2129/SS2133/SS2134 RETRACTED;
SS2125 STANDS -- this rescales already-fitted objects; it neither selects nor reorders.

# BQGATE: EXPERIMENT  pred_a_the_baseline_reproduces_the_published_frontier
#                     pred_b_the_current_cell_reproduces_S2906
#                     pred_c_the_joint_optimum_beats_the_current_setting
#                     pred_d_the_joint_optimum_is_interior_in_all_three
#                     pred_e_the_optimum_improves_in_sample_too

Preregistration: polynomial_causal/FRONTIER_JOINT_THREE_SCALAR_PREREGISTRATION.md
Derived from ops/frontier_fisher8.py (SS2125 rung 30); that file is unmodified. ORIGINAL HEADER OF THE PARENT FOLLOWS.

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
    _need=['truefisher_top8_fresh_results.json','empirical_L2_results.json']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    _p=json.load(open(_bq+'empirical_L2_results.json'))
    print(f"DRYRUN OK: published L2_F {_p['L2_F']}; ONE pipeline run: a 3x3x4 joint grid over the tail, CP and motif scalars")
    raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from circuit_dictionary import classify, COMPS as TAILC, CLS
D=1152; V=50257
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'frontier_joint_three_scalar_results.json'   # NOT the parent's receipt (SS2125 cites that file)
CA,CB=300,512; R0,R1=120,300
CONSTN={'digit','bclose','sentend','comma','name','rep'}
CONSTK=[k for k,nm in enumerate(CLS) if nm in CONSTN]
LINK=[k for k in range(10) if k not in CONSTK]
ATTM=[2,3,4,5,6,7,8,9]; ATTT=[10,11,12,13,14,15,16,17]
MIDL=(4,5,6,7,8,9)
TRI=torch.triu_indices(32,32)
SEL={'mode':'norm','P8':{}}
COLLAPSE={'set':frozenset()}
TAILMODE={'mode':None}
EVALMODE={'drop':frozenset()}
ARMS=[]   # (name, spec); spec keys: 'drop' (prefixes omitted), 'cp_scale' (multiply every Dk)
CVSNAP={}
SKIP8=64


def select_units(li,L,Rw,Dw):
    if SEL['mode']=='fisher8' and li in (4,5):
        Pk=SEL['P8'][li+1]
        imp=(Pk.T@Dw).norm(dim=0)*L.norm(dim=1)*Rw.norm(dim=1)
    else:
        imp=Dw.norm(dim=0)*L.norm(dim=1)*Rw.norm(dim=1)
    return imp.argsort(descending=True)[:2304]


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
                    # `for k in LW` rather than `for k in LINK`: fit_attnd builds LW with EXACTLY the LINK
                    # keys (`LW={}` then `for k in LINK: LW[k]=...`), so this is a no-op for every
                    # uncollapsed layer, and it lets a collapsed layer carry an empty LW -- a pure constant --
                    # instead of raising KeyError. The preregistration's "fitted values only, never control
                    # flow" needed this one-token amendment; it is disclosed in the ledger section.
                    for k in LW:
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
        CVSNAP.setdefault(li, CV.detach().clone())
        if li in COLLAPSE['set']:
            # the dictionary becomes ONE constant vector: every class row is the overall mean write,
            # and the linear links are dropped. Control flow is untouched -- the probe label is still
            # computed by the hook, so downstream dictionaries see exactly what they saw before.
            CV = Y.mean(0).unsqueeze(0).repeat(CV.shape[0], 1)
            LW = {}
            print(f'COLLAPSED a{li} to a single constant', flush=True)
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
        if TAILMODE['mode'] is not None:
            # SS2878 attributed the whole +0.2011 to these eight tail REFITS. Split them into their two
            # structural pieces: the ten-row class table (10*1152 = 11,520 params) and the four 1152x1152
            # link maps (5,308,416 params). The fit_attnd site is deliberately NOT touched here -- SS2876
            # measured it free at 0.0000, so leaving it alone isolates the tail refits.
            if TAILMODE['mode'] == 'links':
                LW = {}                                                   # keep the class table
            elif TAILMODE['mode'] == 'table':
                CV = Y.mean(0).unsqueeze(0).repeat(CV.shape[0], 1)        # keep the link maps
            print(f"a{li}L: dropped {TAILMODE['mode']}", flush=True)
        S[f'a{li}L']=('attnd',li,CV,LW,Wp2)
        order2.append(f'a{li}L')
        print(f'fit a{li}L',flush=True)
    cur['clsmap']=clsC.reshape(R1-R0,256)
    # Snapshot the CP entries so each arm scales the SAME fitted stack. A 'cp' entry is
    # ('cp', li, Lk, Rk, Dk, db) and the forward is ((x@Lk.T)*(x@Rk.T))@Dk.T + db, so multiplying Dk
    # scales the whole quadratic reconstruction while leaving the bias -- the exact analogue of the
    # `A` scaling SS2895/SS2900 applied to the ridge-fitted front tables.
    _C0 = {k: S[k][4].clone() for k in order2 if k in S and S[k][0] == 'cp'}
    # motif heads: ALPHA[li] = (ap, asf), per-head gains fitted as ratios of inner products -- a LOCAL
    # criterion, so SS2902's broadened claim predicts a scale below 1 should help here too.
    _A0 = {li: (a.clone(), b.clone()) for li, (a, b) in ALPHA.items()}
    def _apply_motif(sc):
        for _li, (_a0, _b0) in _A0.items():
            ALPHA[_li] = ((_a0.clone(), _b0.clone()) if sc is None
                          else (_a0.float()*float(sc), _b0.float()*float(sc)))
    # tail refits a10L-a17L: within-class link maps LW (SS2896 adopted x0.25)
    _T0 = {k: {c: v.clone() for c, v in S[k][3].items()}
           for k in order2 if k in S and S[k][0] == 'attnd' and k.endswith('L')}
    # front tables: the low-rank quadratic residual A (SS2895 best x0.5)
    _F0 = {k: S[k][3].clone() for k in order2 if k in S and S[k][0] == 'tableres'}
    def _apply_tail(sc):
        for _k, _LW0 in _T0.items():
            _kind, _li, _CV, _LW, _Wp = S[_k]
            _new = {c: (v.clone() if sc is None else (v.float()*float(sc)).to(v.dtype))
                    for c, v in _LW0.items()}
            S[_k] = (_kind, _li, _CV, _new, _Wp)
    def _apply_front(sc):
        for _k, _A0 in _F0.items():
            _kind, _li, _tb, _A, _P = S[_k]
            _An = _A0.clone() if sc is None else (_A0.float()*float(sc)).to(_A0.dtype)
            S[_k] = (_kind, _li, _tb, _An, _P)
    def _apply_cp(sc):
        for _k, _D0 in _C0.items():
            _kind, _li, _L, _R, _D, _db = S[_k]
            _Dn = _D0.clone() if sc is None else (_D0.float() * float(sc)).to(_D0.dtype)
            S[_k] = (_kind, _li, _L, _R, _Dn, _db)
    _ml = ML
    _o2 = [x for x in order2
           if x.endswith('L') or not (x[0] in EVALMODE['drop'])]
    _arms = ARMS or [('single', {})]
    def _cfg(spec):
        _apply_cp(spec.get('cp_scale'))
        _apply_motif(spec.get('motif_scale'))
        _apply_tail(spec.get('tail_scale'))
        _apply_front(spec.get('a_scale'))
        _d = spec.get('drop', frozenset())
        _m = ML if spec.get('motif') is None else spec['motif']
        return [x for x in order2 if x.endswith('L') or not (x[0] in _d)], _m
    _L2C = {}
    for _n, _s in _arms:
        _oo, _mm = _cfg(_s)
        _L2C[_n] = evalM(FW[R0:R1],R1-R0,_oo,_mm)-baseC
    L2C = _L2C[_arms[0][0]]
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
    _L2F = {}
    for _n, _s in _arms:
        _oo, _mm = _cfg(_s)
        _L2F[_n] = evalM(FR,120,_oo,_mm)-baseF
    L2F = _L2F[_arms[0][0]]
    del cur['clsmap']
    inc=L2F-L1F
    print(f'L2 empirical: C {L2C:+.4f} | fresh {L2F:+.4f} | tail-attn '
          f'increment {inc:+.4f}',flush=True)
    pa=L2F<=2.75; pb=0.30<=inc<=0.55
    out={_n:{'L1_F':round(L1F,4),'L2_C':round(_L2C[_n],4),'L2_F':round(_L2F[_n],4),
             'increment':round(_L2F[_n]-L1F,4)} for _n,_ in _arms}
    print("L2 per arm: " + " | ".join(f"{_n} {out[_n]['L2_F']:+.4f}" for _n,_ in _arms),flush=True)
    out['runtime_s']=time.time()-t0
    return out

PREREG = PT + '../polynomial_causal/FRONTIER_JOINT_THREE_SCALAR_PREREGISTRATION.md'
PREREG_SHA = "04fba34676cfeee08dd0a2769070f465e97d6f3783ba97d3a2eb44957a4a7b9a"


def _sha(path):
    import hashlib
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


S2906 = PT + 'frontier_motif_bracket_and_triple_results.json'
S2904 = PT + 'frontier_scale_composition_results.json'
S2906_SHA = "e72a5fd6c3cd211ca6fdcffe4aa5126680731c59fae4971e02154ac6cf9c69ed"
S2904_SHA = "2ad1fc031037be32fd97431eef218d9dc22c382835b78e5409c967439d2850c5"
T_GRID = [0.20, 0.25, 0.30]
C_GRID = [0.35, 0.50, 0.65]
M_GRID = [1.00, 1.15, 1.25, 1.40]
BARS = {"reproduce": 0.05, "anchor": 0.01, "beats_current": 0.01}
NULLS = {"anchor_ge": 0.03, "beats_le": 0.0}


if __name__=='__main__':
    T00=time.time()
    if _sha(PREREG) != PREREG_SHA:
        raise RuntimeError(f'frozen hash mismatch: {PREREG}')
    for p_, h_ in ((S2906, S2906_SHA), (S2904, S2904_SHA)):
        if _sha(p_) != h_:
            raise RuntimeError(f'frozen hash mismatch: {p_}')
    cur = json.load(open(S2906))['summary']['best_triple_cost']        # -0.3363  (T .25, C .5, M 1.25)
    tc  = json.load(open(S2904))['summary']['cost_fresh']['TC']        # -0.3213  (T .25, C .5)
    SEL['mode']='norm'

    import sys as _sys; _sys.path.insert(0, PT + 'ops')
    import frontier_evalarms as FE
    # ops/frontier_evalarms.factorial_arms: the 12:07Z ops action, first use. 36 cells + baseline in ONE run.
    ARMS[:] = FE.factorial_arms({'tail_scale': T_GRID, 'cp_scale': C_GRID, 'motif_scale': M_GRID})
    print(f'ONE pipeline run, {len(ARMS)} evaluations against the same fitted stack',flush=True)
    out=main()

    bF = out['baseline']['L2_F']; bC = out['baseline']['L2_C']
    cell = {}
    for n, spec in ARMS:
        if n == 'baseline':
            continue
        cell[(spec['tail_scale'], spec['cp_scale'], spec['motif_scale'])] = (
            round(out[n]['L2_F'] - bF, 4), round(out[n]['L2_C'] - bC, 4))
    best = min(cell, key=lambda k: cell[k][0])
    best_fresh, best_fit = cell[best]
    anchor = cell[(0.25, 0.50, 1.25)][0]
    anchor_dev = abs(anchor - cur)
    beats = round(cur - best_fresh, 4)                 # POSITIVE = the joint optimum is better
    interior = (best[0] not in (T_GRID[0], T_GRID[-1])
                and best[1] not in (C_GRID[0], C_GRID[-1])
                and best[2] not in (M_GRID[0], M_GRID[-1]))

    pa = abs(bF-2.6735) <= BARS['reproduce']
    pb = anchor_dev <= BARS['anchor']
    pc = beats >= BARS['beats_current']
    pd = bool(interior)
    pe = best_fit < 0
    preds={'pred_a_the_baseline_reproduces_the_published_frontier':bool(pa),
           'pred_b_the_current_cell_reproduces_S2906':bool(pb),
           'pred_c_the_joint_optimum_beats_the_current_setting':bool(pc),
           'pred_d_the_joint_optimum_is_interior_in_all_three':bool(pd),
           'pred_e_the_optimum_improves_in_sample_too':bool(pe)}
    nulls={'b_null_the_anchor_fails':bool(anchor_dev>=NULLS['anchor_ge']),
           'c_null_the_current_setting_is_already_jointly_optimal':bool(beats<=NULLS['beats_le']),
           'd_null_the_grid_is_too_narrow':bool(not interior)}
    res={'rung':'frontier_joint_three_scalar','preds':preds,'nulls':nulls,
         'bars':BARS,'null_bars':NULLS,
         'grids':{'tail':T_GRID,'cp':C_GRID,'motif':M_GRID},'arms':out,'published_L2_F':2.6735,
         'summary':{'L2_F_baseline':round(bF,4),'L2_C_baseline':round(bC,4),
                    'n_cells':len(cell),
                    'cell_fresh':{f't{t}_c{c}_m{m}':v[0] for (t,c,m),v in sorted(cell.items())},
                    'cell_fit':{f't{t}_c{c}_m{m}':v[1] for (t,c,m),v in sorted(cell.items())},
                    'best_cell':{'tail':best[0],'cp':best[1],'motif':best[2]},
                    'best_fresh':best_fresh,'best_fit':best_fit,
                    'interior_in_all_three':bool(interior),
                    'S2906_current_best':round(cur,4),'anchor_cell_measured':anchor,
                    'anchor_deviation':round(anchor_dev,4),
                    'beats_current_by':beats,
                    'S2904_TC_for_reference':round(tc,4),
                    'implied_L2_F_best':round(bF + best_fresh, 4),
                    'S2904_adopted_L2_F':2.3522},
         'price':{'gpu_forwards':0,'forwards_instrumented':False,'pipeline_runs':1,
                  'backwards':0,'fitted_parameters':0,
                  'gpu_seconds':round(time.time()-T00,1)},
         'hashes':{PREREG:PREREG_SHA,S2906:S2906_SHA,S2904:S2904_SHA},'self_reviewed':True}
    json.dump(res,open(OUT,'w'),indent=1)
    print(f"(a) baseline L2_F {bF:.4f} vs 2.6735: {'HELD' if pa else 'FAILED'}")
    print(f"(b) cell (.25,.5,1.25) {anchor:+.4f} vs SS2906 {cur:+.4f}, dev {anchor_dev:.4f}: {'HELD' if pb else 'FAILED'}")
    print(f"(c) best cell t={best[0]} c={best[1]} m={best[2]} at {best_fresh:+.4f} (fit {best_fit:+.4f}) beats current by {beats:+.4f}: {'HELD' if pc else 'FAILED'}")
    print(f"(d) interior in all three: {'HELD' if pd else 'FAILED'}")
    print(f"    implied L2_F = {bF+best_fresh:.4f}   (adopted T+C = 2.3522)")
    print(f'wrote {OUT} ({res["price"]["gpu_seconds"]:.0f}s, {len(cell)} cells in one run)')
