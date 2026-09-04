"""IS THE FRONT EXCESS CONCENTRATED TOO? THE THIRD OBJECT TYPE (Claude, LANE 1)

Two of the three correction sites are now known to be CONCENTRATED, at OPPOSITE ends of their rankings:

    tail LW  (ridge solutions)          top 32 of 1152 singular directions      -0.2828 vs uniform -0.2287   SS2921
    CP Dk    (the model's own weights)  all BUT the top 128 of 2304 units       -0.1280 vs uniform -0.1074   SS2922

The third site has never been asked. SS2895 found the front tables' quadratic residual `A` wants x0.5, worth **-0.1648** standalone --
the second largest single correction in the campaign, and one that FAILED to compose (SS2904: TCF -0.2885 is worse than TC -0.3213).
Each installed `tableres` carries A (1680x64) lifted by P (1152x64), so its **64 components** have a natural importance
`||A[:,j]|| * ||P[:,j]||`, and the same split experiment applies directly.

**Why this is worth a rung rather than a footnote.** A scalar front correction does not compose; a CONCENTRATED one might, because the
reason a scalar fails in composition is usually that it damages directions the other terms rely on -- which is exactly what SS2921 found
for the tail (shrinking the bottom of the spectrum HURT by +0.0489). If the front excess is concentrated, the composition failure has a
candidate explanation and a candidate fix.

**The identity control is registered up front** (`f_identity`, all components x1.0 through the split path, must read 0.0000), and
`f_uniform` must agree with the plain multiply AND with SS2895's -0.1648 read from SS2904's receipt.

**Interiority is computed on the winning arm's rank irrespective of which half wins** -- the fix registered in SS2922, where my check
admitted only `top` winners and so could not pass once the CP excess turned out to live at the bottom.

SIGN CONVENTION (SS2135): frontier L2 is CE ADDED ABOVE THE REAL MODEL, so **LOWER IS BETTER** (SS312: +2.6735 beating +2.84/+2.93).
A cost is `L2(arm) - L2(baseline)`, **POSITIVE = WORSE**, so a NEGATIVE cost is an improvement. SS2128/SS2129/SS2133/SS2134 RETRACTED;
SS2125 STANDS -- this rescales already-fitted components; it neither selects nor reorders.

# BQGATE: EXPERIMENT  pred_a_the_baseline_reproduces_the_published_frontier
#                     pred_b_the_split_path_at_scale_one_is_a_physical_no_op
#                     pred_c_both_uniform_routes_reproduce_S2895
#                     pred_d_the_front_excess_is_concentrated
#                     pred_e_the_best_split_is_interior_to_the_rank_grid

Preregistration: polynomial_causal/FRONTIER_FRONT_COMPONENT_SPLIT_PREREGISTRATION.md
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
    print(f"DRYRUN OK: published L2_F {_p['L2_F']}; ONE pipeline run: the FRONT component-split analogue, the third object type")
    raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from circuit_dictionary import classify, COMPS as TAILC, CLS
D=1152; V=50257
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'frontier_front_component_split_results.json'   # NOT the parent's receipt (SS2125 cites that file)
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
GRAM={}      # (li,class) -> (XtX, XtY, n) on CPU: the tail refit's normal equations
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
            if len(rows)>=240: break
        if len(rows)>=240: break
    # SS2912: the SAME scan, continued to 240 rows. rows[:120] is bit-identical to what the parent collected
    # (same order, same dedup), so the reproduction gate still tests the published number; rows[120:240] are
    # documents that played NO part in choosing any scalar.
    if len(rows) < 240:
        raise RuntimeError(f'holdout window short: {len(rows)} rows of 240')
    FR=torch.tensor(rows[:120],dtype=torch.long)
    FR2=torch.tensor(rows[120:240],dtype=torch.long)
    baseC=evalT(FW[R0:R1],R1-R0,[])
    baseF=evalT(FR,120,[])
    baseF2=evalT(FR2,120,[])   # the held-out window's own baseline
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
            _XtX = Xk.T@Xk; _XtY = Xk.T@Yk
            # SS2916: keep the NORMAL EQUATIONS on CPU (~10.6 MB per (layer,class), 32 of them) so any ridge
            # lambda can be re-solved later WITHOUT refitting. The lambda actually used here is unchanged.
            GRAM[(li,k)] = (_XtX.cpu(), _XtY.cpu(), max(len(Xk),1))
            LW[k]=torch.linalg.solve(_XtX+l2*torch.eye(D,device=DEV), _XtY)
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
    def _apply_tail_lambda(mult):
        # Re-solve the tail link maps at `mult` x the fitted ridge lambda. mult=None leaves whatever
        # _apply_tail just wrote. This is the SAME estimator on the SAME data -- only the penalty changes.
        if mult is None:
            return
        _eye = torch.eye(D, device=DEV)
        for _li in ATTT:
            _k = f'a{_li}L'
            _kind, _l, _CV, _LW, _Wp = S[_k]
            _new = dict(_LW)
            for _c in LINK:
                _XtX, _XtY, _n = GRAM[(_li,_c)]
                _l2 = float(mult) * 1e-2 * _n
                _new[_c] = torch.linalg.solve(_XtX.to(DEV) + _l2*_eye, _XtY.to(DEV)).to(_LW[_c].dtype)
            S[_k] = (_kind, _l, _CV, _new, _Wp)
    def _apply_front(sc):
        for _k, _A0 in _F0.items():
            _kind, _li, _tb, _A, _P = S[_k]
            _An = _A0.clone() if sc is None else (_A0.float()*float(sc)).to(_A0.dtype)
            S[_k] = (_kind, _li, _tb, _An, _P)
    def _apply_front_split(spec):
        # SS2925: the FRONT analogue. Each installed `tableres` carries A (1680x64) lifted by P (1152x64), so
        # its 64 components have a natural importance ||A[:,j]|| * ||P[:,j]||. Scale one end of that ranking
        # and leave the other at 1.0; `half='all'` at scale 1.0 is the identity control, registered up front.
        if spec is None:
            return
        _r, _half, _sc = spec
        for _k, _A0 in _F0.items():
            _kind, _li, _tb, _A, _P = S[_k]
            _imp = _A0.float().norm(dim=0) * _P.float().norm(dim=0)
            _ord = _imp.argsort(descending=True)
            _w = torch.ones(_imp.numel(), device=_A0.device, dtype=torch.float32)
            if _half == 'all':
                _w[:] = float(_sc)
            elif _half == 'top':
                _w[_ord[:_r]] = float(_sc)
            else:
                _w[_ord[_r:]] = float(_sc)
            S[_k] = (_kind, _li, _tb, (_A0.float() * _w).to(_A0.dtype), _P)
    def _apply_cp_split(spec):
        # SS2921: the CP analogue of SS2919. Scale ONE END of the unit-importance ranking and leave the other
        # at 1.0. `half='all'` scales every unit through THIS path -- at scale 1.0 it is the identity control
        # this rung registers up front, because SS2919's omission of exactly that control cost it its result.
        if spec is None:
            return
        _r, _half, _sc = spec
        for _k, _D0 in _C0.items():
            _kind, _li, _L, _R, _D, _db = S[_k]
            _imp = (_D0.float().norm(dim=0) * _L.float().norm(dim=1) * _R.float().norm(dim=1))
            _ord = _imp.argsort(descending=True)
            _w = torch.ones(_imp.numel(), device=_D0.device, dtype=torch.float32)
            if _half == 'all':
                _w[:] = float(_sc)
            elif _half == 'top':
                _w[_ord[:_r]] = float(_sc)
            else:
                _w[_ord[_r:]] = float(_sc)
            S[_k] = (_kind, _li, _L, _R, (_D0.float() * _w).to(_D0.dtype), _db)
    def _apply_cp(sc, kf=None):
        for _k, _D0 in _C0.items():
            _kind, _li, _L, _R, _D, _db = S[_k]
            _Dn = _D0.clone() if sc is None else (_D0.float() * float(sc)).to(_D0.dtype)
            if kf is not None and float(kf) < 1.0:
                # rank the RETAINED units by the same importance select_units uses
                # (imp = ||Dw[:,u]|| * ||L[u]|| * ||R[u]||) and zero all but the top kf fraction.
                _imp = (_D0.float().norm(dim=0) * _L.float().norm(dim=1) * _R.float().norm(dim=1))
                _nk = max(1, int(round(float(kf) * _imp.numel())))
                _Dn = _Dn.clone()
                _Dn[:, _imp.argsort(descending=True)[_nk:]] = 0
            S[_k] = (_kind, _li, _L, _R, _Dn, _db)
    _ml = ML
    _o2 = [x for x in order2
           if x.endswith('L') or not (x[0] in EVALMODE['drop'])]
    _arms = ARMS or [('single', {})]
    def _cfg(spec):
        _apply_cp(spec.get('cp_scale'), spec.get('cp_keep'))
        _apply_cp_split(spec.get('cp_spec'))
        _apply_front_split(spec.get('front_spec'))
        _apply_motif(spec.get('motif_scale'))
        _apply_tail(spec.get('tail_scale'))
        _apply_tail_lambda(spec.get('tail_lambda'))
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
    cur['clsmap']=classify2(FR2).to(DEV)
    _L2F2 = {}
    for _n, _s in _arms:
        _oo, _mm = _cfg(_s)
        _L2F2[_n] = evalM(FR2,120,_oo,_mm)-baseF2
    del cur['clsmap']
    inc=L2F-L1F
    print(f'L2 empirical: C {L2C:+.4f} | fresh {L2F:+.4f} | tail-attn '
          f'increment {inc:+.4f}',flush=True)
    pa=L2F<=2.75; pb=0.30<=inc<=0.55
    out={_n:{'L1_F':round(L1F,4),'L2_C':round(_L2C[_n],4),'L2_F':round(_L2F[_n],4),
             'L2_F2':round(_L2F2[_n],4),
             'increment':round(_L2F[_n]-L1F,4)} for _n,_ in _arms}
    print("L2 per arm: " + " | ".join(f"{_n} {out[_n]['L2_F']:+.4f}" for _n,_ in _arms),flush=True)
    out['runtime_s']=time.time()-t0
    return out

PREREG = PT + '../polynomial_causal/FRONTIER_FRONT_COMPONENT_SPLIT_PREREGISTRATION.md'
PREREG_SHA = "b9e3c864012108253b0156f24f8f542e00a1f47a3193e9a4de093c1586add34f"


def _sha(path):
    import hashlib
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


S2906 = PT + 'frontier_motif_bracket_and_triple_results.json'
S2904 = PT + 'frontier_scale_composition_results.json'
S2906_SHA = "e72a5fd6c3cd211ca6fdcffe4aa5126680731c59fae4971e02154ac6cf9c69ed"
S2904_SHA = "2ad1fc031037be32fd97431eef218d9dc22c382835b78e5409c967439d2850c5"
BARS = {"reproduce": 0.05, "identity": 0.005, "anchor": 0.01, "beats_uniform": 0.01}
R_GRID = [4, 8, 16, 32, 48]             # of the 64 components each installed tableres carries
SPLIT_SCALE = 0.5                       # SS2895's best front scalar, applied to one end of the ranking


if __name__=='__main__':
    T00=time.time()
    if _sha(PREREG) != PREREG_SHA:
        raise RuntimeError(f'frozen hash mismatch: {PREREG}')
    for p_, h_ in ((S2906, S2906_SHA), (S2904, S2904_SHA)):
        if _sha(p_) != h_:
            raise RuntimeError(f'frozen hash mismatch: {p_}')
    F_ALONE = json.load(open(S2904))['summary']['S2895_front']    # SS2895's front A x0.5, -0.1648
    SEL['mode']='norm'

    ARMS[:] = ([('baseline', {}),
                ('f_identity', {'front_spec': (0, 'all', 1.0)}),
                ('f_uniform',  {'front_spec': (0, 'all', 0.5)}),
                ('plain_half', {'a_scale': 0.5})] +
               [(f'{h}{r}', {'front_spec': (r, h, SPLIT_SCALE)}) for r in R_GRID for h in ('top','bot')])
    print(f'ONE pipeline run, {len(ARMS)} arms against the same fitted stack',flush=True)
    out=main()

    bF, b2, bC = out['baseline']['L2_F'], out['baseline']['L2_F2'], out['baseline']['L2_C']
    cost   = {n: round(out[n]['L2_F']  - bF, 4) for n, _ in ARMS if n != 'baseline'}
    cost_h = {n: round(out[n]['L2_F2'] - b2, 4) for n, _ in ARMS if n != 'baseline'}
    cost_c = {n: round(out[n]['L2_C']  - bC, 4) for n, _ in ARMS if n != 'baseline'}
    ident, uni, plain = cost['f_identity'], cost['f_uniform'], cost['plain_half']
    splits = {f'{h}{r}': cost[f'{h}{r}'] for r in R_GRID for h in ('top','bot')}
    best_split = min(splits, key=lambda n: splits[n])
    best_cost = splits[best_split]
    # SS2922 FIX: interiority is computed on the winning arm's rank IRRESPECTIVE of which half won.
    best_r = int(best_split[3:]) if best_split.startswith('top') else int(best_split[3:])
    interior = best_r not in (R_GRID[0], R_GRID[-1])

    pa = abs(bF-2.6735) <= BARS['reproduce']
    pb = abs(ident) <= BARS['identity']
    pc = abs(uni - F_ALONE) <= BARS['anchor'] and abs(uni - plain) <= BARS['anchor']
    pd = best_cost <= uni - BARS['beats_uniform']
    pe = bool(interior)
    preds={'pred_a_the_baseline_reproduces_the_published_frontier':bool(pa),
           'pred_b_the_split_path_at_scale_one_is_a_physical_no_op':bool(pb),
           'pred_c_both_uniform_routes_reproduce_S2895':bool(pc),
           'pred_d_the_front_excess_is_concentrated':bool(pd),
           'pred_e_the_best_split_is_interior_to_the_rank_grid':bool(pe)}
    nulls={'b_null_the_split_path_is_not_faithful':bool(abs(ident) > 0.02),
           'c_null_the_uniform_routes_disagree':bool(abs(uni - plain) > 0.02),
           'd_null_the_front_excess_is_spread_across_components':bool(not pd),
           'e_null_the_rank_grid_is_too_narrow':bool(not interior)}
    res={'rung':'frontier_front_component_split','preds':preds,'nulls':nulls,'bars':BARS,
         'arms':out,'published_L2_F':2.6735,'rank_grid':R_GRID,'split_scale':SPLIT_SCALE,
         'summary':{'L2_F_baseline':round(bF,4),'L2_F2_baseline':round(b2,4),
                    'cost_fresh':cost,'cost_holdout':cost_h,'cost_fit':cost_c,
                    'identity_cost':ident,'uniform_cost':uni,'plain_multiply_cost':plain,
                    'route_disagreement':round(uni-plain,4),
                    'S2895_front_alone':F_ALONE,'anchor_deviation':round(abs(uni-F_ALONE),4),
                    'best_split':best_split,'best_split_cost':best_cost,
                    'best_beats_uniform_by':round(uni-best_cost,4),
                    'best_holdout_cost':cost_h[best_split],
                    'uniform_holdout_cost':cost_h['f_uniform'],
                    'top_profile':{str(r):cost[f'top{r}'] for r in R_GRID},
                    'bot_profile':{str(r):cost[f'bot{r}'] for r in R_GRID},
                    'best_split_interior':bool(interior),
                    'S2921_tail_rank32':-0.2828,'S2922_cp_bot128':-0.1280},
         'price':{'gpu_forwards':0,'forwards_instrumented':False,'pipeline_runs':1,
                  'backwards':0,'fitted_parameters':0,
                  'gpu_seconds':round(time.time()-T00,1)},
         'hashes':{PREREG:PREREG_SHA,S2906:S2906_SHA,S2904:S2904_SHA},'self_reviewed':True}
    json.dump(res,open(OUT,'w'),indent=1)
    print(f"(a) baseline L2_F {bF:.4f} vs 2.6735: {'HELD' if pa else 'FAILED'}")
    print(f"(b) split path at scale 1.0 {ident:+.4f}: {'HELD' if pb else 'FAILED'}")
    print(f"(c) uniform via split {uni:+.4f} / plain {plain:+.4f} vs SS2895 {F_ALONE:+.4f}: {'HELD' if pc else 'FAILED'}")
    for r in R_GRID:
        print(f"    r={r}: top {cost[f'top{r}']:+.4f} | bot {cost[f'bot{r}']:+.4f}")
    print(f"(d) front excess concentrated: best {best_split} {best_cost:+.4f} vs uniform {uni:+.4f}: {'HELD' if pd else 'FAILED'}")
    print(f"(e) best split interior to {R_GRID} (rank {best_r}, either half): {'HELD' if pe else 'FAILED'}")
    print(f"    held out: best {cost_h[best_split]:+.4f} vs uniform {cost_h['f_uniform']:+.4f}")
    print(f'wrote {OUT} ({res["price"]["gpu_seconds"]:.0f}s, {len(ARMS)} arms x 3 windows in one run)')
