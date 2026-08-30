"""RE-PRICE THE ARM'S PIECES IN CE-AT-THE-CLIFF: oracle-correct each compressed piece alone and rank by CE, not by error energy.

BENCHMARK_BACKLOG rung 23 (the 18:35 review's C). §2114/§2115 showed at head grain that a component's share of the
stream error and its share of the CE gap are nearly unrelated (the sink head: 71% / 19%). The benchmark's registry has
priced every stand-in by rel-MSE-style fidelity since §311. This does the same comparison at PIECE grain on the
certified empirical arm: for each compressed piece (a0 table, m0/m2/m3 table+residual, a1v table, m1 linear map,
c4-c9 CP-2304, the tail spans), drop its hook so the real module runs in its place -- everything else stays as
fitted (no refit; this is the oracle PRICE of the piece's error in situ, not §2102's matched-context arms) -- and
measure CE recovery. Alongside, capture each piece's own output error energy (arm output minus real output at that
module, same positions).

REGISTERED PREDICTIONS (recovery = (gap_arm - gap_piece_real) / gap_arm on the S2086 evaluation rows):
  (a) REL-MSE IS THE WRONG CURRENCY AT PIECE GRAIN: Spearman rho between pieces' CE recoveries and their own output
      error energies <= 0.5. If FALSE, energy does order pieces and the head-grain result does not transfer.
  (b) ADDITIVE: the sum of single-piece recoveries >= 0.8 x the recovery from making ALL pieces real (= 1 by
      construction, so the bar is: singles sum >= 0.8).
  (c) THE FRONT IS THE PRICE: the front pieces (a0, m0, a1v, m1, m2, m3) made real together recover >= 0.5 of the gap
      (§2103: 85% of the gap is error present at block 6).
  (d) REPRODUCTION GATE: cfgE block-6 rel-MSE reproduces 1.7415 within 0.10.

Self-reviewed. Writes piece_price_ladder_results.json.
"""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import os

# PLAN PRE-FLIGHT (LESSON 109): bilin18_joint_removal loads the model at import.
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['stream_error_profile_results.json','metric_units_fresh8_results.json']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    _p=json.load(open(_bq+'stream_error_profile_results.json'))['profile']
    print(f"DRYRUN OK: S2086 present (block 6 rel-MSE {_p['6']}); rebuilding the arm; single-piece oracle corrections, no refit")
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
OUT=PT+'piece_price_ladder_results.json'
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
    cfgE=build_arm(False)
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
    empS=collect(cfgE)

    # ================= single-piece oracle corrections (drop one hook) =================
    prof6=float(((empS[6]-real[6])**2).mean(0).sum())/max(float(real[6].var(0).sum()),1e-12)
    base=evalT(FW[R0:R1],R1-R0,[])
    gap_arm=evalT(FW[R0:R1],R1-R0,cfgE)-base
    PIECES=[p for p in cfgE]                       # e.g. a0, m0E, a1v, m1, m2E, m3E, c4..c9, tailE
    FRONT=[p for p in PIECES if p in ('a0','m0E','a1v','m1','m2E','m3E')]
    MODULE={'a0':('attn',0),'m0E':('mlp',0),'a1v':('attn',1),'m1':('mlp',1),'m2E':('mlp',2),'m3E':('mlp',3)}
    for li in range(4,10): MODULE[f'c{li}']=('mlp',li)
    def out_energy(active,piece):
        """energy of (arm output - real output) at the piece's own module on the collect rows; tail = sum over its MLPs"""
        kind,li=MODULE.get(piece,(None,None))
        mods=[m.transformer.h[li].mlp if kind=='mlp' else m.transformer.h[li].attn] if kind else [m.transformer.h[l].mlp for l in TAILC]
        def cap(active_):
            caps=[[] for _ in mods]; hs=install(active_)
            for i,mod in enumerate(mods):
                def h(mo,a,o,i=i): caps[i].append((o[0] if isinstance(o,tuple) else o).detach().float().reshape(-1,D).cpu())
                hs.append(mod.register_forward_hook(h))
            for r in range(R0,R1,8):
                bb=FW[r:r+4,:257].to(DEV); cur['idx']=bb[:,:-1].contiguous(); cur['mode']='oracle'; m(cur['idx'],bb[:,1:].contiguous())
            for h in hs: h.remove()
            return [torch.cat(c) for c in caps]
        A=cap(active); Rr=cap([])
        return float(sum(((a-r)**2).sum() for a,r in zip(A,Rr)))
    res={}
    for p in PIECES:
        active=[q for q in cfgE if q!=p]
        g=evalT(FW[R0:R1],R1-R0,active)-base
        e=out_energy(cfgE,p)
        res[p]={'gap_with_piece_real':round(g,5),'recovery':round((gap_arm-g)/max(gap_arm,1e-9),4),'own_output_error_energy':e}
        print(f"{p:6s} real: gap {g:+.4f} recovery {res[p]['recovery']:+.4f} | own output error energy {e:.3e}",flush=True)
    g_front=evalT(FW[R0:R1],R1-R0,[q for q in cfgE if q not in FRONT])-base
    rec_front=(gap_arm-g_front)/max(gap_arm,1e-9)
    def spearman(x,y):
        rx=torch.tensor(x).argsort().argsort().double(); ry=torch.tensor(y).argsort().argsort().double()
        rx,ry=rx-rx.mean(),ry-ry.mean(); return float((rx*ry).sum()/(rx.norm()*ry.norm()).clamp_min(1e-12))
    names=list(res); rho=spearman([res[p]['recovery'] for p in names],[res[p]['own_output_error_energy'] for p in names])
    ssum=sum(res[p]['recovery'] for p in names)
    pa=rho<=0.5; pb=ssum>=0.8; pc=rec_front>=0.5; pd=abs(prof6-1.7415)<=0.10
    ladder=sorted(names,key=lambda p:-res[p]['recovery'])
    out={'base_ce':round(base,5),'gap_arm':round(gap_arm,5),'pieces':res,'ladder_by_ce':ladder,
         'ladder_by_energy':sorted(names,key=lambda p:-res[p]['own_output_error_energy']),
         'spearman_recovery_vs_energy':round(rho,4),'sum_single_recoveries':round(ssum,4),'front_recovery':round(rec_front,4),
         'pred_a_energy_is_wrong_currency':bool(pa),'pred_b_additive':bool(pb),'pred_c_front_is_the_price':bool(pc),
         'pred_d_reproduces_S2086':bool(pd),'self_reviewed':True,'runtime_s':round(time.time()-t0,1)}
    json.dump(out,open(OUT,'w'),indent=1)
    print('ladder by CE:     '+' > '.join(f"{p} {res[p]['recovery']:.3f}" for p in ladder))
    print('ladder by energy: '+' > '.join(out['ladder_by_energy']))
    print(f"(d) block-6 rel-MSE {prof6:.4f}: {'HELD' if pd else 'FAILED'}")
    print(f"(a) rho(recovery, energy) {rho:+.3f} <= 0.5: {'HELD' if pa else 'FAILED'}")
    print(f"(b) sum of singles {ssum:.3f} >= 0.8: {'HELD' if pb else 'FAILED'}")
    print(f"(c) front recovery {rec_front:.3f} >= 0.5: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-t0:.0f}s)')

if __name__=='__main__': main()
