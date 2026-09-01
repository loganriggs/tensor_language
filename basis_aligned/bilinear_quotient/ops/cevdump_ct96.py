"""REGISTERED CLAIM: CORNER + TAIL r96 (rung 248) - the 11-certificate point becomes official.

CONVENTION (S2135): CE added above the real model; LOWER IS BETTER. S2346 measured {motifs exact, tail
r96} at census +0.0553 / fresh +0.0565 / valid 11 (84.5M pattern values). Identical rebuild; claim bars
registered, including the certificate count.
REGISTERED PREDICTIONS (anchors: the S2346 receipt):
  (a) REPRODUCTION: |census - 0.0553| <= 0.015.
  (b) CERTIFICATES: valid >= 11.
  (c) REPRODUCTION: |L2_F - 0.0565| <= 0.015.
NULL: noise. PRICE: ~211M values. Self-reviewed."""

















import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['frontier_tail_traj_results.json','circuits/BATTERY.json']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: registered claim corner+tail96')
    raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from circuit_dictionary import classify, COMPS as TAILC, CLS
D=1152; V=50257
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'cevdump_ct96_results.json'
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
    return imp.argsort(descending=True)[:(SEL.get('K',2304) if li in (4,5)
                                          else SEL.get('K69MAP',{}).get(li,SEL.get('K69',2304)))]


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
                    units=(x@Lk.T)*(x@Rk.T)
                    topk=int(SEL.get('cp_topk') or 0)
                    if topk and units.shape[-1]>topk:
                        chosen=(units.abs()*Dk.norm(dim=0)).topk(topk,dim=-1).indices
                        units=units*torch.zeros_like(units).scatter_(-1,chosen,1.0)
                        SEL['_cp_topk_observed']=int(chosen.shape[-1])
                    return ((units@Dk.T)+db).to(o_.dtype)
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
    if SEL.get('cp_swap'):
        _KK=SEL['cp_swap']
        for _li,_nm in ((0,'m0E'),(1,'m1'),(2,'m2E'),(3,'m3E')):
            _mlp=m.transformer.h[_li].mlp
            _L=_mlp.Left.weight.detach().float(); _R=_mlp.Right.weight.detach().float()
            _Dw=_mlp.Down.weight.detach().float(); _db=_mlp.Down_bias.detach().float()
            _kp=(_Dw.norm(dim=0)*_L.norm(dim=1)*_R.norm(dim=1)).argsort(descending=True)[:_KK]
            S[_nm]=('cp',_li,_L[_kp].contiguous(),_R[_kp].contiguous(),
                    _Dw[:,_kp].contiguous(),_db)
        print(f'  cp_swap: front MLPs m0-m3 -> CP top-{_KK} in place (tailE remains table-front-fit)',flush=True)

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
    if SEL.get('qk_tail'): SEL['qk_tail_on']=True
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
    def head_z(at,X2,v1,li=None):
        B=X2.shape[0]
        q=at.c_q(X2).view(B,T,9,128); k=at.c_k(X2).view(B,T,9,128)
        q2=at.c_q2(X2).view(B,T,9,128); k2=at.c_k2(X2).view(B,T,9,128)
        if li is not None and li in SEL.get('_VR',{}):
            v=torch.zeros(B,T,9,128,device=DEV,dtype=X2.dtype)
            Xf=X2.float()
            for hd,(left_v,right_v) in SEL['_VR'][li].items():
                v[:,:,hd]=((Xf@right_v.T)@left_v.T).to(X2.dtype)
        else:
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
            z,vm=head_z(at,X2,v1,li)
            vp=torch.zeros_like(vm); vp[:,1:]=vm[:,:-1]
            vp=vp.permute(0,2,1,3); vs=vm.permute(0,2,1,3)
            num+=(z*vp).sum((0,2,3)); den+=(vp*vp).sum((0,2,3))
            nums+=(z*vs).sum((0,2,3)); dens+=(vs*vs).sum((0,2,3))
        ALPHA[li]=(num/den.clamp_min(1e-9),nums/dens.clamp_min(1e-9))
        if SEL.get('ov_res'):
            _HDim=None
            _ph9=prevh.get(li,[]); _sh9=selfh.get(li,[])
            _AG={('p',hd):None for hd in _ph9}; _AG.update({('s',hd):None for hd in _sh9})
            _BG={k:None for k in _AG}
            for X2,v1 in zip(caps[li]['x'],caps[li]['v1']):
                z,vm=head_z(at,X2,v1,li)
                vp=torch.zeros_like(vm); vp[:,1:]=vm[:,:-1]
                vp=vp.permute(0,2,1,3); vs=vm.permute(0,2,1,3)
                _HDim=z.shape[3]
                for _k9 in _AG:
                    _g,_hd=_k9
                    _vv=(vp if _g=='p' else vs)[:,_hd].reshape(-1,_HDim).float()
                    _al=(ALPHA[li][0] if _g=='p' else ALPHA[li][1])[_hd]
                    _zz=z[:,_hd].reshape(-1,_HDim).float()-_al*_vv
                    if _AG[_k9] is None:
                        _AG[_k9]=_vv.T@_vv; _BG[_k9]=_vv.T@_zz
                    else:
                        _AG[_k9]+=_vv.T@_vv; _BG[_k9]+=_vv.T@_zz
            _WR={}
            for _k9 in _AG:
                _lam9=1e-2*float(_AG[_k9].diagonal().mean().clamp_min(1e-6))*_HDim
                _Rf=torch.linalg.solve(_AG[_k9]+_lam9*torch.eye(_HDim,device=DEV),_BG[_k9])
                _U,_S9,_Vh9=torch.linalg.svd(_Rf)
                _WR[_k9]=(_U[:,:8]*_S9[:8])@_Vh9[:8]
            SEL.setdefault('_WR',{})[li]=_WR
        caps[li]=None
    if SEL.get('value_r'):
        _value_rank=int(SEL['value_r'])
        _value_context=SEL.get('value_context_covariances')
        if not _value_context:
            raise ValueError('value_r requires value_context_covariances')
        SEL['_VR']={}
        SEL['_VALUE_METRIC']='context_rrr'
        SEL['_VALUE_CONTEXT_LAYERS']=tuple(sorted(int(x) for x in _value_context))
        for _li,_cov0 in sorted(_value_context.items()):
            if int(_li) not in range(2,18):
                raise ValueError(f'bad value context layer {_li}')
            _cov=_cov0.to(DEV).float()
            if _cov.shape != (D,D):
                raise ValueError(f'bad value context covariance layer {_li}: {_cov.shape}')
            _ev,_evec=torch.linalg.eigh(0.5*(_cov+_cov.T))
            _ord=torch.argsort(_ev,descending=True); _ev=_ev[_ord]; _evec=_evec[:,_ord]
            _floor=float(_ev[0])*1e-6; _safe=_ev.clamp_min(_floor)
            _sqrt=(_evec*_safe.sqrt())@_evec.T
            _invsqrt=(_evec*_safe.rsqrt())@_evec.T
            _weight=m.transformer.h[int(_li)].attn.c_v.weight.detach().float()
            _heads={}
            for _hd in range(9):
                _map=_weight[_hd*128:(_hd+1)*128]
                _u,_s,_vh=torch.linalg.svd(_map@_sqrt,full_matrices=False)
                _heads[_hd]=((_u[:,:_value_rank]*_s[:_value_rank]).contiguous(),
                             (_vh[:_value_rank]@_invsqrt).contiguous())
            SEL['_VR'][int(_li)]=_heads
        print(f'  value rank-{_value_rank} context factors built for '
              f'{sum(len(x) for x in SEL["_VR"].values())} heads',flush=True)

    if SEL.get('qk_r'):
        _r=int(SEL['qk_r'])
        SEL['_QKR']={}
        SEL['_QK_INDEX_SETS']={}
        _qk_context=SEL.get('qk_context_covariances')
        _qk_context_metric={}
        if _qk_context:
            SEL['_QK_METRIC']='context_rrr'
            for _li,_cov0 in _qk_context.items():
                _cov=_cov0.to(DEV).float()
                if _cov.shape != (D,D):
                    raise ValueError(f'bad QK context covariance layer {_li}: {_cov.shape}')
                _ev,_evec=torch.linalg.eigh(0.5*(_cov+_cov.T))
                _ord=torch.argsort(_ev,descending=True); _ev=_ev[_ord]; _evec=_evec[:,_ord]
                _floor=float(_ev[0])*1e-6; _safe=_ev.clamp_min(_floor)
                _sqrt=(_evec*_safe.sqrt())@_evec.T
                _invsqrt=(_evec*_safe.rsqrt())@_evec.T
                _qk_context_metric[int(_li)]=(_sqrt,_invsqrt)
            SEL['_QK_CONTEXT_LAYERS']=tuple(sorted(int(x) for x in _qk_context_metric))
        else:
            SEL['_QK_METRIC']='weight_svd'

        _qk_storage_dtype=SEL.get('qk_factor_storage_dtype')
        if _qk_storage_dtype not in (None,'float32','float16'):
            raise ValueError(f'unsupported qk_factor_storage_dtype={_qk_storage_dtype!r}')
        SEL['_QK_STORAGE_DTYPE']='float16' if _qk_storage_dtype=='float16' else 'float32'

        def _store_qk_factor(_pair):
            if _qk_storage_dtype=='float16':
                return tuple(_value.to(torch.float16).contiguous() for _value in _pair)
            return _pair

        def _factor_qk_map(_M,_li,_idx):
            if _li not in _qk_context_metric:
                _U,_S,_Vh=torch.linalg.svd(_M,full_matrices=False)
                return _store_qk_factor(((_U[:,_idx]*_S[_idx]).contiguous(),
                                         _Vh[_idx].contiguous()))
            if int(SEL.get('qk_extra_tail') or 0):
                raise ValueError('qk_extra_tail is incompatible with context-RRR QK factors')
            _sqrt,_invsqrt=_qk_context_metric[_li]
            _U,_S,_Vh=torch.linalg.svd(_M@_sqrt,full_matrices=False)
            _rank=int(_idx.numel())
            return _store_qk_factor(((_U[:,:_rank]*_S[:_rank]).contiguous(),
                                     (_Vh[:_rank]@_invsqrt).contiguous()))
        for _li in range(2,10):
            _r=int(SEL.get('qk_rmap',{}).get(_li,SEL['qk_r']))
            _extra=int(SEL.get('qk_extra_tail') or 0)
            if _extra:
                if _r+_extra>128 or _r>128-_extra:
                    raise ValueError(f'overlapping QK index request r={_r}, extra={_extra}')
                _idx=torch.cat([torch.arange(_r,device=DEV),
                                torch.arange(128-_extra,128,device=DEV)])
            else:
                _idx=torch.arange(_r,device=DEV)
            SEL['_QK_INDEX_SETS'][_li]=tuple(int(x) for x in _idx.cpu())
            _at=m.transformer.h[_li].attn
            _heads=list(prevh.get(_li,[]))+list(selfh.get(_li,[]))
            if not _heads: continue
            _d={}
            for _hd in _heads:
                _fac=[]
                for _W in (_at.c_q,_at.c_k,_at.c_q2,_at.c_k2):
                    _M=_W.weight[_hd*128:(_hd+1)*128,:].detach().float()
                    _fac.append(_factor_qk_map(_M,_li,_idx))
                _d[_hd]=_fac
            SEL['_QKR'][_li]=_d
        print(f'  QK rank-{_r} factors built for all motif heads',flush=True)
        if SEL.get('qk_tail'):
            for _li in range(10,18):
                _r=int(SEL.get('qk_rmap',{}).get(_li,SEL['qk_r']))
                _extra=int(SEL.get('qk_extra_tail') or 0)
                if _extra:
                    if _r+_extra>128 or _r>128-_extra:
                        raise ValueError(f'overlapping QK index request r={_r}, extra={_extra}')
                    _idx=torch.cat([torch.arange(_r,device=DEV),
                                    torch.arange(128-_extra,128,device=DEV)])
                else:
                    _idx=torch.arange(_r,device=DEV)
                SEL['_QK_INDEX_SETS'][_li]=tuple(int(x) for x in _idx.cpu())
                _at=m.transformer.h[_li].attn
                _d={}
                for _hd in range(9):
                    _fac=[]
                    for _W in (_at.c_q,_at.c_k,_at.c_q2,_at.c_k2):
                        _M=_W.weight[_hd*128:(_hd+1)*128,:].detach().float()
                        _fac.append(_factor_qk_map(_M,_li,_idx))
                    _d[_hd]=_fac
                SEL['_QKR'][_li]=_d
            print(f'  QK rank-{_r} factors built for tail blocks 10-17',flush=True)
        def qkz(at,X2,li,vm):
            B=X2.shape[0]
            Q=torch.zeros(B,T,9,128,device=DEV); K=torch.zeros_like(Q)
            Q2=torch.zeros_like(Q); K2=torch.zeros_like(Q)
            for hd,(fq,fk,fq2,fk2) in SEL['_QKR'][li].items():
                Xf=X2.float()
                Q[:,:,hd]=(Xf@fq[1].float().T)@fq[0].float().T
                K[:,:,hd]=(Xf@fk[1].float().T)@fk[0].float().T
                Q2[:,:,hd]=(Xf@fq2[1].float().T)@fq2[0].float().T
                K2[:,:,hd]=(Xf@fk2[1].float().T)@fk2[0].float().T
            cos,sin=at.rotary(Q)
            qn=F.rms_norm(Q,(128,)); kn=F.rms_norm(K,(128,))
            qn,kn=are(qn,cos,sin),are(kn,cos,sin)
            q2n=F.rms_norm(Q2,(128,)); k2n=F.rms_norm(K2,(128,))
            q2n,k2n=are(q2n,cos,sin),are(k2n,cos,sin)
            sc=torch.einsum('bqhd,bkhd->bhqk',qn.float(),kn.float())/128
            sc2=torch.einsum('bqhd,bkhd->bhqk',q2n.float(),k2n.float())/128
            pat=(sc*sc2)*torch.tril(torch.ones(T,T,device=DEV))
            return torch.einsum('bhqk,bkhd->bhqd',pat,vm)
        SEL['_qkz']=qkz
        if SEL.get('qk_tail'):
            for _li in range(10,18):
                _at=m.transformer.h[_li].attn
                def _th(mo_,args,out,at=_at,li=_li):
                    if not SEL.get('qk_tail_on'): return None
                    y,v1r=out
                    X2=args[0]; v1=args[1] if args[1] is not None else v1r
                    z,vm=head_z(at,X2,v1,li)
                    zq=qkz(at,X2,li,vm)
                    B=X2.shape[0]
                    ynew=at.c_proj(zq.transpose(1,2).contiguous().view(B,T,-1).to(X2.dtype))
                    return (ynew,v1r)
                _at.register_forward_hook(_th)
            print('  tail QK hooks registered (gated)',flush=True)

    def motif_hooks(layers):
        hs2=[]
        for li in layers:
            if li not in set(list(prevh)+list(selfh)): continue
            at=m.transformer.h[li].attn
            ap,asf=ALPHA[li]
            ph=prevh.get(li,[]); sh=selfh.get(li,[])
            def h(mo_,args,out,at=at,ph=ph,sh=sh,ap=ap,asf=asf,li=li):
                y,v1r=out
                X2=args[0]; v1=args[1] if args[1] is not None else v1r
                z,vm=head_z(at,X2,v1,li)
                vp=torch.zeros_like(vm); vp[:,1:]=vm[:,:-1]
                vp=vp.permute(0,2,1,3); vs=vm.permute(0,2,1,3)
                if SEL.get('qk_r'):
                    _zq=SEL['_qkz'](at,X2,li,vm)
                    for hd in ph: z[:,hd]=_zq[:,hd]
                    for hd in sh: z[:,hd]=_zq[:,hd]
                elif SEL.get('ov_res'):
                    _WRl=SEL['_WR'][li]
                    for hd in ph: z[:,hd]=ap[hd]*vp[:,hd]+vp[:,hd]@_WRl[('p',hd)]
                    for hd in sh: z[:,hd]=asf[hd]*vs[:,hd]+vs[:,hd]@_WRl[('s',hd)]
                else:
                    for hd in ph: z[:,hd]=ap[hd]*vp[:,hd]
                    for hd in sh: z[:,hd]=asf[hd]*vs[:,hd]
                B=X2.shape[0]
                ynew=at.c_proj(z.transpose(1,2).contiguous()
                               .view(B,T,-1).to(X2.dtype))
                return (ynew,v1r)
            hs2.append(at.register_forward_hook(h))
        return hs2
    def final_mlp_projector_hooks(active):
        """Optional literal MLP-output factors, enabled only after dictionaries fit.

        An empty ``active`` denotes the native side of a paired evaluation and
        must remain unprojected.  This keeps extra-eval and fresh baselines live.
        """
        if (not active or not SEL.get('_enable_final_mlp_projectors')
                or not SEL.get('final_mlp_projectors')):
            return []
        handles=[]
        observed={}
        for li,(q0,mu0) in sorted(SEL['final_mlp_projectors'].items()):
            q=q0.to(DEV).float(); mu=mu0.to(DEV).float()
            if q.ndim != 2 or q.shape[0] != D or mu.shape != (D,):
                raise ValueError(f'bad MLP projector layer {li}: q={tuple(q.shape)} mu={tuple(mu.shape)}')
            observed[int(li)]=int(q.shape[1])
            def h(_mo,_args,out,q=q,mu=mu):
                z=out.float()-mu
                return (mu+(z@q)@q.T).to(out.dtype)
            handles.append(m.transformer.h[li].mlp.register_forward_hook(h))
        SEL['_final_mlp_projectors_observed']=observed
        return handles
    def final_mlp_input_program_hooks(active):
        """Optional shared-input bilinear MLP factors, final candidate side only."""
        if (not active or not SEL.get('_enable_final_mlp_input_programs')
                or not SEL.get('final_mlp_input_programs')):
            return []
        handles=[]
        observed={}
        for li,program0 in sorted(SEL['final_mlp_input_programs'].items()):
            encoder=program0['encoder'].to(DEV).float()
            left=program0['left'].to(DEV).float()
            right=program0['right'].to(DEV).float()
            down=program0['down'].to(DEV).float()
            bias=program0['bias'].to(DEV).float()
            rank=int(encoder.shape[0])
            if (encoder.shape != (rank,D) or left.shape != (4608,rank)
                    or right.shape != (4608,rank) or down.shape != (D,4608)
                    or bias.shape != (D,)):
                raise ValueError(f'bad MLP input program layer {li}: '
                                 f'{tuple(encoder.shape)} {tuple(left.shape)} '
                                 f'{tuple(right.shape)} {tuple(down.shape)} {tuple(bias.shape)}')
            observed[int(li)]=rank
            def h(_mo,args,out,encoder=encoder,left=left,right=right,down=down,bias=bias):
                x=args[0].float(); z=x@encoder.T
                hidden=(z@left.T)*(z@right.T)
                return (hidden@down.T+bias).to(out.dtype)
            handles.append(m.transformer.h[li].mlp.register_forward_hook(h))
        SEL['_final_mlp_input_programs_observed']=observed
        return handles
    def evalV(TOK,N,active,mlayers):
        hs=(install(active)+motif_hooks(mlayers)+final_mlp_projector_hooks(active)
            +final_mlp_input_program_hooks(active))
        ces=[]; final_states=[]
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
            hfinal=F.rms_norm(x,(D,))
            if SEL.get('capture_final_state'):
                final_states.append(hfinal.detach().to(torch.float16).cpu())
            lg=(30*torch.tanh(m.lm_head(hfinal)/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        for h in hs: h.remove()
        if SEL.get('capture_final_state'):
            SEL['final_state']=torch.cat(final_states).reshape(-1,D).contiguous()
        return torch.cat(ces)
    def evalM(TOK,N,active,mlayers):
        hs=(install(active)+motif_hooks(mlayers)+final_mlp_projector_hooks(active)
            +final_mlp_input_program_hooks(active))
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
    ML=[l for l in range(2,10) if l not in SEL.get('motif_off',())]
    L1C=evalM(FW[R0:R1],R1-R0,cfgF,ML)-baseC
    L1F=evalM(FR,120,cfgF,ML)-baseF
    print(f'L1 (+38 heads): C {L1C:+.4f} | fresh {L1F:+.4f}',flush=True)
    # tail-attention dicts refit under (empirical base + motifs)
    Yoh=torch.zeros(len(flatA),10,device=DEV)
    Yoh[torch.arange(len(flatA)),flatA]=1.0
    if SEL.get('tail_traj'):
        _YR={li:[] for li in range(10,18)}
        hs=[]
        for _li in range(10,18):
            def _mkr(li=_li):
                def h(mo_,i_,o_):
                    _YR[li].append(o_[0].detach().reshape(-1,D).float())
                return h
            hs.append(m.transformer.h[_li].attn.register_forward_hook(_mkr()))
        for _i in range(CA,CB,4):
            _bb=FW[_i:_i+4,:257].to(DEV)
            cur['idx']=_bb[:,:-1].contiguous(); cur['mode']='oracle'
            cur['lab']=clsA[_i-CA:_i-CA+4].reshape(-1)
            m(cur['idx'],_bb[:,1:].contiguous())
        for h in hs: h.remove()
        SEL['_YR']={li:torch.cat(v) for li,v in _YR.items()}
        print('  captured real-trajectory tail-attention targets',flush=True)

    if SEL.get('tailE_rebuild'):
        _capsT={li:[] for li in TAILC}; _capsI={li:[] for li in TAILC}
        _actm=[nm for nm in cfgF if nm!='tailE']
        hs=install(_actm)+motif_hooks(ML)
        for _li in TAILC:
            def _mkA(li=_li):
                def h(mo,i_,o_):
                    _capsT[li].append(o_.detach().reshape(-1,D).float())
                    _capsI[li].append(i_[0].detach().reshape(-1,D).float())
                return h
            hs.append(m.transformer.h[_li].mlp.register_forward_hook(_mkA()))
        for _i in range(CA,CB,4):
            _bb=FW[_i:_i+4,:257].to(DEV)
            cur['idx']=_bb[:,:-1].contiguous(); cur['mode']='oracle'
            cur['lab']=clsA[_i-CA:_i-CA+4].reshape(-1)
            m(cur['idx'],_bb[:,1:].contiguous())
        for h in hs: h.remove()
        if SEL.get('tailE_traj'):
            _capsT={li:[] for li in TAILC}
            hs=[]
            for _li in TAILC:
                def _mkB(li=_li):
                    def h(mo,i_,o_):
                        _capsT[li].append(o_.detach().reshape(-1,D).float())
                    return h
                hs.append(m.transformer.h[_li].mlp.register_forward_hook(_mkB()))
            for _i in range(CA,CB,4):
                _bb=FW[_i:_i+4,:257].to(DEV)
                cur['idx']=_bb[:,:-1].contiguous(); cur['mode']='oracle'
                m(cur['idx'],_bb[:,1:].contiguous())
            for h in hs: h.remove()
        _X10=torch.cat(_capsI[TAILC[0]])
        _lam=1e-2*len(_X10)
        _Wp=torch.linalg.solve(_X10.T@_X10+_lam*torch.eye(D,device=DEV),_X10.T@Yoh)
        _DICT={}; _LIN={}
        for _li in TAILC:
            _Q,_=spans[_li]; _C=torch.cat(_capsT[_li])@_Q
            _DICT[_li]=torch.stack([_C[flatA==k].mean(0) if (flatA==k).sum()>0
                                    else _C.mean(0) for k in range(10)])
            _LIN[_li]={}
            _Xc=torch.cat(_capsI[_li])
            for _k in (8,9):
                _mk9=flatA==_k
                _Xk=_Xc[_mk9]; _Ck=_C[_mk9]
                _l2=1e-2*max(len(_Xk),1)
                _LIN[_li][_k]=torch.linalg.solve(_Xk.T@_Xk+_l2*torch.eye(D,device=DEV),_Xk.T@_Ck)
            _capsT[_li]=None; _capsI[_li]=None
        S['tailE']=('tail',_Wp,_DICT,_LIN)
        print('  tailE rebuilt under the CP-front+motif frame'
              +(' with trajectory targets' if SEL.get('tailE_traj') else ''),flush=True)

    order2=[nm for nm in cfgF
            if not (SEL.get('drop_tailE') and nm=='tailE')
            and not (SEL.get('drop_a1v') and nm=='a1v')
            and not (SEL.get('drop_a0') and nm=='a0')]
    SEL['_ORDER2']=tuple(order2)
    Wp2=None
    ML=[l for l in range(2,10) if l not in SEL.get('motif_off',())]
    for li in range(10,18):
        if li in SEL.get('skipset',()): continue
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
        Y=(SEL['_YR'][li] if SEL.get('tail_traj') else torch.cat(Ys)); X2=torch.cat(Xs)
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
    # All learned program dictionaries are now frozen.  Optional external MLP
    # factors compose only in final candidate evaluations below, never while
    # fitting those dictionaries.
    SEL['_enable_final_mlp_projectors']=bool(SEL.get('final_mlp_projectors'))
    SEL['_enable_final_mlp_input_programs']=bool(SEL.get('final_mlp_input_programs'))
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
    if SEL.get('clsdmg'):
        _ROWS=SEL.get('ext_rows',FR)
        cur['clsmap']=classify2(_ROWS).to(DEV)
        _abl_handle=None
        if SEL.get('ablate_on_census'):
            if '_ablh' not in SEL:
                raise RuntimeError('ablate_on_census requires SEL[_ablh]')
            # Register after the permanent tail-QK hook so the intervention
            # replaces the compiled module output rather than being overwritten.
            _abl_handle=m.transformer.h[16].attn.register_forward_hook(SEL['_ablh'])
            SEL['abl_on']=True
        try:
            _variants=SEL.get('final_mlp_projector_variants')
            _input_variants=SEL.get('final_mlp_input_program_variants')
            if _variants and _input_variants:
                raise ValueError('cannot combine final MLP projector and input-program variants')
            if _input_variants:
                _primary=str(SEL.get('final_mlp_input_primary_variant'))
                if _primary not in _input_variants:
                    raise ValueError(f'final_mlp_input_primary_variant {_primary!r} not in variants')
                SEL['_final_mlp_input_variant_cevs']={}
                SEL['_final_mlp_input_variant_observed']={}
                for _name,_programs in _input_variants.items():
                    SEL['final_mlp_input_programs']=_programs
                    _value=evalV(_ROWS,_ROWS.shape[0],order2,ML).detach().cpu()
                    SEL['_final_mlp_input_variant_cevs'][str(_name)]=_value
                    SEL['_final_mlp_input_variant_observed'][str(_name)]=dict(
                        SEL.get('_final_mlp_input_programs_observed',{}))
                SEL['final_mlp_input_programs']=_input_variants[_primary]
                SEL['cev']=SEL['_final_mlp_input_variant_cevs'][_primary]
            elif _variants:
                _primary=str(SEL.get('final_mlp_primary_variant'))
                if _primary not in _variants:
                    raise ValueError(f'final_mlp_primary_variant {_primary!r} not in variants')
                SEL['_final_mlp_variant_cevs']={}
                SEL['_final_mlp_variant_observed']={}
                for _name,_projectors in _variants.items():
                    SEL['final_mlp_projectors']=_projectors
                    _value=evalV(_ROWS,_ROWS.shape[0],order2,ML).detach().cpu()
                    SEL['_final_mlp_variant_cevs'][str(_name)]=_value
                    SEL['_final_mlp_variant_observed'][str(_name)]=dict(
                        SEL.get('_final_mlp_projectors_observed',{}))
                SEL['final_mlp_projectors']=_variants[_primary]
                SEL['cev']=SEL['_final_mlp_variant_cevs'][_primary]
            else:
                SEL['cev']=evalV(_ROWS,_ROWS.shape[0],order2,ML).detach().cpu()
        finally:
            if SEL.get('ablate_on_census'):
                SEL['abl_on']=False
                _abl_handle.remove()
        SEL['clsflat']=cur['clsmap'].reshape(-1).cpu()
    if SEL.get('head16'):
        HD16=D//9
        def _zh(hh):
            def pre(mod,args):
                x=args[0].clone(); x[...,hh*HD16:(hh+1)*HD16]=0
                return (x,)+tuple(args[1:])
            return m.transformer.h[16].attn.c_proj.register_forward_pre_hook(pre)
        dh={}
        for hh in list(range(9))+['all']:
            hks=[_zh(k2) for k2 in (range(9) if hh=='all' else [hh])]
            v=evalM(FR,120,order2,ML)-baseF
            for k3 in hks: k3.remove()
            dh[str(hh)]=round(v-L2F,4)
            print(f'  zero attn16 head {hh}: L2 fresh {v:+.4f}  (d={v-L2F:+.4f})',flush=True)
        SEL['head16_result']=dh
    del cur['clsmap']
    if SEL.get('prefix_tail'):
        cur['clsmap']=classify2(FR).to(DEV)
        _pl=[]
        for _kk in range(0,9):
            _act=order2[:len(order2)-8+_kk]
            _v=evalM(FR,120,_act,ML)-baseF
            _pl.append(round(_v,4))
            print(f'  prefix +{_kk} tail-attn dicts: L2 fresh {_v:+.4f}',flush=True)
        del cur['clsmap']
        SEL['prefix_result']={'prefix':_pl,'marginals':[round(_pl[_i+1]-_pl[_i],4) for _i in range(8)]}
    if SEL.get('extra_eval_rows') is not None:
        _XR=SEL['extra_eval_rows'].cpu()
        _XN=int(_XR.shape[0])
        if _XR.ndim != 2 or _XR.shape[1] < 257:
            raise ValueError(f'extra_eval_rows must be [N,>=257], got {tuple(_XR.shape)}')
        cur['clsmap']=classify2(_XR).to(DEV)
        _old_tail=bool(SEL.get('qk_tail_on'))
        try:
            SEL['qk_tail_on']=False
            _xb=evalV(_XR,_XN,[],[]).float().cpu()
            SEL['qk_tail_on']=_old_tail
            _extra_input_variants=SEL.get('final_mlp_input_program_variants')
            if _extra_input_variants:
                _extra_primary=str(SEL.get('final_mlp_input_primary_variant'))
                SEL['extra_eval_variant_cevs']={}
                for _extra_name,_extra_programs in _extra_input_variants.items():
                    SEL['final_mlp_input_programs']=_extra_programs
                    SEL['extra_eval_variant_cevs'][str(_extra_name)]=evalV(
                        _XR,_XN,order2,ML).float().cpu()
                SEL['final_mlp_input_programs']=_extra_input_variants[_extra_primary]
                _xc=SEL['extra_eval_variant_cevs'][_extra_primary]
            else:
                _xc=evalV(_XR,_XN,order2,ML).float().cpu()
        finally:
            SEL['qk_tail_on']=_old_tail
            del cur['clsmap']
        _xd=(_xc-_xb).reshape(_XN,-1)
        SEL['extra_eval']={
            'name':str(SEL.get('extra_eval_name','unnamed')),
            'n_rows':_XN,
            'native_ce':float(_xb.mean()),
            'compiled_ce':float(_xc.mean()),
            'damage_mean':float(_xd.mean()),
            'damage_mean_abs_position':float(_xd.abs().mean()),
            'damage_by_row':[float(v) for v in _xd.mean(1)],
        }
        if SEL.get('extra_eval_variant_cevs'):
            SEL['extra_eval_variants']={}
            for _extra_name,_extra_cev in SEL['extra_eval_variant_cevs'].items():
                _extra_damage=(_extra_cev-_xb).reshape(_XN,-1)
                SEL['extra_eval_variants'][_extra_name]={
                    'name':str(SEL.get('extra_eval_name','unnamed')),
                    'n_rows':_XN,
                    'native_ce':float(_xb.mean()),
                    'compiled_ce':float(_extra_cev.mean()),
                    'damage_mean':float(_extra_damage.mean()),
                    'damage_mean_abs_position':float(_extra_damage.abs().mean()),
                    'damage_by_row':[float(v) for v in _extra_damage.mean(1)],
                }
        print(f"  extra eval {SEL['extra_eval']['name']}: native {_xb.mean():.6f} "
              f"compiled {_xc.mean():.6f} damage {_xd.mean():+.8f}",flush=True)
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
    SEL['L2CF']=(float(L1F),float(L2C),float(L2F))
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
    import sys as _sys
    _sys.path.insert(0,'/workspace/rspd')
    import census_lib as CN
    CN.use_state('census_state_diverse.pt')
    CROWS=CN.rows().cpu()
    CBASE=CN.base_ce().float().cpu()
    NFLAT=CN.nflat()
    ANCH=json.load(open(PT+'frontier_tail_traj_results.json'))
    SEL['mode']='norm'; SEL['K']=4608; SEL['K69']=4608; SEL['K69MAP']={}
    SEL['skipset']=tuple(range(10,18)); SEL['motif_off']=(); SEL['clsdmg']=True; SEL['ext_rows']=CROWS
    SEL['cp_swap']=4608; SEL['qk_r']=96; SEL['qk_rmap']={li:128 for li in range(2,10)}; SEL['qk_tail']=True; SEL['drop_tailE']=True
    print('ARM: rank-16 QK patterns at ALL replaced heads, blocks 2-17 (dicts retired)',flush=True)
    main()
    if 'L2CF' not in SEL: raise SystemExit('INSTRUMENT FAIL: L2CF capture missing')
    L1F,L2C,L2F=SEL['L2CF']
    cev=SEL['cev']
    torch.save(cev,PT+'cev_ct96.pt')
    print('cev saved: cev_ct96.pt',flush=True)
    if cev.shape[0]!=NFLAT: raise SystemExit(f'INSTRUMENT FAIL: cev {cev.shape[0]} != {NFLAT}')
    d=cev.cpu()-CBASE
    agg=float(d.mean())
    inc=L2F-L1F
    BATC=json.load(open(PT+'circuits/BATTERY.json'))['by_tag']
    rows=[]
    for t,v in BATC.items():
        try: lf=CN.leaf(t)
        except Exception: continue
        mm=torch.zeros(NFLAT,dtype=torch.bool); mm[lf['member']]=True
        if mm.sum()==0: continue
        md=float(d[mm].abs().mean())
        ref=v['mean_ablation']['top'][0]['abs_dce_members']
        rows.append({'tag':t,'ref':round(ref,3),'member_absdce':round(md,4),'valid':bool(md<0.5*ref)})
    nv=sum(1 for r in rows if r['valid'])
    clsflat=SEL['clsflat']
    from circuit_dictionary import CLS as _CLS
    PC={}
    for _k in range(10):
        _mk=clsflat==_k
        if int(_mk.sum())>0: PC[_CLS[_k]]=round(float(d[_mk].mean()),4)
    print('  per-class dCE: '+' '.join(f'{k}:{v:+.3f}' for k,v in PC.items()),flush=True)
    _link=(PC.get('ind',0.0)+PC.get('other',0.0))/2
    _cvals=[PC[c] for c in ('digit','bclose','sentend','comma','name','rep') if c in PC]
    _const=sum(_cvals)/max(len(_cvals),1)
    _sw=PC.get('subword',9.9)
    R132={r['tag']:r['member_absdce'] for r in json.load(open(PT+'frontier_rows_results.json'))['circuits']}
    tags=[r['tag'] for r in rows if r['tag'] in R132]
    u=torch.tensor([R132[t] for t in tags]).argsort().argsort().float()
    v=torch.tensor([{r['tag']:r['member_absdce'] for r in rows}[t] for t in tags]).argsort().argsort().float()
    u=u-u.mean(); v=v-v.mean()
    rho=float((u*v).sum()/((u.norm()*v.norm())+1e-9))
    print(f'  era Spearman vs rung-132: {rho:.4f} over {len(tags)}',flush=True)
    ratios=[r['member_absdce']/max(r['ref'],1e-9) for r in rows]
    import statistics as stt
    medrat=stt.median(ratios)
    taucurve={str(t9):sum(1 for x in ratios if x<0.5*t9/0.5) for t9 in (0.5,1.0,1.5,2.0,3.0)}
    taucurve={'0.5':sum(1 for x in ratios if x<0.5),'1.0':sum(1 for x in ratios if x<1.0),
              '1.5':sum(1 for x in ratios if x<1.5),'2.0':sum(1 for x in ratios if x<2.0),
              '3.0':sum(1 for x in ratios if x<3.0)}
    print(f'  median member/ref ratio {medrat:.3f}; tau-curve {taucurve}',flush=True)
    pa=abs(agg-0.0553)<=0.015
    pb=nv>=11
    pc=abs(L2F-0.0565)<=0.015
    if abs(agg-1.9474)<1e-3:
        raise SystemExit('INSTRUMENT FAIL: OV residual inert (census unchanged)')
    res={'L2F_fresh':round(L2F,4),'L1F':round(L1F,4),'increment':round(inc,4),'census_agg':round(agg,4),
         'n_valid':nv,'per_class':PC,'circuits':rows,'era_spearman_vs_132':round(rho,4),'median_member_over_ref':round(medrat,4),'tau_curve':taucurve,'link_mean':round(_link,4),'const_mean':round(_const,4),
         'anchor_rung127':{'L2F':ANCH['L2F_fresh'],'census':ANCH['census_agg'],'increment':ANCH['increment']},
         'convention':'CE added above the real model; LOWER IS BETTER',
         'pred_a_census_recovery':bool(pa),'pred_b_frontier_moves':bool(pb),'pred_c_sane':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(res,open(OUT,'w'),indent=1)
    print(f"L2 fresh {L2F:+.4f} (was +1.8765); census {agg:+.4f} (was +1.9474); increment {inc:+.4f}; valid {nv}/{len(rows)}")
    print(f"(a) |census {agg:+.4f} - 0.0553| <= 0.015: {'HELD' if pa else 'FAILED'}")
    print(f"(b) valid {nv} >= 11; margin {medrat:.3f}: {'HELD' if pb else 'FAILED'}")
    print(f"(c) |L2F {L2F:+.4f} - 0.0565| <= 0.015: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')
