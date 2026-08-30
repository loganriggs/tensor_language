"""PROBE GATE v8 -- is it the TEXT that is hard, or the ASSEMBLY that is bad there?

S2083 found the oracle gating gain is predicted by a window's base CE at
r = -0.8295, and S2084 -- computed free from S2083's own artifact -- found the
RANDOM gate tracks it almost as well (-0.7307), so gate-ability is a property of
the window rather than of the labeler.

S2084 then named the confound it could not resolve: `base_ce` is the REAL
model's cross-entropy on the window, i.e. how hard the TEXT is. The competing
explanation is the ASSEMBLY's excess over real there -- how badly the stand-ins
do on that window -- which probe_gate7 computed internally as the base-config
offset and threw away. Gating only makes sense as "intervene where the stand-ins
hurt", so the assembly's own error is the more mechanistic candidate, and the two
are confounded in every number in S2083 and S2084.

This is probe_gate7 with one extra field recorded per window and the two
correlations compared. Same eight windows, same seeds, so the oracle gains must
reproduce exactly.

REGISTERED PREDICTIONS:
  (a) THE ASSEMBLY, NOT THE TEXT: |r(assembly excess, oracle gain)| exceeds
      |r(base CE, oracle gain)| = 0.8295. Registered in the mechanistic
      direction -- gating substitutes stand-ins, so where the stand-ins are
      already costly is where changing them should matter. If FALSE, text
      difficulty is the better predictor and the effect is about the corpus
      rather than about the assembly, which would make the deployable rule in
      S2084 a statement about text and not about this compilation.
  (b) THEY ARE GENUINELY CONFOUNDED, as S2084 asserted without measuring:
      |r(base CE, assembly excess)| >= 0.7. If FALSE they are separable and
      S2084's "confounded in every number above" was too strong.
  (c) CONTROL: the per-window oracle gains reproduce probe_gate7's to within
      0.005. Same windows, same seeds, deterministic given both -- if this
      fails, the window construction is not reproducible across runs and every
      cross-run comparison in S2081-S2084 is void.

Writes probe_gate8_results.json.
"""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import os

# PLAN PRE-FLIGHT (LESSON 109): bilin18_joint_removal loads the model AT IMPORT, so enqueue's
# BQLIB_DRYRUN gate must be answered before that line or the gate runs the whole experiment.
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['probe_gate7_results.json','experiments/gating/probe_gate.py']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    _v3=json.load(open(_bq+'probe_gate7_results.json'))
    if 'windows' not in _v3 or len(_v3['windows'])!=8:
        print('DRYRUN FAIL: S2083 eight-window results missing'); raise SystemExit(1)
    print(f"DRYRUN OK: S2083 present ({len(_v3['windows'])} windows, "
          f"r(baseCE,oracle) {_v3['pearson_oracle_vs_base_ce']}); "
          f"adding assembly-excess per window")
    raise SystemExit(0)

import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from circuit_dictionary import classify, COMPS as TAILC, CLS
D=1152; V=50257
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'probe_gate8_results.json'
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
                    new=(((x@Lk.T)*(x@Rk.T))@Dk.T+db)
                    gt=cur.get('gate')
                    if gt is not None:
                        gv=gt.view(-1)[:,None]
                        new=torch.where(gv,o_.float().reshape(-1,D),
                                        new.reshape(-1,D)).view(o_.shape)
                    return new.to(o_.dtype)
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
                    gt=cur.get('gate')
                    if gt is not None:
                        gv=gt.view(-1)[:,None]
                        new=torch.where(gv,y.float().reshape(-1,D),new)
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
    # Three fresh windows, each 120 rows, from documents no earlier window used.
    # FR is built exactly as in S2079-S2081 so their numbers stay comparable and
    # the blk2 instrument control keeps its meaning across four runs.
    def take_window(start_di, banned):
        """collect 120 fresh rows from documents not in `banned`; return rows and
        the documents used, so the next window can exclude them."""
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
    NWIN=8
    WIN=[]; banned=set(); nxt=3000
    for wi in range(NWIN):
        rws,used,last = take_window(nxt,banned)
        WIN.append({'tok':torch.tensor(rws,dtype=torch.long),
                    'docs':len(used),'keys':{tuple(r[:32]) for r in rws}})
        banned |= used; nxt = last+1
    for i in range(NWIN):
        for j in range(i+1,NWIN):
            assert not (WIN[i]['keys'] & WIN[j]['keys']), f'windows {i},{j} share a row'
        del WIN[i]['keys']
    FR=WIN[0]['tok']                    # FR is window 0, built exactly as before
    print(f'{NWIN} windows x 120 rows, document-disjoint; docs per window: '
          f'{[w["docs"] for w in WIN]}',flush=True)
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
    # ---- damage-mode machinery: probes on fit window + fresh rows ----
    def ce_vec_rows(TOK,N,hooks):
        ces=[]
        for i in range(0,N,4):
            bb=TOK[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h:
                x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        for h in hooks: h.remove()
        return torch.cat(ces)
    rowsA_tok=FW[CA:CB]; nA=CB-CA
    rowsA=rowsA_tok
    baseA=ce_vec_rows(rowsA,nA,[])
    baseFr=ce_vec_rows(FR,120,[])
    for w in WIN: w['base']=ce_vec_rows(w['tok'],120,[])
    sums2={}; hs2=[]
    for li in range(18):
        for kind,mod in (('a',m.transformer.h[li].attn),
                         ('m',m.transformer.h[li].mlp)):
            key=f'{kind}{li}'; sums2[key]=torch.zeros(D,device=DEV)
            def mk2(key=key):
                def h(mo,i_,o_):
                    y=o_[0] if isinstance(o_,tuple) else o_
                    sums2[key]+=y.detach().float().reshape(-1,D).sum(0)
                return h
            hs2.append(mod.register_forward_hook(mk2()))
    for i in range(0,nA,4):
        bb=rowsA[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
    for h in hs2: h.remove()
    mus2={k:v/(nA*256) for k,v in sums2.items()}
    MODS2={f'a{li}':m.transformer.h[li].attn for li in range(18)}
    MODS2.update({f'm{li}':m.transformer.h[li].mlp for li in range(18)})
    def mp(key):
        mu=mus2[key]; mod=MODS2[key]
        if key[0]=='a':
            def fh(mo,i_,o_,mu=mu):
                y,v1=o_
                return (mu.expand_as(y).to(y.dtype),v1)
        else:
            def fh(mo,i_,o_,mu=mu):
                return mu.expand_as(o_).to(o_.dtype)
        return [mod.register_forward_hook(fh)]
    P36=[f'{k}{li}' for li in range(18) for k in ('a','m')]
    colsA=[]; colsW=[[] for _ in WIN]
    for j,key in enumerate(P36):
        colsA.append((ce_vec_rows(rowsA,nA,mp(key))-baseA).cpu())
        for wi,w in enumerate(WIN):
            colsW[wi].append((ce_vec_rows(w['tok'],120,mp(key))-w['base']).cpu())
        if j%12==0: print(f'probe {j}/36',flush=True)
    MA=torch.stack(colsA,1)
    for wi,w in enumerate(WIN): w['M']=torch.stack(colsW[wi],1)
    muA=MA.mean(0,keepdim=True); sdA=MA.std(0,keepdim=True)        .clamp_min(1e-6)
    MA=torch.clamp((MA-muA)/sdA,-3,3)
    for w in WIN: w['M']=torch.clamp((w['M']-muA)/sdA,-3,3)   # fit-window norm, not refit
    def sv(X):
        try: return torch.linalg.svd(X,full_matrices=False)
        except Exception:
            U,S9,Vh9=torch.linalg.svd(X.double(),full_matrices=False)
            return U.float(),S9.float(),Vh9.float()
    U9,S9,Vh9=sv(MA)
    scA=U9[:,:10]*S9[:10]
    thr=scA.abs().quantile(0.97,dim=0)
    for wi,w in enumerate(WIN):
        sc=w['M']@Vh9[:10].T
        g=(sc.abs()>=thr[None,:]).any(1)
        w['frac']=float(g.float().mean())
        w['gate']=g.view(120,256)
        gg=torch.Generator().manual_seed(100+wi)
        w['rmask']=(torch.rand(120,256,generator=gg)<w['frac'])
    frac=WIN[0]['frac']; gateF=WIN[0]['gate']; rmask=WIN[0]['rmask']
    gateA=(scA.abs()>=thr[None,:]).any(1).to(DEV)
    print(f'gated fractions: {[round(w["frac"],4) for w in WIN]}',flush=True)
    # ---- deploy-legal probes at SEVERAL read points (rung 4) ----
    # v1 read only after block 2. `depths` is a list so a single site and a
    # concatenated pair go through identical code -- no separate two-site path
    # to disagree with the single-site one.
    ARMS=[('blk2',[2]),('blk2+9',[2,9])]
    def stream_rows(TOK,N,depths):
        dmax=max(depths); outs=[]
        for i in range(0,N,4):
            bb=TOK[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous()
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            grab={}
            for lj,blk in enumerate(m.transformer.h):
                x,v1=blk(x,v1,x0)
                if lj in depths: grab[lj]=x.float().reshape(-1,D)
                if lj==dmax: break
            outs.append(torch.cat([grab[d] for d in depths],1))
        return torch.cat(outs)
    yA=gateA.float()*2-1
    ARMR={}
    for nm,depths in ARMS:
        XA=stream_rows(rowsA_tok,nA,depths)
        dim=XA.shape[1]
        lamp=1e-2*len(XA)
        Wp=torch.linalg.solve(XA.T@XA+lamp*torch.eye(dim,device=DEV),XA.T@yA)
        del XA
        ARMR[nm]={'depths':depths,'dim':dim}
        for wi,w in enumerate(WIN):
            XF=stream_rows(w['tok'],120,depths)
            sc=(XF@Wp).cpu(); del XF
            w.setdefault('agate',{})[nm]=(sc>=sc.quantile(1-w['frac'])).view(120,256)
        print(f'  arm {nm:8s} dim {dim}',flush=True)
    def evalG(active,mlayers,gate,TOK=None,CM=None):
        TOK=FR if TOK is None else TOK
        hs=install(active)+motif_hooks(mlayers)
        if CM is not None: cur['clsmap']=CM
        ces=[]
        for i in range(0,120,4):
            bb=TOK[i:i+4,:257].to(DEV)
            cur['idx']=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            cur['mode']='oracle'
            cur['lab']=cur['clsmap'][i:i+4].reshape(-1)
            cur['gate']=(gate[i:i+4].reshape(-1).to(DEV)
                         if gate is not None else None)
            x=F.rms_norm(m.transformer.wte(cur['idx']),(D,))
            x0=x; v1=None
            for blk in m.transformer.h:
                x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        for h in hs: h.remove()
        cur['gate']=None
        return float(torch.cat(ces).mean())
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
                elif st in (')',']') and any(bch in enc4.decode(
                    toks[max(0,pos-60):pos+1]) for bch in ('(','[')): k=1
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
    ML=list(range(2,10))
    for wi,w in enumerate(WIN):
        w['cm']=classify2(w['tok']).to(DEV)
    ROWS=[]
    for wi,w in enumerate(WIN):
        b=evalG(order2,ML,None,w['tok'],w['cm'])-float(w['base'].mean())
        o=evalG(order2,ML,w['gate'],w['tok'],w['cm'])-float(w['base'].mean())
        r=evalG(order2,ML,w['rmask'],w['tok'],w['cm'])-float(w['base'].mean())
        rec={'window':wi,'docs':w['docs'],'gated_frac':round(w['frac'],4),
             'base_ce':round(float(w['base'].mean()),4),
             'assembly_excess':round(b,4),        # base config CE minus the real model's
             'oracle_gain':round(b-o,4),'random_gain':round(b-r,4)}
        for nm,_dp in ARMS:
            p=evalG(order2,ML,w['agate'][nm],w['tok'],w['cm'])-float(w['base'].mean())
            rec[f'gain_{nm}']=round(b-p,4)
        ROWS.append(rec)
        print(f"  win{wi} docs {rec['docs']:3d} baseCE {rec['base_ce']:.4f} "
              f"frac {rec['gated_frac']:.3f} | oracle {rec['oracle_gain']:+.4f} "
              f"random {rec['random_gain']:+.4f} blk2 {rec['gain_blk2']:+.4f} "
              f"blk2+9 {rec['gain_blk2+9']:+.4f}",flush=True)
        del w['cm'], w['gate'], w['rmask'], w['agate'], w['M']
    del cur['clsmap']

    def pearson(x,y):
        n=len(x); mx=sum(x)/n; my=sum(y)/n
        cov=sum((a-mx)*(b2-my) for a,b2 in zip(x,y))
        sx=sum((a-mx)**2 for a in x)**0.5; sy=sum((b2-my)**2 for b2 in y)**0.5
        return cov/(sx*sy) if sx>0 and sy>0 else 0.0
    og=[r['oracle_gain'] for r in ROWS]
    dc=[float(r['docs']) for r in ROWS]
    bc=[r['base_ce'] for r in ROWS]
    ex=[r['assembly_excess'] for r in ROWS]
    rg=[r['random_gain'] for r in ROWS]
    pg=[r['gain_blk2+9'] for r in ROWS]
    r_base=pearson(og,bc); r_ex=pearson(og,ex); r_conf=pearson(bc,ex)
    S2083=[0.0776,0.0577,-0.1305,0.0440,0.0829,0.0527,0.0188,0.0317]
    dev=max(abs(og[i]-S2083[i]) for i in range(min(len(og),len(S2083))))
    pa=abs(r_ex)>abs(r_base)
    pb=abs(r_conf)>=0.7
    pc=dev<=0.005
    out={'windows':ROWS,'n_windows':len(ROWS),
         'pearson_oracle_vs_base_ce':round(r_base,4),
         'pearson_oracle_vs_assembly_excess':round(r_ex,4),
         'pearson_base_ce_vs_assembly_excess':round(r_conf,4),
         'pearson_random_vs_assembly_excess':round(pearson(rg,ex),4),
         'pearson_probe_vs_assembly_excess':round(pearson(pg,ex),4),
         'max_oracle_deviation_from_S2083':round(dev,5),
         'pred_a_assembly_beats_text':bool(pa),
         'pred_b_they_are_confounded':bool(pb),
         'pred_c_reproduces_S2083':bool(pc)}
    print(f"\nr(oracle, base CE)          = {r_base:+.4f}")
    print(f"r(oracle, assembly excess)  = {r_ex:+.4f}")
    print(f"r(base CE, assembly excess) = {r_conf:+.4f}")
    print(f"r(random, assembly excess)  = {pearson(rg,ex):+.4f}")
    print(f"(a) |r_excess| {abs(r_ex):.4f} > |r_baseCE| {abs(r_base):.4f}: "
          f"{'HELD' if pa else 'FAILED'}")
    print(f"(b) |r(baseCE,excess)| {abs(r_conf):.4f} >= 0.7: "
          f"{'HELD' if pb else 'FAILED'}")
    print(f"(c) oracle gains reproduce S2083 within 0.005 (max dev "
          f"{dev:.5f}): {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
