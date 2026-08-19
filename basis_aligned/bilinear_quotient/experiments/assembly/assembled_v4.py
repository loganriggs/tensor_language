"""ASSEMBLED v4 -- SEQUENTIAL + RESIDUAL ABSORBERS: v3 plus rank-32
input-linear residual correctors on the m0/m2/m3 tables, each fit
sequentially with all upstream stand-ins (and their absorbers) active.
Per section 258, absorbers should convert the tables from pure firewalls
into firewall+absorb: lossy clamp plus a learned term that reads the
(substituted) input and cancels the table error the clamp would transmit.
REGISTERED: (a) v4 <= +1.20 (beats v1 +1.85); (b) v4 < v3 +1.97; (c) each
absorber in-fit residual R^2 reported (floor rule).

(v3 docstring:)
ASSEMBLED v3 -- SEQUENTIAL: every fitted stand-in is fit with its
upstream replacements ACTIVE (the layer-track sequential-refit lesson,
section 257), in computation-graph order:
  a0 (weights-exact) -> m0 table -> a1 value-table -> m1 linear ->
  m2 table -> m3 table -> tail probe+dictionary.
REGISTERED: (a) v3 assembled <= +0.90 nats; (b) v3 < v1 (+1.85) and
< v2 (+2.69); (c) m1's in-context marginal cost (assembled minus
assembled-without-m1) <= 2x its clean solo (+0.24)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from circuit_dictionary import classify, COMPS as TAILC
D=1152; V=50257
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'assembled_v4_results.json'
CA,CB=300,512; R0,R1=120,300

@torch.no_grad()
def main():
    t0=time.time()
    S={}                       # fitted stand-ins by name
    cur={}
    # a0 exact table (vocab pass)
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
    clsA=classify(CA,CB).to(DEV); clsC=classify(R0,R1).to(DEV)
    def install(names):
        hs=[]
        for nm in names:
            kind=S[nm][0]
            if kind=='cv':
                _,li,tb=S[nm]
                def h(mod,i_,o_,tb=tb):
                    return tb[cur['idx']].to(o_.dtype)
                hs.append(m.transformer.h[li].attn.c_v
                          .register_forward_hook(h))
            elif kind=='table':
                _,li,tb=S[nm]
                def h(mod,i_,o_,tb=tb):
                    return tb[cur['idx']].to(o_.dtype)
                hs.append(m.transformer.h[li].mlp
                          .register_forward_hook(h))
            elif kind=='tableres':
                _,li,tb,A,P=S[nm]
                def h(mod,i_,o_,tb=tb,A=A,P=P):
                    x=i_[0].float().reshape(-1,D)
                    new=tb[cur['idx']].float().reshape(-1,D)+(x@A)@P.T
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
        P=orth(Vh[:32].T)
        lam=1e-2*len(X)
        A=torch.linalg.solve(X.T@X+lam*torch.eye(D,device=DEV),
                             X.T@(Rr@P))
        r2=1-float(((Rr@P-X@A)**2).sum())/float(((Rr@P)**2).sum())
        print(f'absorber m{li}: residual-subspace R^2 {r2:.2f}',flush=True)
        return ('tableres',li,tb,A,P)
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
    for li,nm in ((2,'m2'),(3,'m3')):
        Y,X,ids=runA(order,m.transformer.h[li].mlp)
        S[nm]=fit_tableres(Y,X,ids,li); order.append(nm)
    # tail: capture every tail comp under full front
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
        m(cur['idx'], bb[:,1:].contiguous())
    for h in hs: h.remove()
    flatA=clsA.reshape(-1)
    X10=torch.cat(capsI[TAILC[0]])
    Yoh=torch.zeros(len(flatA),10,device=DEV)
    Yoh[torch.arange(len(flatA)),flatA]=1.0
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
    clsC2=clsC.view(R1-R0,256)
    def evalC(active):
        hs=install(active)
        ces=[]
        for i in range(R0,R1,4):
            bb=FW[i:i+4,:257].to(DEV)
            cur['idx']=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            x=F.rms_norm(m.transformer.wte(cur['idx']),(D,)); x0=x; v1=None
            for blk in m.transformer.h:
                x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        for h in hs: h.remove()
        return float(torch.cat(ces).mean())
    base=evalC([])
    full=evalC(order)-base
    nom1=evalC([n for n in order if n!='m1'])-base
    m1ctx=full-nom1
    pa=full<=1.20; pb=full<1.97; pc=m1ctx<=0.24
    out={'base':round(base,4),'assembled_v4':round(full,4),
         'without_m1':round(nom1,4),'m1_in_context':round(m1ctx,4),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f'base {base:.3f} | ASSEMBLED v4 {full:+.4f} | without m1 '
          f'{nom1:+.4f} | m1 in-context {m1ctx:+.4f}')
    print(f"(a) <=1.20: {'HELD' if pa else 'FAILED'}")
    print(f"(b) beats v3: {'HELD' if pb else 'FAILED'}")
    print(f"(c) m1 in-context <=0.24: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
