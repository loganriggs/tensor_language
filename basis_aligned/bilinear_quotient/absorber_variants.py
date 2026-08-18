"""ABSORBER VARIANTS -- can weight structure and connection-SVD cut the
~0.12-nat per-component cost of the v5 table+absorber rungs?

Same v5 pipeline (19/36 components, sequential), four arms differing ONLY
in how each MLP rung's absorber is built:
  LIN      -- v5 baseline: rank-32 PCA residual basis + linear ridge read.
  QUAD     -- absorber features are [x] plus the 136 pairwise products
              z_i z_j of z = V16^T x, where V16 = top-16 right-singular
              directions of the layer's own [L;R] weights. Rationale: the
              MLP is exactly quadratic, so the table residual is quadratic
              in the input; the weights say WHICH quadratic.
  READ     -- the residual basis P is chosen inside the downstream READ
              subspace (top-64 eigenvectors of the summed, trace-normalized
              Grams of every later layer's input weights + lm_head).
              Rationale: error outside what downstream reads is invisible;
              spend the rank where the wiring looks (the connection-SVD /
              interface idea).
  READRAND -- control for READ: a random 64-dim subspace instead.
REGISTERED PREDICTIONS:
  (a) LIN reproduces v5 within +-0.05 of +2.095;
  (b) QUAD total <= +1.95 (weight-derived quadratic features cut >=0.15);
  (c) READ total <= +2.00 AND READRAND >= READ + 0.05 (the read subspace,
      not mere 64-dim restriction, does the work);
  (d) per-rung absorber R^2 reported per arm."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from circuit_dictionary import classify, COMPS as TAILC
D=1152; V=50257
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'absorber_variants_results.json'
CA,CB=300,512; R0,R1=120,300
IU,IL=torch.triu_indices(16,16)

def quadfeat(X,V16):
    Z=X@V16
    return torch.cat([X,Z[:,IU]*Z[:,IL]],1)

@torch.no_grad()
def main():
    t0=time.time()
    clsA=classify(CA,CB).to(DEV); clsC=classify(R0,R1).to(DEV)
    flatA=clsA.reshape(-1)
    g=torch.Generator(device=DEV).manual_seed(0)
    reads={}; v16={}
    lmG=m.lm_head.weight.detach().float()
    lmG=lmG.T@lmG; lmG=lmG/lmG.trace()
    for li in range(0,10):
        G=lmG.clone()
        for lj in range(li+1,18):
            blk=m.transformer.h[lj]
            for W in (blk.mlp.Left.weight,blk.mlp.Right.weight,
                      blk.attn.c_q.weight,blk.attn.c_k.weight,
                      blk.attn.c_q2.weight,blk.attn.c_k2.weight,
                      blk.attn.c_v.weight):
                Wf=W.detach().float(); Gi=Wf.T@Wf
                G=G+Gi/Gi.trace()
        ev,evec=torch.linalg.eigh(G)
        reads[li]=evec[:,-64:].contiguous()
        mlp=m.transformer.h[li].mlp
        _,_,Vh=torch.linalg.svd(torch.cat([mlp.Left.weight.detach().float(),
                                           mlp.Right.weight.detach().float()]),
                                full_matrices=False)
        v16[li]=Vh[:16].T.contiguous()
    randU={li:orth(torch.randn(D,64,device=DEV,generator=g))
           for li in range(0,10)}

    def build(arm):
        S={}; cur={}; r2s={}
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
        def install(active):
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
                    _,li,tb,A,P,fmode=S[nm]
                    def h(mod,i_,o_,tb=tb,A=A,P=P,fmode=fmode,li=li):
                        x=i_[0].float().reshape(-1,D)
                        ft=quadfeat(x,v16[li]) if fmode=='quad' else x
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
            if arm in ('read','readrand'):
                U=reads[li] if arm=='read' else randU[li]
                Rp=(Rr@U)@U.T
                _,_,Vh=torch.linalg.svd(Rp[:30000], full_matrices=False)
            else:
                _,_,Vh=torch.linalg.svd(Rr[:30000], full_matrices=False)
            P=orth(Vh[:32].T)
            ft=quadfeat(X,v16[li]) if arm=='quad' else X
            lam=1e-2*len(X)
            A=torch.linalg.solve(ft.T@ft+lam*torch.eye(ft.shape[1],
                                                       device=DEV),
                                 ft.T@(Rr@P))
            r2=1-float(((Rr@P-ft@A)**2).sum())/float(((Rr@P)**2).sum())
            r2s[li]=round(r2,3)
            fmode='quad' if arm=='quad' else 'lin'
            return ('tableres',li,tb,A,P,fmode)
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
        for li,nm in ((2,'m2'),(3,'m3'),(4,'m4'),(5,'m5'),(6,'m6'),
                      (7,'m7'),(8,'m8'),(9,'m9')):
            Y,X,ids=runA(order,m.transformer.h[li].mlp)
            S[nm]=fit_tableres(Y,X,ids,li); order.append(nm)
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
        X10=torch.cat(capsI[TAILC[0]])
        Yoh=torch.zeros(len(flatA),10,device=DEV)
        Yoh[torch.arange(len(flatA)),flatA]=1.0
        lam=1e-2*len(X10)
        Wp=torch.linalg.solve(X10.T@X10+lam*torch.eye(D,device=DEV),
                              X10.T@Yoh)
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
                LIN[li][k]=torch.linalg.solve(
                    Xk.T@Xk+l2*torch.eye(D,device=DEV),Xk.T@Ck)
            capsT[li]=None; capsI[li]=None
        S['tail']=('tail',Wp,DICT,LIN); order.append('tail')
        def evalC(active):
            hs=install(active)
            ces=[]
            for i in range(R0,R1,4):
                bb=FW[i:i+4,:257].to(DEV)
                cur['idx']=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
                x=F.rms_norm(m.transformer.wte(cur['idx']),(D,))
                x0=x; v1=None
                for blk in m.transformer.h:
                    x,v1=blk(x,v1,x0)
                lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
                ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                           reduction='none'))
            for h in hs: h.remove()
            return float(torch.cat(ces).mean())
        base=evalC([])
        tot=evalC(order)-base
        print(f'{arm:9s} total {tot:+.4f} | R2 {r2s}',flush=True)
        return tot,r2s
    out={}
    for arm in ('lin','quad','read','readrand'):
        tot,r2s=build(arm)
        out[arm]={'total':round(tot,4),'r2':r2s}
    pa=abs(out['lin']['total']-2.095)<=0.05
    pb=out['quad']['total']<=1.95
    pc=(out['read']['total']<=2.00
        and out['readrand']['total']>=out['read']['total']+0.05)
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['pred_c']=bool(pc)
    print(f"\n(a) LIN reproduces v5 +-0.05: {'HELD' if pa else 'FAILED'}")
    print(f"(b) QUAD <= +1.95: {'HELD' if pb else 'FAILED'}")
    print(f"(c) READ <= +2.00 and READRAND worse by >=0.05: "
          f"{'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
