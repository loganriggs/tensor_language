"""EARLY FOLD -- how far up is an MLP still a function of the token?
535 established that attention layer 0's input is EXACTLY the
token embedding, because block 0 forms a scalar multiple of E and
rms_norm is scale invariant. The exact writer coefficients (503)
say something almost as strong about the blocks above it: each
block rescales the running residual by lam0 and re-adds lam1 times
the embedding, and with lam0 = 0.0127 at block 1 and lam1 = 8.0,
mlp1's input is about 8.01 x E plus 0.0127 x (everything layer 0
wrote). That is a coefficient ratio of roughly 630 to 1 in favour
of the raw token.
If that is what it looks like, the early MLPs are nearly pure
functions of the current token whether or not anyone fits them a
lookup table -- and the program's older finding that mlp1 is 79%
replaceable by a per-token table would be a consequence of the
architecture rather than a discovery about the weights.
Measured directly: for each of mlp0 through mlp5, replace every
COMPONENT writer in that MLP's input by its mean over positions,
leaving only the embedding's contribution position-dependent, and
price the whole model. The surgery uses cl.writer_parts, which is
exact to 1e-7, and checks the reconstruction before scoring.
The reverse arm is the control that makes the claim falsifiable:
mean-fill the EMBEDDING's contribution instead and keep every
component writer live. If the embedding carries the computation,
that must hurt far more.
REGISTERED PREDICTIONS:
  (0) EXACTNESS: the writer parts reproduce each MLP's real input
      to 1e-4 relative, checked per layer before scoring;
  (a) THE FRONT IS TOKEN-DRIVEN: restricting mlp0 and mlp1 to the
      embedding each costs under 0.30 nats;
  (b) IT DECAYS UPWARD: the cost rises monotonically from mlp0 to
      mlp5, because the lam0 product accumulates context the
      further up you go;
  (c) THE COEFFICIENTS PREDICT IT: across the six layers, the
      log ratio of the wte coefficient to the summed magnitude of
      the component coefficients rank-correlates with the cost at
      Spearman rho <= -0.60 (a bigger token share means a smaller
      cost). These coefficients come from the weights alone;
  NULL / REVERSE ARM: mean-filling the embedding's contribution
      while keeping all component writers must cost MORE than the
      forward arm at mlp0 and mlp1. If removing the token is
      cheaper than removing the context, the reading is backwards
      and (a) means nothing."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'early_fold_results.json'
NFRESH=48; NLAYERS=6

def spearman(a,b):
    def rank(v):
        o=sorted(range(len(v)),key=lambda i:v[i]); r=[0.0]*len(v)
        for p,i in enumerate(o): r[i]=p
        return r
    ra,rb=rank(a),rank(b); n=len(a)
    ma=sum(ra)/n; mb=sum(rb)/n
    num=sum((x-ma)*(y-mb) for x,y in zip(ra,rb))
    da=sum((x-ma)**2 for x in ra)**0.5
    db=sum((y-mb)**2 for y in rb)**0.5
    return num/max(da*db,1e-9)

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    fresh=cl.fineweb_rows(NFRESH)
    coef={}
    for li in range(NLAYERS):
        c=cl.writer_coeffs(li,'m')
        wte=c['wte']; others=sum(abs(v) for k,v in c.items()
                                 if k!='wte')
        coef[li]={'wte':round(wte,4),
                  'components':round(others,4),
                  'log_ratio':round(float(torch.log(
                      torch.tensor(wte/max(others,1e-9)))),3)}
        print(f'mlp{li}: wte coefficient {wte:.4f} vs summed '
              f'component coefficients {others:.4f} '
              f'(log ratio {coef[li]["log_ratio"]})',flush=True)
    errs=[]

    def run(li=None,mode=None):
        """mode: 'token_only' (mean-fill component writers) |
        'context_only' (mean-fill the embedding's share)"""
        ce=torch.zeros(NFRESH,T)
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=bb.shape[0]
            outs={}; hs=[]
            if li is not None:
                for lj in range(li+1):
                    for kind,mod in (('a',m.transformer.h[lj].attn),
                                     ('m',m.transformer.h[lj].mlp)):
                        def mk(k9=f'{kind}{lj}'):
                            def h(mo,i_,o_):
                                y=o_[0] if isinstance(o_,tuple) else o_
                                outs[k9]=y.detach().float()
                            return h
                        hs.append(mod.register_forward_hook(mk()))
                E=F.rms_norm(m.transformer.wte(idx),(D,)).float()
                def ph(mo_,args,li=li,E=E):
                    X=args[0]
                    parts=cl.writer_parts(li,E,outs,'m')
                    tot=sum(parts.values())
                    errs.append(float((F.rms_norm(tot,(D,))
                        -X.float()).norm()
                        /X.float().norm().clamp_min(1e-9)))
                    if mode=='token_only':
                        t2=parts['wte']+sum(
                            p.mean(dim=(0,1),keepdim=True)
                            for k,p in parts.items() if k!='wte')
                    else:
                        p=parts['wte']
                        t2=tot-p+p.mean(dim=(0,1),keepdim=True)
                    return (F.rms_norm(t2,(D,)).to(X.dtype),) \
                           +tuple(args[1:])
                hs.append(m.transformer.h[li].mlp
                          .register_forward_pre_hook(ph))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            ce[i:i+B]=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                      reduction='none').view(B,T).cpu()
            for h in hs: h.remove()
        return float(ce.mean())

    base=run()
    print(f'\nbaseline CE {base:.4f}',flush=True)
    res={}
    for li in range(NLAYERS):
        tok=run(li,'token_only')-base
        ctx=run(li,'context_only')-base
        res[li]={'token_only':round(tok,4),
                 'context_only':round(ctx,4),**coef[li]}
        print(f"mlp{li}: token-only {tok:+.4f} | context-only "
              f"{ctx:+.4f}",flush=True)
        json.dump({str(k):v for k,v in res.items()},
                  open(OUT,'w'),indent=1)
    ex=max(errs) if errs else 1.0
    p0=ex<=1e-4
    print(f'\n(0) writer reconstruction {ex:.3e}: '
          f"{'HELD' if p0 else 'FAILED -- VOID'}")
    if not p0:
        json.dump({'pred_0':False,'exactness':ex},
                  open(OUT,'w'),indent=1); return
    va=(res[0]['token_only']<0.30 and res[1]['token_only']<0.30)
    costs=[res[i]['token_only'] for i in range(NLAYERS)]
    vb=all(costs[i]<=costs[i+1]+1e-6 for i in range(NLAYERS-1))
    rho=spearman([res[i]['log_ratio'] for i in range(NLAYERS)],costs)
    vc=rho<=-0.60
    nul=(res[0]['context_only']>res[0]['token_only']
         and res[1]['context_only']>res[1]['token_only'])
    print(f"(a) mlp0 {costs[0]:+.4f} and mlp1 {costs[1]:+.4f} both "
          f"< 0.30: {'HELD' if va else 'FAILED'}")
    print(f"(b) cost rises monotonically {['%+.3f'%c for c in costs]}"
          f": {'HELD' if vb else 'FAILED'}")
    print(f"(c) Spearman(log coefficient ratio, cost) = {rho:.3f} "
          f"<= -0.60: {'HELD' if vc else 'FAILED'}")
    print(f"REVERSE ARM (removing the token costs more than "
          f"removing context, at mlp0 and mlp1): "
          f"{'ok' if nul else 'VIOLATED'}")
    out={'baseline_ce':round(base,4),
         'layers':{str(k):v for k,v in res.items()},
         'exactness':ex,'spearman':round(rho,3),
         'pred_0':True,'pred_a':bool(va),'pred_b':bool(vb),
         'pred_c':bool(vc),'null_ok':bool(nul),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
