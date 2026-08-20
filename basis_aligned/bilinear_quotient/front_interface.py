"""FRONT INTERFACE -- how many numbers does layer 0 send upward?
537 showed that from mlp1 onward the model does not need the
re-injected token embedding at all -- mean-filling it costs 0.0004
to 0.036 nats -- because layer 0 has already written the token
information into the stream. 535 and 536 showed what layer 0 is:
attention that compares tokens along 2 directions per head, and an
MLP that writes into about 64 of 1152 directions.
The question those three leave is the interface one, and it is the
number a benchmark most wants: the rest of the network reads layer
0 only through what layer 0 writes, so how many DIRECTIONS of that
write are load-bearing? If it is small, the front of the model can
be replaced by an explicit function of the token pair emitting r
numbers, and everything above it is unchanged by construction.
Method: capture the combined layer-0 write (attn0 + mlp0) over the
corpus, take the SVD of that write, and keep only its top r
directions -- replacing the discarded component by its mean over
positions so the average level is preserved and only the
position-dependent part is removed. Then price the whole model.
Three interfaces are measured: the combined write, attn0's alone,
and mlp0's alone, because 536 found the MLP writes narrowly and
the attention layer may not.
REGISTERED PREDICTIONS:
  (0) FULL RANK IS FREE: keeping all 1152 directions costs under
      1e-3 nats. Failure VOIDS the run;
  (a) NARROW INTERFACE: keeping 32 directions of the combined
      layer-0 write costs under 0.10 nats;
  (b) BEATS RANDOM: at r = 32, the SVD interface costs less than a
      third of a random 32-dimensional interface, three draws;
  (c) THE SPLIT: report the smallest r under 0.10 for attn0's
      write alone and for mlp0's write alone. 536 predicts mlp0's
      number should be near 64; no bar on attn0's, it is the
      unknown;
  NULL: keeping ZERO directions -- mean-filling the entire layer-0
      write -- must cost far more than any r tested, at least
      1.0 nats. If deleting the whole front is cheap, the
      interface question is not worth asking and the run says so."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'front_interface_results.json'
NFRESH=48
RANKS=[0,4,8,16,32,64,128,256,1152]

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    fresh=cl.fineweb_rows(NFRESH)
    at0=m.transformer.h[0].attn; mlp0=m.transformer.h[0].mlp
    # collect the layer-0 writes
    caps={'a':[],'m':[]}
    ha=at0.register_forward_hook(
        lambda mo,i_,o_: caps['a'].append(
            (o_[0] if isinstance(o_,tuple) else o_)
            .detach().float().reshape(-1,D).cpu()))
    hm=mlp0.register_forward_hook(
        lambda mo,i_,o_: caps['m'].append(
            o_.detach().float().reshape(-1,D).cpu()))
    for i in range(0,NFRESH,4):
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous()
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
    ha.remove(); hm.remove()
    A=torch.cat(caps['a']).to(DEV); M=torch.cat(caps['m']).to(DEV)
    S={'combined':A+M,'attn0':A,'mlp0':M}
    basis={}; means={}
    for k,Y in S.items():
        mu=Y.mean(0); means[k]=mu
        _,_,Vh=torch.linalg.svd(Y-mu,full_matrices=False)
        basis[k]=Vh          # (D, D) rows are directions
        print(f'{k}: write norm {float(Y.norm(dim=-1).mean()):.3f} '
              f'| mean-share '
              f'{float(mu.norm()/Y.norm(dim=-1).mean()):.3f}',
              flush=True)

    def price(which=None,r=None,random=False,seed=0):
        ce=torch.zeros(NFRESH,T)
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=bb.shape[0]; hs=[]
            if which is not None:
                if random:
                    g=torch.Generator(device=DEV).manual_seed(seed)
                    Q,_=torch.linalg.qr(torch.randn(D,max(r,1),
                                        generator=g,device=DEV))
                    P=Q@Q.T if r>0 else torch.zeros(D,D,device=DEV)
                else:
                    V=basis[which][:r] if r>0 else None
                    P=(V.T@V) if r>0 else torch.zeros(D,D,device=DEV)
                mu=means[which]
                def keepproj(y):
                    yf=y.float()
                    dev=yf-mu
                    return (mu+dev@P).to(y.dtype)
                if which in ('combined','attn0'):
                    def fa(mo,i_,o_):
                        y,v1=o_
                        return (keepproj(y),v1) \
                            if which=='attn0' else (y,v1)
                    if which=='attn0':
                        hs.append(at0.register_forward_hook(fa))
                if which in ('combined','mlp0'):
                    if which=='mlp0':
                        hs.append(mlp0.register_forward_hook(
                            lambda mo,i_,o_: keepproj(o_)))
                if which=='combined':
                    # project the SUM: apply to mlp0's output the
                    # projection of (a0+m0) by correcting with a0
                    store={}
                    hs.append(at0.register_forward_hook(
                        lambda mo,i_,o_: (store.__setitem__(
                            'a',(o_[0] if isinstance(o_,tuple)
                                 else o_).detach().float()),o_)[1]))
                    def fm(mo,i_,o_):
                        a=store['a']; tot=a+o_.float()
                        newtot=keepproj(tot)
                        return (newtot.float()-a).to(o_.dtype)
                    hs.append(mlp0.register_forward_hook(fm))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            ce[i:i+B]=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                      reduction='none').view(B,T).cpu()
            for h in hs: h.remove()
        return float(ce.mean())

    base=price()
    print(f'\nbaseline CE {base:.4f}',flush=True)
    res={}
    for which in ('combined','attn0','mlp0'):
        row={}
        for r in RANKS:
            c=price(which,r)-base
            rnd=[]
            if r==32:
                for s in (1,2,3):
                    rnd.append(round(price(which,r,True,s)-base,4))
            row[r]={'cost':round(c,4),'random':rnd}
            print(f'{which:>9} r={r:>5}: {c:+.4f}'
                  +(f' | random {rnd}' if rnd else ''),flush=True)
        res[which]=row
        json.dump({k:{str(a):b for a,b in v.items()}
                   for k,v in res.items()},open(OUT,'w'),indent=1)
    p0=abs(res['combined'][1152]['cost'])<1e-3
    print(f"\n(0) full rank free "
          f"({res['combined'][1152]['cost']:+.5f}): "
          f"{'HELD' if p0 else 'FAILED -- VOID'}")
    if not p0:
        json.dump({'pred_0':False},open(OUT,'w'),indent=1); return
    va,_=cl.score_bar('a',0.10-res['combined'][32]['cost'],1e-9)
    r32=res['combined'][32]
    best=min(r32['random']) if r32['random'] else 1.0
    vb,_=cl.score_bar('b',best-3*max(r32['cost'],0),1e-9)
    firsts={}
    for which in ('combined','attn0','mlp0'):
        firsts[which]=next((r for r in RANKS
                            if res[which][r]['cost']<0.10),None)
    print(f"(c) smallest rank under 0.10 nats: {firsts}")
    zero=res['combined'][0]['cost']
    nul=zero>=1.0
    print(f"NULL (deleting the whole layer-0 write costs "
          f"{zero:+.4f} >= 1.0): {'ok' if nul else 'VIOLATED'}")
    out={'baseline_ce':round(base,4),
         'results':{k:{str(a):b for a,b in v.items()}
                    for k,v in res.items()},
         'first_under_0.10':firsts,'delete_all_cost':round(zero,4),
         'pred_0':True,'pred_a':va=='HELD','pred_b':vb=='HELD',
         'null_ok':bool(nul),'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
