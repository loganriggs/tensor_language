"""MLP0 LOWRANK -- the compression operator that suits the layer.
534 showed subset selection is the wrong tool for mlp0. Keeping
the loudest 16 hidden units and mean-filling the rest costs 1.58
nats while 16 RANDOM units cost 0.81; top-K only overtakes random
at 256 of 4608. The layer's output is a signed sum whose loud
terms largely annihilate, so removing their partners leaves an
unopposed remainder that is worse than removing anything in
particular. That is a fact about the layer, and it says the
compression should act on the MAP rather than on a subset of its
terms.
Two low-rank claims, both interpretable statements in their own
right and both exact truncations rather than fits:
  INPUT RANK. Replace L and R by their rank-r truncations in the
    INPUT dimension -- the same r directions for both, taken from
    the SVD of the stacked [L;R] -- so the layer reads x only
    through r of its 1152 directions. The claim "mlp0 reads r
    features of the embedding" is then literally true.
  OUTPUT RANK. Replace Down by its rank-r truncation, so the layer
    writes into only r directions of the residual.
Because layer 0's input is exactly the token embedding (verified
at 1e-7 elsewhere), an input-rank-r mlp0 is exactly "r numbers per
token, combined quadratically" -- the interpretable object the
benchmark wants, with r as its size.
REGISTERED PREDICTIONS:
  (0) FULL RANK IS FREE: r = 1152 on the input side and r = 1152
      on the output side each cost under 1e-3 nats. Failure means
      the truncation machinery is wrong and VOIDS the run;
  (a) INPUT: reading through 64 of 1152 directions costs under
      0.10 nats;
  (b) OUTPUT: writing into 64 of 1152 directions costs under
      0.10 nats;
  (c) BEATS RANDOM: at r = 64, each SVD truncation costs less than
      a third of the same-rank RANDOM projection, three draws
      each. This is the bar 534's subset ranking failed, and it is
      the test of whether low-rank is the right operator;
  (d) THE BENCHMARK LINE: report the smallest r on each side whose
      cost stays under the per-token table's 1.466 nats, and the
      smallest r under 0.10. No bar -- these are the numbers.
  NULL: random projections must get monotonically worse as r
      falls. A random control that does not degrade with rank is
      not a control."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=0
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'mlp0_lowrank_results.json'
NFRESH=48
RANKS=[8,16,32,64,128,256,1152]

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    mlp=m.transformer.h[LJ].mlp
    oL=mlp.Left.weight.data.clone(); oR=mlp.Right.weight.data.clone()
    oD=mlp.Down.weight.data.clone()
    Lf=oL.float(); Rf=oR.float(); Df=oD.float()
    fresh=cl.fineweb_rows(NFRESH)
    # shared input basis from the stacked read maps
    _,_,Vin=torch.linalg.svd(torch.cat([Lf,Rf],0),
                             full_matrices=False)
    Uout,Sout,Vout=torch.linalg.svd(Df,full_matrices=False)

    def price():
        ce=torch.zeros(NFRESH,T)
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=bb.shape[0]
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            ce[i:i+B]=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                      reduction='none').view(B,T).cpu()
        return float(ce.mean())

    def restore():
        mlp.Left.weight.data.copy_(oL)
        mlp.Right.weight.data.copy_(oR)
        mlp.Down.weight.data.copy_(oD)

    def set_input(r,random=False,seed=0):
        if random:
            g=torch.Generator(device=DEV).manual_seed(seed)
            Q,_=torch.linalg.qr(torch.randn(D,r,generator=g,
                                            device=DEV))
        else:
            Q=Vin[:r].T
        P=Q@Q.T
        mlp.Left.weight.data.copy_((Lf@P).to(oL.dtype))
        mlp.Right.weight.data.copy_((Rf@P).to(oR.dtype))

    def set_output(r,random=False,seed=0):
        if random:
            g=torch.Generator(device=DEV).manual_seed(seed)
            Q,_=torch.linalg.qr(torch.randn(D,r,generator=g,
                                            device=DEV))
            mlp.Down.weight.data.copy_((Q@(Q.T@Df)).to(oD.dtype))
        else:
            mlp.Down.weight.data.copy_(
                ((Uout[:,:r]*Sout[:r])@Vout[:r]).to(oD.dtype))

    base=price()
    print(f'baseline CE {base:.4f} | per-token table reference '
          f'+1.4660 nats (534, same corpus)',flush=True)
    res={'input':{},'output':{}}
    for side,setter in (('input',set_input),('output',set_output)):
        for r in RANKS:
            setter(r); c=price()-base; restore()
            rnd=[]
            if r<D:
                for s in (1,2,3):
                    setter(r,True,s); rnd.append(round(price()-base,4))
                    restore()
            res[side][r]={'cost':round(c,4),'random':rnd}
            print(f'{side:>6} rank {r:>5}: cost {c:+.4f} | random '
                  f'{rnd}',flush=True)
            json.dump({k:{str(a):b for a,b in v.items()}
                       for k,v in res.items()},
                      open(OUT,'w'),indent=1)
    restore()
    p0=(abs(res['input'][D]['cost'])<1e-3
        and abs(res['output'][D]['cost'])<1e-3)
    print(f"(0) full rank free (input {res['input'][D]['cost']:+.5f}"
          f", output {res['output'][D]['cost']:+.5f}): "
          f"{'HELD' if p0 else 'FAILED -- VOID'}")
    if not p0:
        json.dump({'pred_0':False,'res':{k:{str(a):b
                   for a,b in v.items()} for k,v in res.items()}},
                  open(OUT,'w'),indent=1); return
    va,_=cl.score_bar('a',0.10-res['input'][64]['cost'],1e-9)
    vb,_=cl.score_bar('b',0.10-res['output'][64]['cost'],1e-9)
    okc=True
    for side in ('input','output'):
        r64=res[side][64]; best=min(r64['random']) if r64['random'] \
            else 1.0
        good=r64['cost']*3<max(best,1e-9)
        okc=okc and good
        print(f'(c) {side} rank 64: svd {r64["cost"]:+.4f} vs best '
              f'random {best:+.4f} -> {"ok" if good else "FAILED"}')
    lines={}
    for side in ('input','output'):
        u10=next((r for r in RANKS
                  if res[side][r]['cost']<0.10),None)
        ut=next((r for r in RANKS
                 if res[side][r]['cost']<1.466),None)
        lines[side]={'first_under_0.10':u10,'first_under_table':ut}
        print(f'(d) {side}: under 0.10 nats from rank {u10}; '
              f'under the table (1.466) from rank {ut}')
    mono=True
    for side in ('input','output'):
        rs=[r for r in RANKS if r<D and res[side][r]['random']]
        vals=[min(res[side][r]['random']) for r in rs]
        mono=mono and all(vals[i]>=vals[i+1]-1e-6
                          for i in range(len(vals)-1))
    print(f"NULL (random controls degrade as rank falls): "
          f"{'ok' if mono else 'VIOLATED'}")
    out={'baseline_ce':round(base,4),
         'input':{str(k):v for k,v in res['input'].items()},
         'output':{str(k):v for k,v in res['output'].items()},
         'table_reference':1.466,'benchmark_lines':lines,
         'pred_0':True,'pred_a':va=='HELD','pred_b':vb=='HELD',
         'pred_c':bool(okc),'null_ok':bool(mono),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
