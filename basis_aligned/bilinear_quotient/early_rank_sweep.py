"""EARLY RANK SWEEP -- the benchmark table for the first six blocks.
535 found that attention layer 0 compares tokens along TWO
directions of its 128 (0.053 nats), while random projections of
the same rank cost 1.2-1.4. 534 found mlp0 needs 22% of its atoms
before a subset stand-in reaches 0.10 nats, though a low-rank
truncation of the map has not yet been tried there.
Those are two data points. The benchmark wants a table: for each
early component, how few directions reproduce the model, and does
the answer differ between attention and MLP.
Every truncation here is an exact SVD of a weight matrix, not a
fit, and each is an interpretable claim in its own right --
"this component reads its input through r directions".
  attention layers 0-5: rank-truncate both QK factors per head
  MLPs 0-5: rank-truncate the shared input basis of L and R, so
    the MLP reads x through r of 1152 directions
Ranks 2, 8, 32, 128 for attention (of 128 per head) and 8, 32,
128, 512 for MLPs (of 1152), chosen so both grids span the same
fraction of their space.
REGISTERED PREDICTIONS:
  (0) FULL RANK IS FREE for every one of the twelve components
      (cost < 1e-3 nats). Any component failing this has broken
      truncation machinery and is dropped from the table rather
      than reported;
  (a) ATTENTION IS CHEAP: at least three of the six attention
      layers reach under 0.10 nats at rank <= 8;
  (b) ATTENTION BEATS MLP: the median smallest-passing rank,
      expressed as a FRACTION of each component's input dimension,
      is lower for the six attention layers than for the six MLPs.
      This makes 535's aside -- that the attention layer is more
      compressible than the MLP beside it -- into a claim across
      six blocks that can fail;
  (c) COMPOSITION HOLDS: truncating ALL SIX attention layers
      simultaneously, each to its own smallest passing rank, costs
      under 0.30 nats. Compression that only works one component
      at a time is not a stand-in for the front of the model;
  NULL: at each component's chosen rank, a RANDOM projection of
      the same rank must cost at least five times as much. 535
      measured 1.2-1.4 nats for random against 0.05 for SVD at
      layer 0, so this bar is set well inside what was already
      observed."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'early_rank_sweep_results.json'
NFRESH=48; NLAYERS=6
ARANKS=[2,8,32,128]; MRANKS=[8,32,128,512]

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    fresh=cl.fineweb_rows(NFRESH)
    orig={}
    for li in range(NLAYERS):
        at=m.transformer.h[li].attn; mlp=m.transformer.h[li].mlp
        orig[f'a{li}']=[w.weight.data.clone()
                        for w in (at.c_q,at.c_k,at.c_q2,at.c_k2)]
        orig[f'm{li}']=[mlp.Left.weight.data.clone(),
                        mlp.Right.weight.data.clone()]

    def restore(key):
        li=int(key[1:])
        if key[0]=='a':
            at=m.transformer.h[li].attn
            for w,o in zip((at.c_q,at.c_k,at.c_q2,at.c_k2),
                           orig[key]): w.weight.data.copy_(o)
        else:
            mlp=m.transformer.h[li].mlp
            mlp.Left.weight.data.copy_(orig[key][0])
            mlp.Right.weight.data.copy_(orig[key][1])

    def apply(key,r,random=False,seed=0):
        li=int(key[1:])
        if key[0]=='a':
            at=m.transformer.h[li].attn
            NH=at.c_q.weight.shape[0]//128
            for wi,w in enumerate((at.c_q,at.c_k,at.c_q2,at.c_k2)):
                W=orig[key][wi].float().clone()
                for h in range(NH):
                    blk=W[h*128:(h+1)*128]
                    if random:
                        g=torch.Generator(device=DEV).manual_seed(
                            seed*1000+wi*10+h)
                        Q,_=torch.linalg.qr(
                            torch.randn(D,r,generator=g,device=DEV))
                        W[h*128:(h+1)*128]=blk@Q@Q.T
                    else:
                        U,S,Vh=torch.linalg.svd(blk,
                                                full_matrices=False)
                        W[h*128:(h+1)*128]=(U[:,:r]*S[:r])@Vh[:r]
                w.weight.data.copy_(W.to(w.weight.dtype))
        else:
            mlp=m.transformer.h[li].mlp
            Lf=orig[key][0].float(); Rf=orig[key][1].float()
            if random:
                g=torch.Generator(device=DEV).manual_seed(seed)
                Q,_=torch.linalg.qr(torch.randn(D,r,generator=g,
                                                device=DEV))
            else:
                _,_,V=torch.linalg.svd(torch.cat([Lf,Rf],0),
                                       full_matrices=False)
                Q=V[:r].T
            P=Q@Q.T
            mlp.Left.weight.data.copy_((Lf@P).to(
                mlp.Left.weight.dtype))
            mlp.Right.weight.data.copy_((Rf@P).to(
                mlp.Right.weight.dtype))

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

    base=price()
    print(f'baseline CE {base:.4f}',flush=True)
    table={}; broken=[]
    for key in [f'a{i}' for i in range(NLAYERS)] \
              +[f'm{i}' for i in range(NLAYERS)]:
        grid=ARANKS if key[0]=='a' else MRANKS
        dim=128 if key[0]=='a' else D
        row={}
        for r in grid:
            apply(key,r); c=price()-base; restore(key)
            row[r]=round(c,4)
        if abs(row[grid[-1]])>=1e-3 and grid[-1]==dim:
            broken.append(key)
        passing=next((r for r in grid if row[r]<0.10),None)
        rnd=[]
        if passing is not None and passing<dim:
            for s in (1,2,3):
                apply(key,passing,True,s)
                rnd.append(round(price()-base,4)); restore(key)
        table[key]={'costs':row,'passing_rank':passing,
                    'dim':dim,
                    'fraction':round(passing/dim,4)
                              if passing else None,
                    'random_at_passing':rnd}
        print(f"{key}: {row} | passes 0.10 at rank {passing} "
              f"({table[key]['fraction']}) | random {rnd}",
              flush=True)
        json.dump(table,open(OUT,'w'),indent=1)
    # (c) all attention layers together
    combo=None
    picks={k:v['passing_rank'] for k,v in table.items()
           if k[0]=='a' and v['passing_rank']}
    if len(picks)==NLAYERS:
        for k,r in picks.items(): apply(k,r)
        combo=price()-base
        for k in picks: restore(k)
        print(f'\n(c) all six attention layers at their own passing '
              f'ranks {picks}: {combo:+.4f} nats',flush=True)
    acheap=[k for k,v in table.items()
            if k[0]=='a' and v['passing_rank'] and v['passing_rank']<=8]
    pa=len(acheap)>=3
    def med(pref):
        f=[v['fraction'] for k,v in table.items()
           if k[0]==pref and v['fraction'] is not None]
        return sorted(f)[len(f)//2] if f else None
    ma,mm=med('a'),med('m')
    pb=(ma is not None and mm is not None and ma<mm)
    pc=(combo is not None and combo<0.30)
    nul=all(not v['random_at_passing'] or
            min(v['random_at_passing'])>=5*max(
                v['costs'][v['passing_rank']],1e-4)
            for v in table.values())
    print(f"(a) >=3 attention layers pass at rank <=8: {acheap} -> "
          f"{'HELD' if pa else 'FAILED'}")
    print(f"(b) median passing fraction attention {ma} vs MLP {mm}: "
          f"{'HELD' if pb else 'FAILED'}")
    print(f"(c) composition under 0.30 nats: "
          f"{'HELD' if pc else 'FAILED'} ({combo})")
    print(f"NULL (random costs >=5x at each passing rank): "
          f"{'ok' if nul else 'VIOLATED'}")
    if broken: print(f'*** full-rank not free for {broken} ***')
    out={'baseline_ce':round(base,4),'table':table,
         'combo_cost':round(combo,4) if combo is not None else None,
         'combo_ranks':picks,'median_fraction_attn':ma,
         'median_fraction_mlp':mm,'broken':broken,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'null_ok':bool(nul),'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
