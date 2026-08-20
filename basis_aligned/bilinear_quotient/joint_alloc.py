"""JOINT ALLOC -- if the front needs joint optimization, do it.
543 showed the joint numbers of 540 and 541 are not sums of local
damage. Compressing block 0 costs 0.081 alone and 0.384 when the
other five are already compressed, and compressing blocks 4 and 5
on top of 0-3 IMPROVES cross-entropy by 0.10 and 0.20 nats. With
cancellation like that, no per-component rule can pick ranks, and
the conclusion was that a front-of-model stand-in needs joint
search rather than a table of individual answers.
This does the search. Starting from every block at full rank
(free by construction), the allocation is reduced greedily: at
each step every block is offered its next lower rank, all six
candidates are PRICED JOINTLY, and the cheapest reduction is
taken. That is the smallest honest version of joint optimization
and it costs one forward sweep per candidate.
The comparison is uniform allocation at the same total budget --
which is what 541 measured, and which came in at 1.314 nats for
six blocks at rank 64 (a budget of 384 directions).
REGISTERED PREDICTIONS:
  (0) THE START IS FREE: all blocks at rank 1152 cost under 1e-3
      nats. Failure VOIDS the run;
  (a) JOINT SEARCH WINS: at a total budget of 384 directions the
      greedy allocation costs at least 0.30 nats less than the
      uniform rank-64 allocation's 1.314. If greedy cannot beat
      uniform by that much, joint optimization is not the missing
      ingredient and 543's conclusion needs qualifying;
  (b) THE ANSWER IS LOPSIDED: the greedy allocation at that budget
      has a max-to-min rank ratio of at least 4. A uniform answer
      would mean the blocks really are interchangeable and the
      whole exercise reduces to picking one number;
  (c) THE ALLOCATION IS REPORTED, block by block, with the cost
      curve of the greedy path. No bar -- this is the artifact the
      benchmark wants;
  NULL: a RANDOM allocation with the same total budget, three
      draws, must cost at least as much as uniform. If random
      allocations beat uniform too, then uniform is simply a bad
      baseline and (a) is not evidence for search."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; NB=6
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'joint_alloc_results.json'
NFRESH=32
GRID=[1152,512,256,128,64,32,16]
BUDGET=384

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    fresh=cl.fineweb_rows(NFRESH)
    caps={i:[] for i in range(NB)}; st={}
    hs=[]
    for i in range(NB):
        hs.append(m.transformer.h[i].attn.register_forward_hook(
            (lambda i: lambda mo,i_,o_: st.__setitem__(
                i,(o_[0] if isinstance(o_,tuple) else o_)
                .detach().float()))(i)))
        hs.append(m.transformer.h[i].mlp.register_forward_hook(
            (lambda i: lambda mo,i_,o_: (caps[i].append(
                (st[i]+o_.float()).reshape(-1,D).cpu()),o_)[1])(i)))
    for i in range(0,NFRESH,4):
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous()
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
    for h in hs: h.remove()
    basis={}; means={}
    for i in range(NB):
        Y=torch.cat(caps[i]).to(DEV); means[i]=Y.mean(0)
        _,_,Vh=torch.linalg.svd(Y-means[i],full_matrices=False)
        basis[i]=Vh; caps[i]=None

    def price(alloc,random=False,seed=0):
        ce=torch.zeros(NFRESH,T)
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            Bn=bb.shape[0]; hs=[]
            for b,r in alloc.items():
                if r>=D: continue
                if random:
                    g=torch.Generator(device=DEV).manual_seed(
                        seed*100+b)
                    Q,_=torch.linalg.qr(torch.randn(D,r,
                        generator=g,device=DEV))
                    P=Q@Q.T
                else:
                    V=basis[b][:r]; P=V.T@V
                mu=means[b]; sl={}
                hs.append(m.transformer.h[b].attn
                          .register_forward_hook(
                    (lambda sl: lambda mo,i_,o_: (sl.__setitem__(
                        'a',(o_[0] if isinstance(o_,tuple) else o_)
                        .detach().float()),o_)[1])(sl)))
                def fm(mo,i_,o_,P=P,mu=mu,sl=sl):
                    a=sl['a']; tot=a+o_.float()
                    return ((mu+(tot-mu)@P)-a).to(o_.dtype)
                hs.append(m.transformer.h[b].mlp
                          .register_forward_hook(fm))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            ce[i:i+Bn]=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none').view(Bn,T).cpu()
            for h in hs: h.remove()
        return float(ce.mean())

    base=price({})
    start={i:GRID[0] for i in range(NB)}
    c0=price(start)-base
    print(f'baseline CE {base:.4f} | all blocks full rank '
          f'{c0:+.5f}',flush=True)
    p0=abs(c0)<1e-3
    print(f"(0) start is free: {'HELD' if p0 else 'FAILED -- VOID'}")
    if not p0:
        json.dump({'pred_0':False,'start_cost':c0},
                  open(OUT,'w'),indent=1); return
    alloc=dict(start); path=[]
    step=0
    while sum(alloc.values())>BUDGET and step<40:
        best=None
        for b in range(NB):
            gi=GRID.index(alloc[b])
            if gi+1>=len(GRID): continue
            trial=dict(alloc); trial[b]=GRID[gi+1]
            c=price(trial)-base
            if best is None or c<best[0]: best=(c,b,GRID[gi+1])
        if best is None: break
        c,b,r=best
        alloc[b]=r; step+=1
        path.append({'step':step,'block':b,'new_rank':r,
                     'alloc':dict(alloc),
                     'budget':sum(alloc.values()),
                     'cost':round(c,4)})
        print(f'step {step}: block {b} -> rank {r} | budget '
              f'{sum(alloc.values())} | cost {c:+.4f}',flush=True)
        json.dump({'path':path},open(OUT,'w'),indent=1)
    greedy_cost=path[-1]['cost'] if path else c0
    greedy_alloc=dict(alloc)
    uni={i:64 for i in range(NB)}
    uni_cost=price(uni)-base
    rnd=[]
    g=torch.Generator().manual_seed(5)
    for s in range(3):
        while True:
            pick={i:GRID[int(torch.randint(1,len(GRID),(1,),
                  generator=g))] for i in range(NB)}
            if abs(sum(pick.values())-BUDGET)<=BUDGET*0.35: break
        rnd.append({'alloc':pick,'budget':sum(pick.values()),
                    'cost':round(price(pick)-base,4)})
        print(f'random alloc {pick} budget {sum(pick.values())}: '
              f'{rnd[-1]["cost"]:+.4f}',flush=True)
    va=(uni_cost-greedy_cost)>=0.30
    ranks=list(greedy_alloc.values())
    vb=(max(ranks)/max(min(ranks),1))>=4
    nul=all(r['cost']>=uni_cost-1e-6 for r in rnd)
    print(f"\ngreedy allocation {greedy_alloc} (budget "
          f"{sum(ranks)}): {greedy_cost:+.4f}")
    print(f"uniform rank 64 (budget 384): {uni_cost:+.4f}")
    print(f"(a) greedy beats uniform by "
          f"{uni_cost-greedy_cost:+.4f} >= 0.30: "
          f"{'HELD' if va else 'FAILED'}")
    print(f"(b) max/min rank ratio "
          f"{max(ranks)/max(min(ranks),1):.1f} >= 4: "
          f"{'HELD' if vb else 'FAILED'}")
    print(f"NULL (random allocations no better than uniform): "
          f"{'ok' if nul else 'VIOLATED'}")
    out={'baseline_ce':round(base,4),'path':path,
         'greedy_alloc':greedy_alloc,
         'greedy_cost':round(greedy_cost,4),
         'greedy_budget':sum(ranks),
         'uniform_cost':round(uni_cost,4),'random':rnd,
         'pred_0':True,'pred_a':bool(va),'pred_b':bool(vb),
         'null_ok':bool(nul),'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
