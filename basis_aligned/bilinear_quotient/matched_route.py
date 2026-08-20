"""MATCHED ROUTE -- the two stand-in routes, finally on one scale.
549 withdrew the claim that tables beat rank truncation. On clean
held-out text three pair-indexed tables cost 1.6463 nats, while
545's best greedy rank allocation of all six blocks cost 1.18 --
but those two numbers were never comparable. 545 priced on
different rows (baseline CE 3.3536 against 3.1244 here), fitted
its SVD basis in-sample on the very positions it scored, and
covered six blocks where the tables cover three.
This measures both routes on exactly the same held-out rows, with
every fitted object -- table and basis alike -- built only on the
fitting rows, and at matched scope.
  route A, tables      blocks 0-2 replaced by pair-indexed tables
  route B, rank        blocks 0-2 interfaces projected to rank r,
                       swept, with the basis fitted out of sample
  route C, rank at
    matched budget     blocks 0-2 at the rank whose total
                       directions equal the tables' effective
                       description, reported for reference
  delete               blocks 0-2 writes replaced by their means,
                       the do-nothing baseline
The honest comparison needs one more thing the earlier runs never
reported: DESCRIPTION SIZE. A rank-r interface for three blocks is
3 x r x 1152 numbers of basis plus r numbers per position at run
time; a pair table is 64 numbers per key for 120,358 keys plus a
64 x 1152 basis. Both are printed so the costs can be read against
what each object actually costs to write down.
REGISTERED PREDICTIONS:
  (0) DISJOINT: zero priced rows appear in the fitting corpus,
      via cl.assert_disjoint. Any overlap VOIDS the run;
  (a) THE TABLES REPRODUCE: route A comes within 0.05 nats of
      549's 1.6463. Failure means the two runs differ somewhere
      unintended and VOIDS the comparison;
  (b) THE HONEST WINNER: report which route is cheaper at rank 64
      -- the width the tables use -- and by how much. I predict
      RANK WINS at matched width, because 549's reversal already
      pointed that way and because a projection keeps the top
      directions of the true write while a table must guess them
      from a 43.8%-covered index;
  (c) WHERE THEY CROSS: report the rank at which the projection
      route first costs more than the tables, if any. No bar;
  NULL: the delete baseline must cost more than both routes at
      every rank tested. A stand-in that is worse than replacing
      the write with its mean is not a stand-in, and if either
      route falls below that line the run says so."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; R=64
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'matched_route_results.json'
NFIT=800; NPRICE=96; BLOCKS=[0,1,2]
RANKS=[8,16,32,64,128,256]

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    V=m.transformer.wte.weight.shape[0]
    allrows=cl.rows()
    fit=allrows[:NFIT]; price_rows=allrows[NFIT:NFIT+NPRICE]
    ok,ns=cl.assert_disjoint(fit,price_rows,label='matched_route')
    if not ok:
        json.dump({'pred_0':False,'shared':ns},
                  open(OUT,'w'),indent=1); return
    st={}
    def write_hooks(b,sink):
        at=m.transformer.h[b].attn; mlp=m.transformer.h[b].mlp
        return [at.register_forward_hook(
                    lambda mo,i_,o_,b=b: st.__setitem__(
                        b,(o_[0] if isinstance(o_,tuple) else o_)
                        .detach().float())),
                mlp.register_forward_hook(
                    lambda mo,i_,o_,b=b: (sink.append(
                        (st[b]+o_.float()).reshape(-1,D)),o_)[1])]
    TAB={}
    for b in BLOCKS:
        sink=[]; hs=write_hooks(b,sink)
        for i in range(0,200,4):
            bb=fit[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous()
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for h in hs: h.remove()
        Y=torch.cat(sink); mu=Y.mean(0)
        _,_,Vh=torch.linalg.svd(Y-mu,full_matrices=False)
        del sink,Y
        uni=torch.zeros(V,R,device=DEV); cnt=torch.zeros(V,device=DEV)
        pidx={}; pac=[]; pcn=[]
        B64=Vh[:R]
        for i in range(0,NFIT,4):
            bb=fit[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous()
            sk=[]; hs=write_hooks(b,sk)
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            for h in hs: h.remove()
            W=(sk[0]-mu)@B64.T
            cur=idx.reshape(-1)
            prev=torch.cat([torch.zeros(idx.shape[0],1,
                dtype=idx.dtype,device=DEV),idx[:,:-1]],1).reshape(-1)
            uni.index_add_(0,cur,W)
            cnt.index_add_(0,cur,torch.ones_like(cur,
                                                 dtype=torch.float))
            for j,k in enumerate((prev.long()*V+cur.long())
                                 .cpu().tolist()):
                e=pidx.get(k)
                if e is None:
                    pidx[k]=len(pac); pac.append(W[j]); pcn.append(1)
                else:
                    pac[e]=pac[e]+W[j]; pcn[e]+=1
        TAB[b]={'mu':mu,'V':Vh,'B':B64,
                'uni':uni/cnt.clamp_min(1).unsqueeze(1),
                'pidx':pidx,
                'P':torch.stack(pac)/torch.tensor(pcn,device=DEV,
                    dtype=torch.float).unsqueeze(1)}
        print(f'block {b}: {len(pidx)} pairs tabulated',flush=True)

    def run(mode,r=None):
        ce=torch.zeros(price_rows.shape[0],T)
        for i in range(0,price_rows.shape[0],4):
            bb=price_rows[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            Bn=bb.shape[0]; hs=[]
            for b in BLOCKS:
                t=TAB[b]; sl={}
                hs.append(m.transformer.h[b].attn
                    .register_forward_hook(
                    (lambda sl: lambda mo,i_,o_: (sl.__setitem__(
                        'a',(o_[0] if isinstance(o_,tuple) else o_)
                        .detach().float()),o_)[1])(sl)))
                if mode=='table':
                    cur=idx.reshape(-1)
                    prev=torch.cat([torch.zeros(Bn,1,
                        dtype=idx.dtype,device=DEV),idx[:,:-1]],1) \
                        .reshape(-1)
                    keys=(prev.long()*V+cur.long()).cpu().tolist()
                    sel=torch.tensor([t['pidx'].get(k,-1)
                                      for k in keys],device=DEV)
                    C=t['uni'][cur].clone(); hit=sel>=0
                    if int(hit.sum()): C[hit]=t['P'][sel[hit]]
                    lk=(t['mu']+C@t['B']).view(Bn,T,D)
                    hs.append(m.transformer.h[b].mlp
                        .register_forward_hook(
                        (lambda lk,sl: lambda mo,i_,o_:
                         (lk-sl['a']).to(o_.dtype))(lk,sl)))
                elif mode=='rank':
                    Vr=t['V'][:r]; P=Vr.T@Vr; mu=t['mu']
                    def fm(mo,i_,o_,P=P,mu=mu,sl=sl):
                        a=sl['a']; tot=a+o_.float()
                        return ((mu+(tot-mu)@P)-a).to(o_.dtype)
                    hs.append(m.transformer.h[b].mlp
                              .register_forward_hook(fm))
                else:
                    mu=t['mu']
                    hs.append(m.transformer.h[b].mlp
                        .register_forward_hook(
                        (lambda mu,sl: lambda mo,i_,o_:
                         (mu.expand_as(o_.float())-sl['a'])
                         .to(o_.dtype))(mu,sl)))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            ce[i:i+Bn]=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none').view(Bn,T).cpu()
            for h in hs: h.remove()
        return float(ce.mean())

    base=run('none_at_all') if False else None
    # true baseline: no hooks
    ce=torch.zeros(price_rows.shape[0],T)
    for i in range(0,price_rows.shape[0],4):
        bb=price_rows[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        Bn=bb.shape[0]
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
        ce[i:i+Bn]=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                   reduction='none').view(Bn,T).cpu()
    base=float(ce.mean())
    tab=run('table')-base
    dele=run('delete')-base
    curve={r:round(run('rank',r)-base,4) for r in RANKS}
    npairs=sum(len(TAB[b]['pidx']) for b in BLOCKS)
    size_tab=npairs*R+len(BLOCKS)*R*D
    print(f'\nbaseline CE {base:.4f}')
    print(f'delete blocks 0-2: {dele:+.4f}')
    print(f'tables (pair-indexed): {tab:+.4f} | description '
          f'{size_tab/1e6:.1f}M numbers')
    for r in RANKS:
        print(f'rank {r:>4}: {curve[r]:+.4f} | description '
              f'{len(BLOCKS)*r*D/1e6:.2f}M numbers')
    p0=True
    pa=abs(tab-1.6463)<=0.05
    r64=curve[64]
    pb='rank' if r64<tab else 'table'
    cross=next((r for r in RANKS if curve[r]>tab),None)
    nul=dele>max([tab]+list(curve.values()))
    print(f"\n(a) tables reproduce 549's 1.6463 "
          f"({tab:+.4f}): {'HELD' if pa else 'FAILED -- VOID'}")
    print(f"(b) at rank 64 the cheaper route is {pb.upper()} "
          f"(rank {r64:+.4f} vs tables {tab:+.4f}, difference "
          f"{abs(r64-tab):.4f})")
    print(f"(c) projection first exceeds the tables at rank "
          f"{cross}")
    print(f"NULL (delete {dele:+.4f} costs more than every "
          f"stand-in): {'ok' if nul else 'VIOLATED'}")
    out={'baseline_ce':round(base,4),'delete':round(dele,4),
         'tables':round(tab,4),'rank_curve':{str(k):v
             for k,v in curve.items()},
         'n_pairs':npairs,'table_numbers':size_tab,
         'winner_at_64':pb,'crossover_rank':cross,
         'pred_0':True,'pred_a':bool(pa),'null_ok':bool(nul),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
