"""INTERFACE SWEEP -- compress what a block sends, jointly.
540 killed the weight-truncation route to a front-of-model
stand-in: six attention layers at a common rank cost 0.51 nats,
six MLPs cost 1.54 where their individual costs sum to 0.18, and
both together cost 1.78 -- worse than replacing a single MLP with
a fitted lookup table. Per-component ranks do not compose.
539 found the opposite for outputs. The rest of the network reads
layer 0 only through what layer 0 writes, and projecting that
write onto 64 of 1152 principal directions costs 0.081 nats
against 0.57-0.61 for a random 64-dimensional interface. The
finding was one block deep and one measurement wide.
This asks whether the output route composes where the weight route
did not, which is the question that decides whether a benchmark
should be built on interfaces. For each of the first six blocks
the combined write (attn_i + mlp_i) is projected onto its top r
principal directions, with the discarded part replaced by its mean
so only the position-dependent portion is removed. Then all six
are projected AT ONCE.
Projecting a block's write leaves that block's own computation
untouched -- it changes only what the block sends onward -- so a
low joint rank here is exactly the claim "the first six blocks
communicate upward through r numbers each".
REGISTERED PREDICTIONS:
  (0) FULL RANK IS FREE, singly and jointly (cost < 1e-3 nats).
      Failure VOIDS the run;
  (a) EACH BLOCK IS NARROW: every one of the six blocks passes
      0.10 nats at rank <= 128 on its own;
  (b) IT COMPOSES: all six writes projected to rank 64
      SIMULTANEOUSLY cost under 0.30 nats. This is the bar the
      weight route failed at 1.78, and it is the point of the run.
      If it fails too, then nothing about the front of this model
      is replaceable at a few dozen numbers and the benchmark
      needs a different target;
  (c) THE LINE: report the smallest joint rank under 0.30 nats and
      under 0.10, and the per-block ranks. No bar;
  NULL: at the joint rank chosen in (b), random interfaces of the
      same rank in every block must cost at least three times as
      much. 540's null failed because the damage was large enough
      to leave the measurable regime, so if this one fails the
      same way the joint number should not be quoted."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'interface_sweep_results.json'
NFRESH=48; NB=6
RANKS=[8,16,32,64,128,256,1152]

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    fresh=cl.fineweb_rows(NFRESH)
    # capture each block's combined write
    caps={i:{'a':[],'m':[]} for i in range(NB)}
    hs=[]
    for i in range(NB):
        hs.append(m.transformer.h[i].attn.register_forward_hook(
            (lambda i: lambda mo,i_,o_: caps[i]['a'].append(
                (o_[0] if isinstance(o_,tuple) else o_)
                .detach().float().reshape(-1,D).cpu()))(i)))
        hs.append(m.transformer.h[i].mlp.register_forward_hook(
            (lambda i: lambda mo,i_,o_: caps[i]['m'].append(
                o_.detach().float().reshape(-1,D).cpu()))(i)))
    for i in range(0,NFRESH,4):
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous()
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
    for h in hs: h.remove()
    basis={}; means={}
    for i in range(NB):
        Y=(torch.cat(caps[i]['a'])+torch.cat(caps[i]['m'])).to(DEV)
        mu=Y.mean(0); means[i]=mu
        _,S,Vh=torch.linalg.svd(Y-mu,full_matrices=False)
        basis[i]=Vh
        pr=float((S.sum()**2)/S.pow(2).sum().clamp_min(1e-12))
        print(f'block {i}: write norm '
              f'{float(Y.norm(dim=-1).mean()):.1f} | participation '
              f'ratio {pr:.1f}',flush=True)
        caps[i]=None

    def hooks_for(blocks,r,random=False,seed=0):
        hs=[]
        for i in blocks:
            if random:
                g=torch.Generator(device=DEV).manual_seed(
                    seed*100+i)
                Q,_=torch.linalg.qr(torch.randn(D,r,generator=g,
                                                device=DEV))
                P=Q@Q.T
            else:
                V=basis[i][:r]; P=V.T@V
            mu=means[i]; store={}
            hs.append(m.transformer.h[i].attn.register_forward_hook(
                (lambda st: lambda mo,i_,o_: (st.__setitem__(
                    'a',(o_[0] if isinstance(o_,tuple) else o_)
                    .detach().float()),o_)[1])(store)))
            def fm(mo,i_,o_,P=P,mu=mu,st=store):
                a=st['a']; tot=a+o_.float()
                new=mu+(tot-mu)@P
                return (new-a).to(o_.dtype)
            hs.append(m.transformer.h[i].mlp
                      .register_forward_hook(fm))
        return hs

    def price(blocks=None,r=None,random=False,seed=0):
        ce=torch.zeros(NFRESH,T)
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=bb.shape[0]
            hs=hooks_for(blocks,r,random,seed) if blocks else []
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
    per={}; joint={}
    for i in range(NB):
        row={}
        for r in RANKS:
            row[r]=round(price([i],r)-base,4)
        per[i]=row
        passing=next((r for r in RANKS if row[r]<0.10),None)
        print(f'block {i}: {row} | passes 0.10 at rank {passing}',
              flush=True)
        json.dump({'per':{str(k):v for k,v in per.items()}},
                  open(OUT,'w'),indent=1)
    for r in RANKS:
        c=price(list(range(NB)),r)-base
        rnd=[]
        if r==64:
            for s in (1,2,3):
                rnd.append(round(price(list(range(NB)),r,True,s)
                                 -base,4))
        joint[r]={'cost':round(c,4),'random':rnd}
        print(f'joint r={r:>5}: {c:+.4f}'
              +(f' | random {rnd}' if rnd else ''),flush=True)
        json.dump({'per':{str(k):v for k,v in per.items()},
                   'joint':{str(k):v for k,v in joint.items()}},
                  open(OUT,'w'),indent=1)
    p0=(abs(joint[1152]['cost'])<1e-3
        and all(abs(per[i][1152])<1e-3 for i in range(NB)))
    print(f"\n(0) full rank free (joint "
          f"{joint[1152]['cost']:+.5f}): "
          f"{'HELD' if p0 else 'FAILED -- VOID'}")
    if not p0:
        json.dump({'pred_0':False},open(OUT,'w'),indent=1); return
    firsts={i:next((r for r in RANKS if per[i][r]<0.10),None)
            for i in range(NB)}
    va=all(v is not None and v<=128 for v in firsts.values())
    vb,_=cl.score_bar('b',0.30-joint[64]['cost'],1e-9)
    j30=next((r for r in RANKS if joint[r]['cost']<0.30),None)
    j10=next((r for r in RANKS if joint[r]['cost']<0.10),None)
    best=min(joint[64]['random']) if joint[64]['random'] else 1.0
    nul=best>=3*max(joint[64]['cost'],1e-6)
    print(f"(a) every block passes 0.10 at rank <=128: {firsts} -> "
          f"{'HELD' if va else 'FAILED'}")
    print(f"(c) joint rank under 0.30: {j30} | under 0.10: {j10}")
    print(f"NULL (random joint interface {best:+.4f} vs "
          f"{joint[64]['cost']:+.4f}): "
          f"{'ok' if nul else 'VIOLATED'}")
    out={'baseline_ce':round(base,4),
         'per_block':{str(k):v for k,v in per.items()},
         'joint':{str(k):v for k,v in joint.items()},
         'per_block_passing':{str(k):v for k,v in firsts.items()},
         'joint_under_0.30':j30,'joint_under_0.10':j10,
         'pred_0':True,'pred_a':bool(va),'pred_b':vb=='HELD',
         'null_ok':bool(nul),'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
