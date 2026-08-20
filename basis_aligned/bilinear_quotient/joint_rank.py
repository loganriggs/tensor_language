"""JOINT RANK -- the honest front-of-model number.
538 produced a per-component rank table and then undercut it:
truncating all six early attention layers at once, each to the
rank that passes 0.10 nats on its own, costs 0.718 nats where the
individual costs sum to 0.19. Compression is superadditive across
layers by nearly a factor of four, so every number in that table
flatters a stand-in that nobody can actually build.
The honest measurement truncates jointly. All six early attention
layers to a COMMON rank, swept; all six early MLPs to a common
rank, swept; and then both together, which is the front-of-model
stand-in the benchmark is actually asking about.
Everything is an exact SVD of a weight matrix, so each point on
each curve is a statement of the form "the first six blocks read
their inputs through r directions each" with no fitting anywhere.
REGISTERED PREDICTIONS:
  (0) FULL RANK IS FREE on all three curves (cost < 1e-3 nats).
      Failure VOIDS the run;
  (a) ATTENTION JOINTLY: all six attention layers at a common rank
      32 cost under 0.30 nats. 538's mixed-rank attempt cost 0.718
      with a1 and a5 left untruncated at 128, so a uniform 32 is a
      harder condition and this bar could easily fail;
  (b) SUPERADDITIVITY QUANTIFIED: at rank 32, the joint attention
      cost is at least twice the sum of the six individual costs
      at the same rank. 538 measured a factor of four with mixed
      ranks; this asks whether it survives a uniform one, and the
      ratio is the number to report;
  (c) THE FRONT-OF-MODEL LINE: report the smallest common rank at
      which attention-only, MLP-only, and both-together each stay
      under 0.30 nats, and the cost of both-together at rank 64.
      No bar -- these are the benchmark numbers, and quoting them
      honestly matters more than passing anything;
  NULL: at the rank chosen in (c) for both-together, a RANDOM
      projection of the same rank in every component must cost at
      least three times as much. 538 found this null fails for
      individually-insensitive components, so it is checked on the
      joint intervention where insensitivity cannot hide."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'joint_rank_results.json'
NFRESH=48; NLAYERS=6
ARANKS=[8,16,32,64,128]
MRANKS=[32,64,128,256,512,1152]

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

    def restore_all():
        for li in range(NLAYERS):
            at=m.transformer.h[li].attn; mlp=m.transformer.h[li].mlp
            for w,o in zip((at.c_q,at.c_k,at.c_q2,at.c_k2),
                           orig[f'a{li}']): w.weight.data.copy_(o)
            mlp.Left.weight.data.copy_(orig[f'm{li}'][0])
            mlp.Right.weight.data.copy_(orig[f'm{li}'][1])

    def apply_attn(li,r,random=False,seed=0):
        at=m.transformer.h[li].attn
        NH=at.c_q.weight.shape[0]//128
        for wi,w in enumerate((at.c_q,at.c_k,at.c_q2,at.c_k2)):
            W=orig[f'a{li}'][wi].float().clone()
            for h in range(NH):
                blk=W[h*128:(h+1)*128]
                if random:
                    g=torch.Generator(device=DEV).manual_seed(
                        seed*10000+li*100+wi*10+h)
                    Q,_=torch.linalg.qr(torch.randn(D,r,generator=g,
                                                    device=DEV))
                    W[h*128:(h+1)*128]=blk@Q@Q.T
                else:
                    U,S,Vh=torch.linalg.svd(blk,full_matrices=False)
                    W[h*128:(h+1)*128]=(U[:,:r]*S[:r])@Vh[:r]
            w.weight.data.copy_(W.to(w.weight.dtype))

    def apply_mlp(li,r,random=False,seed=0):
        mlp=m.transformer.h[li].mlp
        Lf=orig[f'm{li}'][0].float(); Rf=orig[f'm{li}'][1].float()
        if random:
            g=torch.Generator(device=DEV).manual_seed(
                seed*10000+li*100+7)
            Q,_=torch.linalg.qr(torch.randn(D,r,generator=g,
                                            device=DEV))
        else:
            _,_,V=torch.linalg.svd(torch.cat([Lf,Rf],0),
                                   full_matrices=False)
            Q=V[:r].T
        P=Q@Q.T
        mlp.Left.weight.data.copy_((Lf@P).to(mlp.Left.weight.dtype))
        mlp.Right.weight.data.copy_((Rf@P).to(mlp.Right.weight.dtype))

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
    curves={'attn':{},'mlp':{},'both':{}}
    # attention only
    for r in ARANKS:
        for li in range(NLAYERS): apply_attn(li,r)
        c=price()-base; restore_all()
        indiv=[]
        for li in range(NLAYERS):
            apply_attn(li,r); indiv.append(price()-base); restore_all()
        curves['attn'][r]={'joint':round(c,4),
                           'sum_individual':round(sum(indiv),4),
                           'individual':[round(x,4) for x in indiv]}
        print(f'attn  r={r:>4}: joint {c:+.4f} | sum of individual '
              f'{sum(indiv):+.4f} | ratio '
              f'{c/max(sum(indiv),1e-6):.2f}',flush=True)
        json.dump({k:{str(a):b for a,b in v.items()}
                   for k,v in curves.items()},open(OUT,'w'),indent=1)
    # mlp only
    for r in MRANKS:
        for li in range(NLAYERS): apply_mlp(li,r)
        c=price()-base; restore_all()
        curves['mlp'][r]={'joint':round(c,4)}
        print(f'mlp   r={r:>4}: joint {c:+.4f}',flush=True)
        json.dump({k:{str(a):b for a,b in v.items()}
                   for k,v in curves.items()},open(OUT,'w'),indent=1)
    # both together, attention at r and MLPs at 4r (same fraction)
    for r in ARANKS:
        rm=min(r*9,1152)
        for li in range(NLAYERS):
            apply_attn(li,r); apply_mlp(li,rm)
        c=price()-base; restore_all()
        rnd=[]
        if r==64:
            for s in (1,2,3):
                for li in range(NLAYERS):
                    apply_attn(li,r,True,s); apply_mlp(li,rm,True,s)
                rnd.append(round(price()-base,4)); restore_all()
        curves['both'][r]={'joint':round(c,4),'mlp_rank':rm,
                           'random':rnd}
        print(f'both  attn r={r:>4} mlp r={rm:>4}: joint {c:+.4f}'
              +(f' | random {rnd}' if rnd else ''),flush=True)
        json.dump({k:{str(a):b for a,b in v.items()}
                   for k,v in curves.items()},open(OUT,'w'),indent=1)
    restore_all()
    p0=(abs(curves['attn'][128]['joint'])<1e-3
        and abs(curves['mlp'][1152]['joint'])<1e-3)
    _fa=curves['attn'][128]['joint']; _fm=curves['mlp'][1152]['joint']
    print(f"\n(0) full rank free (attn {_fa:+.5f}, mlp {_fm:+.5f}): "
          f"{'HELD' if p0 else 'FAILED -- VOID'}")
    va=curves['attn'][32]['joint']<0.30
    ratio=(curves['attn'][32]['joint']
           /max(curves['attn'][32]['sum_individual'],1e-6))
    vb=ratio>=2.0
    firsts={}
    for k,grid in (('attn',ARANKS),('mlp',MRANKS),('both',ARANKS)):
        firsts[k]=next((r for r in grid
                        if curves[k][r]['joint']<0.30),None)
    b64=curves['both'][64]
    best=min(b64['random']) if b64['random'] else 1.0
    nul=best>=3*max(b64['joint'],1e-6)
    print(f"(a) six attention layers jointly at rank 32 cost "
          f"{curves['attn'][32]['joint']:+.4f} < 0.30: "
          f"{'HELD' if va else 'FAILED'}")
    print(f"(b) superadditivity at rank 32: joint/sum = "
          f"{ratio:.2f} >= 2.0: {'HELD' if vb else 'FAILED'}")
    print(f"(c) smallest common rank under 0.30 nats: {firsts} | "
          f"both-together at attn rank 64: {b64['joint']:+.4f}")
    print(f"NULL (random at the joint rank costs >=3x: {best:+.4f} "
          f"vs {b64['joint']:+.4f}): {'ok' if nul else 'VIOLATED'}")
    out={'baseline_ce':round(base,4),
         'curves':{k:{str(a):b for a,b in v.items()}
                   for k,v in curves.items()},
         'superadditivity_at_32':round(ratio,2),
         'first_under_0.30':firsts,'pred_0':bool(p0),
         'pred_a':bool(va),'pred_b':bool(vb),'null_ok':bool(nul),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
