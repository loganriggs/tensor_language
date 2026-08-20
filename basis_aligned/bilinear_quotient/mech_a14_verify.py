"""MECH A14 VERIFY -- the last screen positive, tested causally.
509 killed the two new ones: silencing m5's contribution to m14
and m15 at r.1.2.0 and r.1.2.2 damages member positions by 0.0006
and 0.0037 nats, while matched RANDOM directions damage them by
0.012 to 0.026 -- three to seven times more. The screen's
correlation did not survive.
That leaves r.3.0.2, the census's oldest and strongest screen
positive: a15, a16 and a17 all enriched for a14's writes at
1.95-2.01, unchanged by the decomposition correction because its
components are late enough that the intervening lambda product is
about 1. It has never had a causal test.
Same surgery, corrected readout. 509 showed the member-versus-
offslice contrast is worthless as a bar -- member positions were
SELECTED for being damage-sensitive, so random directions score
8x and 18x on it. The readout here is member damage measured
against matched random directions, which is the comparison that
can actually fail.
REGISTERED PREDICTIONS:
  (0) EXACTNESS to 1e-4 relative, checked before scoring;
  (a) BEATS PEERS: a14's member damage at r.3.0.2 exceeds the
      largest among peer leaves on the same components;
  (b) BEATS NOISE: it exceeds the largest of FIVE matched random
      directions substituted the same way. This is the bar 509's
      leads failed by a factor of three to seven;
  (c) WRITER-SPECIFIC: it exceeds the same surgery on a13 and a16,
      the neighbouring writers.
  NULL: identical to (b), stated separately so the run cannot be
      read as passing when noise wins.
If this fails too, the screen has produced no causally verified
writer-level mechanism in 60 leaves on either decomposition, and
SOP step 3M needs rebuilding rather than re-running."""
import ast, json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'mech_a14_verify_results.json'
TAGS=['r.3.0.2']
COMPS=[15,16,17]

@torch.no_grad()
def ce_grid(writer=None,seed=None):
    """CE over the census grid; if writer is given, that writer's
    contribution to each of m14/m15's inputs is replaced by its own
    mean over positions. writer='RAND' uses a matched random
    direction instead."""
    R=cl.rows(); out=torch.zeros(R.shape[0],T); errs=[]
    for i in range(0,R.shape[0],4):
        bb=R[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        B=bb.shape[0]
        outs={}; hs=[]
        for lj in range(max(COMPS)+1):
            for kind,mod in (('a',m.transformer.h[lj].attn),
                             ('m',m.transformer.h[lj].mlp)):
                def mk(k9=f'{kind}{lj}'):
                    def h(mo,i_,o_):
                        y=o_[0] if isinstance(o_,tuple) else o_
                        outs[k9]=y.detach().float()
                    return h
                hs.append(mod.register_forward_hook(mk()))
        E=F.rms_norm(m.transformer.wte(idx),(D,)).float()
        def mkpre(li):
            def ph(mo_,args):
                X=args[0]
                parts=cl.writer_parts(li,E,outs,'a')
                tot=sum(parts.values())
                errs.append(float((F.rms_norm(tot,(D,))-X.float())
                            .norm()/X.float().norm().clamp_min(1e-9)))
                if writer is None: return None
                if writer=='RAND':
                    p=parts['a14']
                    g=torch.Generator(device=DEV).manual_seed(seed)
                    r=torch.randn(p.shape,generator=g,device=DEV)
                    r=r/r.norm()*p.norm()
                    t2=tot-p+r
                else:
                    if writer not in parts: return None
                    p=parts[writer]
                    t2=tot-p+p.mean(dim=(0,1),keepdim=True)
                return (F.rms_norm(t2,(D,)).to(X.dtype),)+tuple(args[1:])
            return ph
        for li in COMPS:
            hs.append(m.transformer.h[li].attn
                      .register_forward_pre_hook(mkpre(li)))
        x=E.to(m.transformer.wte.weight.dtype); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
        out[i:i+B]=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                   reduction='none').view(B,T).cpu()
        for h in hs: h.remove()
    return out.reshape(-1), (max(errs) if errs else 0.0)

def contrast(base,abl,mem,off):
    dm=float((abl-base)[mem].mean()); do=float((abl-base)[off].mean())
    return dm,do

def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    NF=cl.nflat()
    base,relerr=ce_grid(None)
    print(f'reconstruction relative error: {relerr:.3e}',flush=True)
    if relerr>1e-4:
        print('*** (0) EXACTNESS FAILED -- run VOID ***')
        json.dump({'pred_0':False,'relerr':relerr},
                  open(OUT,'w'),indent=1); return
    print('(0) exactness: HELD',flush=True)
    abl={}
    for w in ('a14','a13','a16'):
        abl[w],_=ce_grid(w)
        print(f'ran {w}',flush=True)
    rnd=[]
    for s in (5,17,29,41,53):
        r,_=ce_grid('RAND',seed=s); rnd.append(r)
        print(f'ran RAND seed {s}',flush=True)
    # peer leaves using the same components
    def machinery(tg):
        pr=[ast.literal_eval(p) if isinstance(p,str) else p
            for p in cl.leaf(tg)['top_probes']]
        return sorted({p[1] for p in pr if p[0]=='pca'})
    tgt_keys=machinery(TAGS[0])
    peers=[t for t in cl.all_tags()
           if t not in TAGS and machinery(t)==tgt_keys][:6]
    print(f'peers on {tgt_keys}: {peers}',flush=True)
    res={}
    for tag in TAGS+peers:
        lf=cl.leaf(tag)
        mem=torch.zeros(NF,dtype=torch.bool); mem[lf['member']]=True
        off=torch.zeros(NF,dtype=torch.bool); off[lf['slice']]=True
        off=~off
        row={'n_member':int(mem.sum())}
        for w in ('a14','a13','a16'):
            dm,do=contrast(base,abl[w],mem,off)
            row[w]={'member':round(dm,5),'offslice':round(do,5)}
        rv=[contrast(base,r,mem,off) for r in rnd]
        row['rand']=[{'member':round(a,5),'offslice':round(b,5)}
                     for a,b in rv]
        res[tag]=row
        print(f"{tag}: a14 member {row['a14']['member']:+.5f} "
              f"offslice {row['a14']['offslice']:+.5f} | a13 "
              f"{row['a13']['member']:+.5f} | a16 "
              f"{row['a16']['member']:+.5f}",flush=True)
        json.dump(res,open(OUT,'w'),indent=1)
    verdicts={}
    for tag in TAGS:
        r=res[tag]
        dm,do=r['a14']['member'],r['a14']['offslice']
        rmean=sum(x['member'] for x in r['rand'])/len(r['rand'])
        rmax=max(x['member'] for x in r['rand'])
        vb,_=cl.score_bar(f'{tag}-b',dm-rmax,1e-9)
        print(f'   member damage a14 {dm:+.5f} vs random mean '
              f'{rmean:+.5f} max {rmax:+.5f}')
        peer_best=max((res[p]['a14']['member'] for p in peers),
                      default=0.0)
        va,_=cl.score_bar(f'{tag}-a',dm-peer_best,1e-9)
        vc='HELD' if (dm>r['a13']['member']
                      and dm>r['a16']['member']) else 'FAILED'
        rr=rmax
        nul=dm>rmax
        print(f'({tag}-c) m5 member {dm:+.5f} beats m4 '
              f"{r['a13']['member']:+.5f} and m6 "
              f"{r['a16']['member']:+.5f}: {vc}")
        print(f'({tag}-NULL) real beats every random direction '
              f"({dm:+.5f} vs {rr:+.5f}): {'ok' if nul else 'VIOLATED'}")
        verdicts[tag]={'a':va,'b':vb,'c':vc,'null_ok':bool(nul),
                       'peer_best':round(peer_best,5)}
    out={'per_leaf':res,'verdicts':verdicts,'peers':peers,
         'relerr':relerr,'pred_0':True,'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
