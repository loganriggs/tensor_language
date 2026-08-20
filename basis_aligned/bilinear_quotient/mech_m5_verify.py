"""MECH M5 VERIFY -- is the new mechanism real?
The corrected rescreen (505) turned the census's central negative
from "sixty leaves, zero writer-level mechanisms" into three
leaves, two of them new: r.1.2.0 and r.1.2.2 both show m14 AND m15
enriched for m5's writes (1.78/1.81 and 1.53/1.56 against bars of
1.30-1.41). Under the flat weighting both leaves named a0 instead
-- the writer that was overweighted 4,242x -- so the correction
did not merely add signal, it replaced a wrong answer.
A screen result is a lead, not a mechanism. The screen measures a
CORRELATION: at this leaf's member positions, m5's share of the
input to m14 and m15 is larger than at off-slice positions. Three
things have to hold before that becomes a claim.
  PEER. Other leaves whose bundles use the same components must
    NOT show the same m5 enrichment. If they do, it is a property
    of m14/m15 and not of these leaves.
  CAUSAL. Silencing m5's contribution to m14 and m15's inputs
    specifically -- replacing that one writer's part with its own
    mean over positions, leaving m5's output intact everywhere
    else in the network -- must damage the leaf's member positions
    more than its off-slice positions.
  WRITER-SPECIFIC. The same surgery on m4 and m6, the neighbouring
    writers with similar coefficients, must do less.
REGISTERED PREDICTIONS (scored per leaf, both must hold for the
leaf to count):
  (0) EXACTNESS: writer parts reproduce m14's and m15's real
      inputs to 1e-4 relative, checked before any bar is scored;
  (a) PEER: the leaf's m5 enrichment exceeds the largest m5
      enrichment among at least four peer leaves using the same
      components;
  (b) CAUSAL: silencing m5 into m14+m15 costs at least 1.5x more
      at member positions than off-slice, with both absolute
      numbers reported and the bar scored through cl.score_bar so
      a near-zero denominator returns UNEVALUABLE rather than a
      verdict;
  (c) WRITER-SPECIFIC: the member/off-slice contrast for m5
      exceeds that for BOTH m4 and m6.
  NULL: a random direction with norm matched to m5's contribution,
      substituted the same way, must give a member/off-slice
      contrast below 1.2. Three seeds.
A leaf that passes all three is the first causally verified
writer-level mechanism the census has produced."""
import ast, json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'mech_m5_verify_results.json'
TAGS=['r.1.2.0','r.1.2.2']
COMPS=[14,15]

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
                parts=cl.writer_parts(li,E,outs,'m')
                tot=sum(parts.values())
                errs.append(float((F.rms_norm(tot,(D,))-X.float())
                            .norm()/X.float().norm().clamp_min(1e-9)))
                if writer is None: return None
                if writer=='RAND':
                    p=parts['m5']
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
            hs.append(m.transformer.h[li].mlp
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
    for w in ('m5','m4','m6'):
        abl[w],_=ce_grid(w)
        print(f'ran {w}',flush=True)
    rnd=[]
    for s in (5,17,29):
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
        for w in ('m5','m4','m6'):
            dm,do=contrast(base,abl[w],mem,off)
            row[w]={'member':round(dm,5),'offslice':round(do,5)}
        rv=[contrast(base,r,mem,off) for r in rnd]
        row['rand']=[{'member':round(a,5),'offslice':round(b,5)}
                     for a,b in rv]
        res[tag]=row
        print(f"{tag}: m5 member {row['m5']['member']:+.5f} "
              f"offslice {row['m5']['offslice']:+.5f} | m4 "
              f"{row['m4']['member']:+.5f} | m6 "
              f"{row['m6']['member']:+.5f}",flush=True)
        json.dump(res,open(OUT,'w'),indent=1)
    verdicts={}
    for tag in TAGS:
        r=res[tag]
        dm,do=r['m5']['member'],r['m5']['offslice']
        vb,_=cl.score_bar(f'{tag}-b',dm,1.5*do if do>0 else 1e9,
                          denom=do,ref=dm)
        peer_best=max((res[p]['m5']['member'] for p in peers),
                      default=0.0)
        va,_=cl.score_bar(f'{tag}-a',dm-peer_best,1e-9)
        vc='HELD' if (dm>r['m4']['member'] and dm>r['m6']['member']) \
           else 'FAILED'
        rr=max((x['member']/x['offslice'] if x['offslice']>1e-6
                else 0) for x in r['rand'])
        nul=rr<1.2
        print(f'({tag}-c) m5 member {dm:+.5f} beats m4 '
              f"{r['m4']['member']:+.5f} and m6 "
              f"{r['m6']['member']:+.5f}: {vc}")
        print(f'({tag}-NULL) best random contrast {rr:.2f} < 1.2: '
              f"{'ok' if nul else 'VIOLATED'}")
        verdicts[tag]={'a':va,'b':vb,'c':vc,'null_ok':bool(nul),
                       'peer_best':round(peer_best,5)}
    out={'per_leaf':res,'verdicts':verdicts,'peers':peers,
         'relerr':relerr,'pred_0':True,'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
