"""LEAF INPUT DECOMP -- the swarm's generic MECHANISM tool
(SOP v3 step 3M). Usage:  python leaf_input_decomp.py <tag>

For each component named by a leaf's probe bundle (its machinery),
decompose the residual ENTERING that component into exact writer
contributions (wte + every component write below it; the block
lambda mix and the rms scale are per-position scalars, so the
split is exact), then contrast MEMBER positions against OFF-SLICE
positions in the same rows. What comes out is a mechanism lead in
named parts: "this leaf's machinery fires where writer X's
contribution to component C's input is enriched".

Computed bars (printed, so the agent cannot certify junk):
  ENRICHED: top writer's member-vs-offslice share ratio >= 1.3
  NOT-NULL: that ratio exceeds the row-matched random-position
            null's top ratio
Writes leaf_mech/<tag>.json with the per-component tables; the
agent quotes these numbers in the record's mechanism field.
"""
import json, os, sys, time, ast, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
MAXROWS=24

@torch.no_grad()
def main(tag):
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    lf=cl.leaf(tag)
    probes=[ast.literal_eval(p) if isinstance(p,str) else p
            for p in lf['top_probes']]
    keys=[]
    for p in probes:
        k=p[1] if p[0] in ('comp','pca') else f'a{p[1]}'
        if k not in keys: keys.append(k)
    mem=lf['member']; sl=lf['slice']
    NF=cl.nflat()
    memm=torch.zeros(NF,dtype=torch.bool); memm[mem]=True
    slm=torch.zeros(NF,dtype=torch.bool); slm[sl]=True
    allrows=(mem//256).unique()
    rows=cl.rows()
    # --seeds N widens the bootstrap (wave-6 follow-up: the
    # near-miss and underpowered-with-a-hint negatives need more
    # draws before the negative can be called). Default 5 keeps
    # every earlier record reproducible.
    _ns=5
    if '--seeds' in sys.argv:
        _ns=int(sys.argv[sys.argv.index('--seeds')+1])
    SEEDS=[5,17,29,41,53,67,79,91,103,115,
           127,139,151,163,175,187,199,211,223,235][:_ns]
    def draw(seed):
        g=torch.Generator().manual_seed(seed)
        if len(allrows)<=MAXROWS: return allrows
        return allrows[torch.randperm(len(allrows),generator=g)
                       [:MAXROWS]].sort().values
    print(f'{tag}: machinery {keys} | {min(len(allrows),MAXROWS)} '
          f'rows x {len(SEEDS)} bootstrap draws',flush=True)
    tables={}
    for key in keys:
      per_seed={}
      for _si,_seed in enumerate(SEEDS):
        rr=draw(_seed)
        li=int(key[1:])
        WR=['wte']+[f'{k}{l}' for l in range(li)
                    for k in ('a','m')]
        if key[0]=='m': WR.append(f'a{li}')   # attn of same block
        share={'mem':{w:0.0 for w in WR},'off':{w:0.0 for w in WR},
               'null':{w:0.0 for w in WR}}
        cnt={'mem':0,'off':0,'null':0}
        for i in range(0,len(rr),4):
            rid=rr[i:i+4]
            bb=rows[rid,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); B=len(rid)
            outs={}; hs=[]
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
            x=E.to(m.transformer.wte.weight.dtype); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            for h in hs: h.remove()
            lam=m.transformer.h[li].lambdas.detach().float()
            parts={w:((lam[0]+lam[1])*E if w=='wte'
                      else lam[0]*outs[w]) for w in WR
                   if w!=f'a{li}'}
            if f'a{li}' in WR: parts[f'a{li}']=outs[f'a{li}']
            tot=sum(parts.values())
            tn=(tot*tot).sum(-1).clamp_min(1e-9)
            frac={w:((p*tot).sum(-1)/tn) for w,p in parts.items()}
            gi=(rid[:,None].to(DEV)*256
                +torch.arange(T,device=DEV)[None,:])
            mmask=memm.to(DEV)[gi]
            omask=(~slm.to(DEV)[gi])
            npick=torch.rand(mmask.shape,generator=None,
                             device=DEV)<(mmask.float().mean()+1e-6)
            for nm,msk in (('mem',mmask),('off',omask),
                           ('null',npick)):
                c=int(msk.sum())
                if c==0: continue
                cnt[nm]+=c
                for w,fv in frac.items():
                    share[nm][w]+=float(fv[msk].sum())
        tbl={}
        for w in share['mem']:
            mv=share['mem'][w]/max(cnt['mem'],1)
            ov=share['off'][w]/max(cnt['off'],1)
            nv=share['null'][w]/max(cnt['null'],1)
            tbl[w]={'member':round(mv,4),'offslice':round(ov,4),
                    'null':round(nv,4),
                    'ratio':round(mv/ov if abs(ov)>1e-4
                                  else float('nan'),3)}
        nullratio=max((v['null']/v['offslice']
                       if abs(v['offslice'])>1e-4 else 0)
                      for v in tbl.values())
        per_seed[_seed]={'tbl':tbl,'null':nullratio}
      # --- bootstrap aggregation (2026-08-20, wave-3 reviewer
      # catch: a single hardcoded row subsample made a 1.46 ratio
      # out of a writer whose resampled range is 1.00-1.07) ---
      wl=list(per_seed[SEEDS[0]]['tbl'])
      agg={}
      for w in wl:
          vals=[per_seed[s]['tbl'][w]['ratio'] for s in SEEDS
                if per_seed[s]['tbl'][w]['ratio']
                ==per_seed[s]['tbl'][w]['ratio']]
          if not vals: continue
          agg[w]={'mean':round(sum(vals)/len(vals),3),
                  'min':round(min(vals),3),'max':round(max(vals),3),
                  'seed5':per_seed[SEEDS[0]]['tbl'][w]['ratio'],
                  'member':per_seed[SEEDS[0]]['tbl'][w]['member'],
                  'offslice':per_seed[SEEDS[0]]['tbl'][w]['offslice']}
      top=sorted(agg.items(),key=lambda kv:-kv[1]['mean'])[:5]
      nulls=[per_seed[s]['null'] for s in SEEDS]
      nullratio=max(nulls)
      # 2026-08-20 (wave-4 reviewer catch): max-of-5 is an
      # extreme-value statistic -- on some leaves it lands at
      # 1.333 against a 1.3 bar, i.e. 2.5% headroom, so the gate
      # could not distinguish a weak true effect from null noise.
      # ENRICHED_STABLE2 gates on the null's mean + 2 sd instead.
      nmean=sum(nulls)/len(nulls)
      nsd=(sum((x-nmean)**2 for x in nulls)/max(len(nulls)-1,1))**0.5
      thresh=max(1.3,nmean+2*nsd)
      enriched=top[0][1]['seed5'] if top else 0
      stable=bool(top and top[0][1]['min']>=1.3
                  and top[0][1]['min']>nullratio)
      stable2=bool(top and top[0][1]['min']>thresh)
      tables[key]={'writers':agg,'top':top,
                   'top_ratio':enriched,
                   'top_ratio_mean':top[0][1]['mean'] if top else 0,
                   'top_ratio_min':top[0][1]['min'] if top else 0,
                   'top_ratio_max':top[0][1]['max'] if top else 0,
                   'null_top_ratio':round(nullratio,3),
                   'bootstrap_seeds':SEEDS,
                   'ENRICHED':bool(enriched>=1.3),
                   'BEATS_NULL':bool(enriched>nullratio),
                   'ENRICHED_STABLE':stable,
                   'null_mean':round(nmean,3),
                   'null_sd':round(nsd,3),
                   'threshold_v2':round(thresh,3),
                   'ENRICHED_STABLE2':stable2,
                   'headroom':round(top[0][1]['min']-thresh,3)
                                if top else None,
                   # 2026-08-20 (wave-6 reviewer catch, r.4.1.1):
                   # a negative is only informative if a real
                   # effect could have cleared the bar. FIRST
                   # VERSION USED thresh MINUS THE NULL'S WORST
                   # DRAW AND WAS WRONG: max-of-N grows with N, so
                   # the statistic shrank mechanically when the
                   # bootstrap was widened (writeup 500). The
                   # N-stable question is how high the bar sits:
                   # thresh IS the minimum enrichment this test can
                   # register, and it is only pushed above 1.3 when
                   # the null's own spread is large.
                   'min_detectable_enrichment':round(thresh,3),
                   'bar_source':('fixed_1.3' if thresh<=1.3001
                                 else 'null_driven'),
                   'negative_power':(None if stable2 else
                       ('UNDERPOWERED' if thresh>1.35
                        else 'DECISIVE')),
                   'NEAR_MISS':bool(top and not stable2 and
                       thresh-top[0][1]['min']<0.10)}
      _s=', '.join(f"{w} {v['mean']} [{v['min']}-{v['max']}]"
                   for w,v in top[:3])
      print(f"  {key}: top {_s} | ENRICHED_STABLE2={stable2}"
            f" (thresh {thresh:.3f} = max(1.3, null {nmean:.3f}"
            f"+2sd {nsd:.3f}); v1 stable={stable}, single-draw"
            f" ENRICHED={tables[key]['ENRICHED']})",flush=True)
      if not stable2:
          print(f"    negative power: "
                f"{tables[key]['negative_power']} "
                f"(needs {tables[key]['min_detectable_enrichment']}x"
                f" enrichment to register, bar "
                f"{tables[key]['bar_source']}"
                f"{'; NEAR MISS -- widen the bootstrap'
                   if tables[key]['NEAR_MISS'] else ''})",
                flush=True)
    os.makedirs(PT+'leaf_mech',exist_ok=True)
    out={'tag':tag,'machinery':keys,'tables':tables,
         'runtime_s':time.time()-t0}
    # a widened bootstrap writes beside the record, never over it
    _sfx='' if len(SEEDS)==5 else f'_s{len(SEEDS)}'
    out['n_seeds']=len(SEEDS)
    json.dump(out,open(PT+f'leaf_mech/{tag}{_sfx}.json','w'),indent=1)
    print(f'wrote leaf_mech/{tag}{_sfx}.json '
          f'({out["runtime_s"]:.0f}s)')

def baseline(key,tag,k=3):
    """Cross-leaf baseline (added 2026-08-20 from a wave-3
    reviewer catch: a14 dominates a15's input for unrelated leaves
    too, so a ratio against 1.0 overstates specificity). Returns
    the same component's top-writer ratios for k other leaves whose
    machinery includes this component."""
    import random
    st=cl.state()
    cand=[lf['tag'] for lf in st['leaves'] if lf['tag']!=tag and
          any(key in str(p) for p in lf['top_probes'])]
    random.Random(7).shuffle(cand)
    return cand[:k]

if __name__=='__main__':
    main(sys.argv[1])
    if '--baseline' in sys.argv:
        tag=sys.argv[1]
        seen=json.load(open(PT+f'leaf_mech/{tag}.json'))
        for key in seen['machinery']:
            peers=baseline(key,tag)
            print(f'-- baseline for {key}: peers {peers} '
                  f'(run leaf_input_decomp.py on each and compare '
                  f'{key}\'s top-writer ratio; a ratio that '
                  f'reproduces on unrelated leaves is a LAYER '
                  f'property, not this circuit\'s mechanism)')
