"""ENRICHMENT PREDICTS -- GENERALIZATION -- 470: on r.3.0.2 the
mechanism table's enrichment ratio predicts causal ablation
selectivity at Spearman 0.842 across ten components (a14 2.43 ->
conc 5.76, a15 2.33 -> 5.40, a16 2.17 -> 8.56, down to a0 0.74 ->
1.68). That validates the swarm's central instrument on ONE leaf
-- and r.3.0.2 is the program's only leaf with a positive
mechanism table.
Does the ranking still work where the table says NOTHING is
enriched? On negative leaves every writer ratio sits near 1.0, so
either the tool retains ordering information below its own
threshold (valuable -- the negatives are still informative) or it
does not (also worth knowing, and it would mean a negative table
carries no signal at all).
Leaves: r.13.2.1 and r.5.3.1 (both ENRICHED_STABLE2 negative
everywhere), with r.3.0.2 rerun on a different component sample
as the positive control.
REGISTERED PREDICTIONS:
  (a) ORDERING SURVIVES: Spearman >= 0.5 on at least one of the
      two negative leaves;
  (b) FLATTER: the concentration range (max - min) on each
      negative leaf is smaller than r.3.0.2's 7.08;
  (c) POOLED: across all three leaves' components pooled,
      Spearman >= 0.5."""

import json, ast, time, torch
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'enrichment_generalize_results.json'
TAGS=['r.13.2.1','r.5.3.1','r.3.0.2']

def spearman(a,b):
    a=torch.tensor(a,dtype=torch.float)
    b=torch.tensor(b,dtype=torch.float)
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    return float(torch.corrcoef(torch.stack([ra,rb]))[0,1])

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    mus=cl.comp_means()
    MODS={f'a{li}':m.transformer.h[li].attn for li in range(18)}
    MODS.update({f'm{li}':m.transformer.h[li].mlp
                 for li in range(18)})
    def hooks(key):
        mu=mus[key].to(DEV); mod=MODS[key]
        if key[0]=='a':
            def fh(mo,i_,o_,mu=mu):
                y,v1=o_
                return (mu.expand_as(y).to(y.dtype),v1)
        else:
            def fh(mo,i_,o_,mu=mu):
                return mu.expand_as(o_).to(o_.dtype)
        return [mod.register_forward_hook(fh)]
    ALL={}
    for TAG in TAGS:
      tab=json.load(open(PT+f'leaf_mech/{TAG}.json'))['tables']
    # pool the enrichment ratios across the leaf's components
      ratio={}
      for comp,t in tab.items():
        for w,v in t['writers'].items():
            r=v.get('mean')
            if r is None: continue
            ratio[w]=max(ratio.get(w,0),r)
      ranked=sorted(ratio,key=ratio.get,reverse=True)
      picks=ranked[:4]+ranked[len(ranked)//2:len(ranked)//2+3] \
        +ranked[-2:]
      picks=[p for p in dict.fromkeys(picks) if p!='wte']
      print(f'{TAG}: {len(picks)} components {picks}',flush=True)
      res={}
      for key in picks:
        if key not in MODS: continue
        torch.cuda.empty_cache()
        d=cl.ce_sweep(hooks(key))-cl.base_ce()
        st=cl.sign_stats(TAG,d)
        res[key]={'enrichment':ratio.get(key),
                  'concentration':st['concentration']}
        print(f"  {key}: enr {ratio.get(key)} conc "
              f"{st['concentration']}",flush=True)
      ALL[TAG]=res
      json.dump(ALL,open(OUT,'w'),indent=1)
    def rho_of(res):
        have=[k for k in res if res[k]['enrichment'] is not None]
        if len(have)<4: return None
        return round(spearman([res[k]['enrichment'] for k in have],
                              [res[k]['concentration']
                               for k in have]),3)
    rhos={t:rho_of(ALL[t]) for t in ALL}
    rng={t:round(max(v['concentration'] for v in ALL[t].values())
                 -min(v['concentration'] for v in ALL[t].values()),2)
         for t in ALL}
    pooled_e=[v['enrichment'] for t in ALL for v in ALL[t].values()
              if v['enrichment'] is not None]
    pooled_c=[v['concentration'] for t in ALL
              for v in ALL[t].values() if v['enrichment'] is not None]
    prho=round(spearman(pooled_e,pooled_c),3)
    neg=[t for t in ('r.13.2.1','r.5.3.1') if t in rhos]
    pa=any(rhos[t] is not None and rhos[t]>=0.5 for t in neg)
    pb=all(rng[t]<7.08 for t in neg)
    pc=prho>=0.5
    out={'per_leaf':ALL,'spearman_per_leaf':rhos,
         'concentration_range':rng,'pooled_spearman':prho,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print(f'per-leaf spearman {rhos} | ranges {rng} | pooled {prho}')
    for nm,v in (('a','ordering survives on a negative leaf'),
                 ('b','negative leaves are flatter'),
                 ('c','pooled spearman >=0.5')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')
    return
    have=[k for k in res if res[k]['enrichment'] is not None]
    rho=spearman([res[k]['enrichment'] for k in have],
                 [res[k]['concentration'] for k in have]) \
        if len(have)>=4 else None
    top_e=sorted(have,key=lambda k:-res[k]['enrichment'])[:2]
    top_c=sorted(res,key=lambda k:-res[k]['concentration'])[:2]
    absent_c=[res[k]['concentration'] for k in res
              if res[k]['enrichment'] is None]
    top5_c=[res[k]['concentration'] for k in
            sorted(have,key=lambda k:-res[k]['enrichment'])[:5]]
    pa=(rho is not None and rho>=0.5)
    pb=('a14' in top_e and 'a14' in top_c)
    pc=(bool(absent_c) and
        sum(absent_c)/len(absent_c)
        < sum(top5_c)/max(len(top5_c),1))
    out={'components':res,'spearman':round(rho,3)
         if rho is not None else None,
         'top_by_enrichment':top_e,'top_by_concentration':top_c,
         'mean_conc_absent':round(sum(absent_c)/len(absent_c),3)
         if absent_c else None,
         'mean_conc_top5_enriched':round(
             sum(top5_c)/max(len(top5_c),1),3),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print(f'spearman(enrichment, concentration) = {out["spearman"]}')
    print(f'top by enrichment {top_e} | top by concentration {top_c}')
    for nm,v in (('a','the table predicts selectivity (rho>=0.5)'),
                 ('b','a14 tops both measures'),
                 ('c','absent components are less selective')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
