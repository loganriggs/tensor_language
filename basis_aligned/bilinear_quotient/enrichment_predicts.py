"""ENRICHMENT PREDICTS SELECTIVITY? -- 469: ablating a14, the
writer the mechanism table flags as enriched into r.3.0.2's
machinery, damages that leaf's members at concentration 5.76 --
nearly the leaf's own bundle (6.1), so the enrichment IS causal
(468a). But the adjacent-depth control a13 also reaches 3.4, so
selectivity is not unique to the enriched writer (468b failed),
and a14 plus the bundle turn out to be ADDITIVE (0.892 joint
against 0.918 summed) rather than one pathway (468c failed).
That raises the question the whole swarm tool depends on: does
input-writer ENRICHMENT predict causal SELECTIVITY at all, or is
selectivity just a property of depth and size? Test it as a
correlation across many components rather than one pair.
For r.3.0.2, take twelve components spanning the writer table --
the enriched writer, mid-table writers, bottom-table writers, and
components absent from it -- mean-ablate each, and correlate the
resulting member concentration against the enrichment ratio the
table assigns.
REGISTERED PREDICTIONS:
  (a) THE TABLE PREDICTS: Spearman correlation between enrichment
      ratio and ablation concentration >= 0.5 across the twelve;
  (b) a14 ranks in the top two on both measures;
  (c) CONTROL: components absent from the writer table average
      lower concentration than the top-five enriched ones."""
import json, ast, time, torch
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'enrichment_predicts_results.json'
TAG='r.3.0.2'

def spearman(a,b):
    a=torch.tensor(a,dtype=torch.float)
    b=torch.tensor(b,dtype=torch.float)
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    return float(torch.corrcoef(torch.stack([ra,rb]))[0,1])

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    tab=json.load(open(PT+f'leaf_mech/{TAG}.json'))['tables']
    # pool the enrichment ratios across the leaf's components
    ratio={}
    for comp,t in tab.items():
        for w,v in t['writers'].items():
            r=v.get('mean')
            if r is None: continue
            ratio[w]=max(ratio.get(w,0),r)
    ranked=sorted(ratio,key=ratio.get,reverse=True)
    picks=ranked[:5]+ranked[len(ranked)//2:len(ranked)//2+3] \
        +ranked[-2:]
    absent=[k for k in ('m14','a11') if k not in ratio]
    picks=[p for p in dict.fromkeys(picks) if p!='wte']+absent
    print(f'{len(picks)} components: {picks}',flush=True)
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
    res={}
    for key in picks:
        if key not in MODS: continue
        torch.cuda.empty_cache()
        d=cl.ce_sweep(hooks(key))-cl.base_ce()
        s=cl.sign_stats(TAG,d)
        res[key]={'enrichment':ratio.get(key),
                  'concentration':s['concentration'],
                  'abs_dce_members':s['abs_dce_members']}
        print(f"{key}: enrichment {ratio.get(key)} conc "
              f"{s['concentration']}",flush=True)
        json.dump(res,open(OUT,'w'),indent=1)
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
