"""BUNDLE SPLIT -- trace r.0.0.1's machinery bundle by bundle (user
questions: what is the mechanism at the improving examples? is the
circuit a composition -- subset or complement -- of simpler parts?
and are sign-mixed circuits in TENSION with siblings?).
Ablate each of the leaf's 4 probe bundles SINGLY (+ m0-pair, m3-pair,
joint), per-position dCE each time.
REGISTERED PREDICTIONS:
  (a) DISSOCIATION: the 4 single-bundle member-damage profiles are
      not interchangeable -- min pairwise Pearson r over members
      < 0.5;
  (b) WING SPLIT: the positive wing and negative wing are governed
      by different machinery -- the bundle with max |mean dCE| on
      the pos wing != the bundle with max |mean dCE| on the neg
      wing (the composition structure the user hypothesized);
  (c) TENSION: under the joint ablation, >=1 OTHER census leaf's
      member set improves by <= -0.3 mean dCE (first measured
      tension edge; recorded on both circuit records);
  (d) the Westminster-Abbey example (improves +(-6.8) jointly) is
      attributable: some single bundle accounts for >=60% of its
      joint improvement (pure report if not).
"""
import json, time, torch
import census_lib as cl
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'bundle_split_results.json'
TAG='r.0.0.1'

def main():
    t0=time.time()
    lf=cl.leaf(TAG); probes=lf['top_probes']
    mem=lf['member']; msc=cl.member_scores(TAG)
    bv=cl.base_ce()
    runs={}
    for i,p in enumerate(probes):
        runs[f'b{i}']=cl.ce_sweep(cl.proj_hooks([p]))-bv
        print(f'bundle {i} {p} done',flush=True)
    runs['m0_pair']=cl.ce_sweep(cl.proj_hooks([probes[0],probes[3]]))-bv
    runs['m3_pair']=cl.ce_sweep(cl.proj_hooks([probes[1],probes[2]]))-bv
    runs['joint']=cl.ce_sweep(cl.proj_hooks(probes))-bv
    print('pairs+joint done',flush=True)
    prof={k:v[mem] for k,v in runs.items()}
    B=[f'b{i}' for i in range(4)]
    cors=[]
    for i in range(4):
        for j in range(i+1,4):
            r=float(torch.corrcoef(torch.stack([prof[B[i]],
                                                prof[B[j]]]))[0,1])
            cors.append(round(r,3))
    pa=min(cors)<0.5
    posw=msc>0; negw=msc<0
    wp={k:round(float(prof[k][posw].mean()),3) for k in B}
    wn={k:round(float(prof[k][negw].mean()),3) for k in B}
    topp=max(B,key=lambda k:abs(wp[k])); topn=max(B,key=lambda k:abs(wn[k]))
    pb=topp!=topn
    # tension scan: every other leaf under the joint ablation
    dj=runs['joint']; tension=[]
    for o in cl.state()['leaves']:
        if o['tag']==TAG: continue
        md=float(dj[o['member']].mean())
        if md<=-0.3: tension.append({'tag':o['tag'],
                                     'value':round(md,3),
                                     'n':o['n_members']})
    pc=len(tension)>=1
    gi=55*256+176   # Westminster Abbey example from 344
    abbey={k:round(float(runs[k][gi]),3) for k in runs}
    jd=abbey['joint']
    share=max((abs(abbey[k])/max(abs(jd),1e-4)) for k in B
              if abbey[k]*jd>0) if jd else 0
    pd_=share>=0.6
    out={'tag':TAG,'bundles':[str(p) for p in probes],
         'pairwise_corr_members':cors,
         'wing_pos_by_bundle':wp,'wing_neg_by_bundle':wn,
         'top_pos_bundle':topp,'top_neg_bundle':topn,
         'member_mean_by_run':{k:round(float(prof[k].mean()),3)
                               for k in runs},
         'tension_edges':tension,'abbey_gi':gi,'abbey':abbey,
         'abbey_top_share':round(share,3),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'pred_d':bool(pd_)}
    print('pairwise corr:',cors)
    print('pos wing:',wp,'-> top',topp)
    print('neg wing:',wn,'-> top',topn)
    print('tension edges:',tension[:8],f'({len(tension)} total)')
    print('abbey per-run:',abbey)
    print(f"(a) dissociation min r<0.5: {'HELD' if pa else 'FAILED'}")
    print(f"(b) wings governed by different bundles: {'HELD' if pb else 'FAILED'}")
    print(f"(c) >=1 tension edge <=-0.3: {'HELD' if pc else 'FAILED'}")
    print(f"(d) abbey attributable >=60%: {'HELD' if pd_ else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
