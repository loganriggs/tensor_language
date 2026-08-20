"""MECH RESCREEN -- re-run the whole mechanism screen with the
corrected writer decomposition (writeup 503).
Every leaf_input_decomp result in this program was computed with a
flat lambda weighting: writer j's output was multiplied by the
current block's lam0, when the exact coefficient is the PRODUCT of
lam0 over every block between j and the target. Because block 1
has lam0 = 0.0127, layer-0 writers were overweighted by a factor
of 4,242 and wte was underweighted 7x. The flat version
reconstructs the layer-12 attention input to 68% relative error;
the corrected one to 1.2e-7, and the tool now refuses to run
unless the reconstruction checks out.
That screen produced the program's central negative -- sixty
leaves, zero writer-level mechanisms -- so the negative has to be
recomputed before it can be believed. Every stored 5-seed record
is re-run and its verdict compared. Old files are preserved in
leaf_mech_flatweight_backup/.
REGISTERED PREDICTIONS:
  (a) THE BUG WAS NOT INERT: at least one leaf changes its
      ENRICHED_STABLE2 verdict, or at least five change their top
      writer. A correction of four orders of magnitude that
      changes nothing would mean the statistic never depended on
      the weighting at all, which would itself need explaining;
  (b) REGRESSION: r.3.0.2, the one confirmed positive, stays
      positive (verified by hand before this run: a15/a16/a17 all
      True, top writer a14 at 2.17-2.21);
  (c) EXACTNESS EVERYWHERE: every tag passes the reconstruction
      check. Any failure is reported by tag rather than skipped.
  The new count of positives is reported either way; if it is
  still zero, the census negative is confirmed on a correct
  decomposition rather than an incorrect one."""
import json, glob, os, subprocess, sys, time
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'mech_rescreen_results.json'
BK=PT+'leaf_mech_flatweight_backup/'

def main():
    t0=time.time()
    tags=[]
    for f in sorted(glob.glob(BK+'*.json')):
        d=json.load(open(f))
        if d.get('n_seeds',5)!=5: continue
        tags.append(d['tag'])
    print(f'{len(tags)} tags to rescreen',flush=True)
    res={}; fails=[]
    for ti,tag in enumerate(tags):
        p=subprocess.run([sys.executable,PT+'leaf_input_decomp.py',
                          tag],capture_output=True,text=True,cwd=PT)
        old=json.load(open(BK+f'{tag}.json'))
        if p.returncode!=0:
            fails.append({'tag':tag,'err':p.stdout[-400:]+p.stderr[-400:]})
            print(f'[{ti+1}/{len(tags)}] {tag} FAILED',flush=True)
            continue
        new=json.load(open(PT+f'leaf_mech/{tag}.json'))
        row={}
        for k,nt in new['tables'].items():
            ot=old['tables'].get(k,{})
            row[k]={'old_verdict':ot.get('ENRICHED_STABLE2'),
                    'new_verdict':nt['ENRICHED_STABLE2'],
                    'old_top':(ot.get('top') or [[None]])[0][0],
                    'new_top':(nt.get('top') or [[None]])[0][0],
                    'old_ratio':ot.get('top_ratio_min'),
                    'new_ratio':nt.get('top_ratio_min'),
                    'new_bar':nt.get('min_detectable_enrichment'),
                    'new_power':nt.get('negative_power')}
        res[tag]=row
        ch=[k for k,v in row.items()
            if v['old_verdict']!=v['new_verdict']]
        tc=[k for k,v in row.items() if v['old_top']!=v['new_top']]
        print(f'[{ti+1}/{len(tags)}] {tag}: verdict changes '
              f'{ch or "none"} | top-writer changes {len(tc)}/'
              f'{len(row)}',flush=True)
        json.dump(res,open(OUT,'w'),indent=1)
    flips=[(t,k) for t,r in res.items() for k,v in r.items()
           if v['old_verdict']!=v['new_verdict']]
    topch=[(t,k) for t,r in res.items() for k,v in r.items()
           if v['old_top']!=v['new_top']]
    pos=[(t,k) for t,r in res.items() for k,v in r.items()
         if v['new_verdict']]
    oldpos=[(t,k) for t,r in res.items() for k,v in r.items()
            if v['old_verdict']]
    r302=res.get('r.3.0.2',{})
    pa=bool(flips) or len(topch)>=5
    pb=bool(r302) and all(v['new_verdict'] for v in r302.values())
    pc=not fails
    out={'per_tag':res,'n_tags':len(tags),
         'verdict_flips':flips,'n_top_writer_changes':len(topch),
         'positives_new':pos,'positives_old':oldpos,
         'n_positives_new':len(pos),'n_positives_old':len(oldpos),
         'failures':fails,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc),'runtime_s':time.time()-t0}
    print(f'\nverdict flips: {flips or "none"}')
    print(f'top-writer changes: {len(topch)} of '
          f'{sum(len(r) for r in res.values())} component tests')
    print(f'positives: {len(oldpos)} before -> {len(pos)} after')
    for nm,v in (('a','the correction changes something'),
                 ('b','r.3.0.2 stays positive'),
                 ('c','every tag passes the exactness check')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    if fails: print('failures:',[f['tag'] for f in fails])
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
