"""POWER RECHECK -- 497: are the census's mechanism negatives
real absences, or did the bar sit on top of the null?
A wave-6 reviewer noticed that on r.4.1.1 the ENRICHED_STABLE2
threshold (1.305) sat only 0.055 above the null's own worst draw
(1.25), so noise alone nearly cleared the bar and a real-but-weak
writer could not have been distinguished from nothing. A negative
under those conditions is uninformative.
Auditing all 148 stored component tests retroactively (the stored
records already carry both numbers, so no GPU was needed): 142 are
negatives, and 43 of them -- 30% -- have the bar within 0.10 of
the null ceiling. That sounds fatal for the census's zero, but the
observed signal matters too: in 33 of the 43 the top writer's
ratio is below 1.05, i.e. no enrichment at all, so no amount of
power would have changed the call. Only FOUR negatives are both
underpowered AND showing a hint:
  r.1.1.3 m15  signal 1.460  separation 0.064
  r.3.3.2 a17  signal 1.500  separation 0.047
  r.3.2.3 a17  signal 1.417  separation 0.077
  r.1.2.0 m14  signal 1.223  separation 0.075
(r.1.2.0's m14->m15 pair is independently the one that passed the
leaf-specificity control in mech_map_specificity at 2.277 against
peers 0.934/1.167/0.691, which makes it the most interesting of
the four.)
Re-run those four leaves with a 20-seed bootstrap instead of 5.
Widening shrinks the null's sd, which lowers the threshold, and
shrinks the signal's own spread, which raises its bootstrap
minimum -- both help, so the run needs a control against
manufacturing positives: four DECISIVE negatives (signal <= 0.99,
separation <= 0.05) get the same treatment. If widening flips
those, it is not measuring anything.
REGISTERED PREDICTIONS:
  (a) POWER WAS THE PROBLEM: at least one of the four
      underpowered-with-a-hint components reaches
      ENRICHED_STABLE2=True at 20 seeds;
  (b) THE WIDENING WORKS: all four reach a null-bar separation
      above 0.10, i.e. the bar is no longer sitting on the null;
  (c) CONTROL: none of the four decisive negatives flips to True.
      A flip there voids (a).
Whatever happens, the census headline is amended honestly: with 33
of 43 underpowered negatives showing no enrichment whatever, the
"zero writer-level mechanisms in 60 leaves" claim stands or falls
on these four tests, not on all 43."""
import json, subprocess, sys, time
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'power_recheck_results.json'
HINT=[('r.1.1.3','m15'),('r.3.3.2','a17'),
      ('r.3.2.3','a17'),('r.1.2.0','m14')]
CTRL=[('r.0.0.1','a4'),('r.4.0.1','a9'),
      ('r.2.0.3','a8'),('r.9.3.1','a3')]
NS=20

def run(tag):
    p=subprocess.run([sys.executable,PT+'leaf_input_decomp.py',
                      tag,'--seeds',str(NS)],
                     capture_output=True,text=True,cwd=PT)
    if p.returncode!=0:
        print(f'{tag} FAILED\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}',
              flush=True)
        return None
    try:
        return json.load(open(PT+f'leaf_mech/{tag}_s{NS}.json'))
    except Exception as e:
        print(f'{tag}: no output ({e})',flush=True); return None

def main():
    t0=time.time(); res={}
    for grp,pairs in (('hint',HINT),('control',CTRL)):
        for tag,comp in pairs:
            d=run(tag)
            if d is None: continue
            t=d['tables'].get(comp)
            if t is None:
                print(f'{tag}: component {comp} not in machinery '
                      f'{list(d["tables"])}',flush=True); continue
            res[f'{tag}:{comp}']={
                'group':grp,'n_seeds':NS,
                'signal_min':t['top_ratio_min'],
                'signal_mean':t['top_ratio_mean'],
                'top_writer':t['top'][0][0] if t['top'] else None,
                'threshold':t['threshold_v2'],
                'null_top':t['null_top_ratio'],
                'separation':t['null_bar_separation'],
                'STABLE2':t['ENRICHED_STABLE2']}
            r=res[f'{tag}:{comp}']
            print(f"[{grp}] {tag} {comp}: writer {r['top_writer']} "
                  f"signal {r['signal_min']} vs bar {r['threshold']}"
                  f" | sep {r['separation']} | "
                  f"STABLE2={r['STABLE2']}",flush=True)
            json.dump(res,open(OUT,'w'),indent=1)
    H=[v for v in res.values() if v['group']=='hint']
    C=[v for v in res.values() if v['group']=='control']
    pa=any(v['STABLE2'] for v in H)
    pb=bool(H) and all(v['separation']>0.10 for v in H)
    pc=not any(v['STABLE2'] for v in C)
    out={'components':res,'n_seeds':NS,
         'n_hint':len(H),'n_control':len(C),
         'flipped':[k for k,v in res.items()
                    if v['group']=='hint' and v['STABLE2']],
         'control_flips':[k for k,v in res.items()
                          if v['group']=='control' and v['STABLE2']],
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    for nm,v in (('a','a hinted negative flips at 20 seeds'),
                 ('b','all four separations exceed 0.10'),
                 ('c','CONTROL: no decisive negative flips')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    if out['control_flips']:
        print(f"*** control flipped {out['control_flips']} -- "
              f"widening manufactures positives, (a) is VOID ***")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
