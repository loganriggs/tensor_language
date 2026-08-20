"""MECH MAP SPECIFICITY -- 472: the census-scale map (471) is
turning up positives that look adjacent-layer -- m14 into m15,
m15 into m17. The wave-3 reviewer catch (425) showed exactly this
shape can be a LAYER property: a14's enrichment into a15
reproduced on an unrelated leaf. Systematize that check over
every positive the map finds.
For each positive (leaf, component, writer): find peer leaves
whose machinery also includes that component, read the same
writer's ratio from their tables (computing them if absent), and
ask whether the positive leaf stands out.
A positive is LEAF-SPECIFIC if its min ratio exceeds the peers'
median ratio for that (component, writer) pair by >= 0.5;
otherwise it is a LAYER PROPERTY.
REGISTERED PREDICTIONS:
  (a) SOME ARE REAL: >= 30% of positive pairs are leaf-specific;
  (b) ADJACENCY IS SUSPECT: pairs with |layer(component) -
      layer(writer)| <= 2 are leaf-specific LESS often than
      distant pairs;
  (c) the full specificity table is written for the swarm to
      claim from."""
import json, os, sys, time, subprocess, statistics
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'mech_map_specificity_results.json'
SRC=PT+'mech_map_all_results.json'
MAXPEERS=4

def table(tag):
    f=PT+f'leaf_mech/{tag}.json'
    if not os.path.exists(f):
        subprocess.run([sys.executable,'-u',
                        PT+'leaf_input_decomp.py',tag],
                       cwd=PT,capture_output=True,text=True,
                       timeout=900)
    try: return json.load(open(f))
    except Exception: return None

def main():
    t0=time.time()
    if not os.path.exists(SRC):
        print('mech_map_all_results.json missing -- run 471 first')
        return
    src=json.load(open(SRC))
    leaves=src.get('leaves',src)
    pos=[(t,p['component'],p['writer'],p['min_ratio'])
         for t,v in leaves.items() for p in v.get('positives',[])]
    print(f'{len(pos)} positive pairs to check',flush=True)
    if not pos:
        json.dump({'note':'no positives found','pairs':[]},
                  open(OUT,'w'),indent=1)
        print('no positives; nothing to check'); return
    import census_lib as cl
    cl.use_state(PT+'census_state_diverse.pt')
    st=cl.state()
    def peers(comp,skip):
        out=[]
        for lf in st['leaves']:
            if lf['tag']==skip: continue
            if any(comp in str(p) for p in lf['top_probes']):
                out.append(lf['tag'])
            if len(out)>=MAXPEERS: break
        return out
    rows=[]
    for tag,comp,writer,mn in pos:
        pr=peers(comp,tag); vals=[]
        for p in pr:
            t=table(p)
            if not t: continue
            tb=t['tables'].get(comp)
            if not tb: continue
            w=tb['writers'].get(writer)
            if w and w.get('mean') is not None:
                vals.append(w['mean'])
        med=statistics.median(vals) if vals else None
        spec=(med is not None and (mn-med)>=0.5)
        lay=lambda x:int(x[1:])
        rows.append({'tag':tag,'component':comp,'writer':writer,
                     'min_ratio':mn,'peers':pr,
                     'peer_ratios':[round(v,3) for v in vals],
                     'peer_median':round(med,3) if med else None,
                     'leaf_specific':bool(spec),
                     'gap':round(lay(comp)-lay(writer),1)})
        print(f"{tag} {writer}->{comp}: {mn} vs peers "
              f"{rows[-1]['peer_ratios']} -> "
              f"{'SPECIFIC' if spec else 'layer property'}",
              flush=True)
        json.dump({'pairs':rows},open(OUT,'w'),indent=1)
    scored=[r for r in rows if r['peer_median'] is not None]
    nspec=sum(r['leaf_specific'] for r in scored)
    frac=nspec/max(len(scored),1)
    adj=[r for r in scored if abs(r['gap'])<=2]
    dist=[r for r in scored if abs(r['gap'])>2]
    fa=sum(r['leaf_specific'] for r in adj)/max(len(adj),1)
    fd=sum(r['leaf_specific'] for r in dist)/max(len(dist),1)
    pa=frac>=0.30
    pb=(len(adj)>0 and len(dist)>0 and fa<fd)
    out={'pairs':rows,'n_scored':len(scored),
         'n_leaf_specific':nspec,'frac_leaf_specific':round(frac,3),
         'frac_specific_adjacent':round(fa,3),
         'frac_specific_distant':round(fd,3),
         'n_adjacent':len(adj),'n_distant':len(dist),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':True,
         'runtime_s':time.time()-t0}
    print(f'{nspec}/{len(scored)} leaf-specific ({frac:.1%}); '
          f'adjacent {fa:.1%} vs distant {fd:.1%}')
    for nm,v in (('a','>=30% of positives are leaf-specific'),
                 ('b','adjacent pairs are specific less often'),
                 ('c','table written')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
