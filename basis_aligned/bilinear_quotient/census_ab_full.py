"""CENSUS A/B FULL -- run the 393 A/B machinery-replication gate
over ALL eligible leaves of the diverse tree (the sampled run
covered 45; at 149s that projects ~17min for 311). Output is the
swarm's production shortlist: leaves whose damage profile
replicates across corpus halves are certifiable under the 381
identity rule; the rest are flagged.
REGISTERED PREDICTIONS:
  (a) full-tree replication rate within 10 points of the sampled
      77%;
  (b) >=120 leaves certified at cos >= 0.7 (a real production
      pool for the swarm);
  (c) depth>=2 rate confirmed < 70% (the sampled 56% was not
      sampling noise)."""
import census_ab_replication as rep
rep.NLEAF_PER_DEPTH=999
rep.OUT=rep.PT+'census_ab_full_results.json'

if __name__=='__main__':
    rep.main()
    import json
    d=json.load(open(rep.OUT))
    ncert=sum(1 for r in d['leaves'] if r['cos']>=0.7)
    pa=abs(d['replication_rate']-0.77)<=0.10
    pb=ncert>=120
    pc=d['rate_depth_ge2']<0.70
    d['n_certified']=ncert
    d['pred_a_full']=bool(pa); d['pred_b_full']=bool(pb)
    d['pred_c_full']=bool(pc)
    json.dump(d,open(rep.OUT,'w'),indent=1)
    print(f'certified {ncert} leaves')
    for nm,v in (('a','rate within 10pts of 0.77'),
                 ('b','>=120 certified'),('c','depth>=2 <70%')):
        held={'a':pa,'b':pb,'c':pc}[nm]
        print(f"({nm}) {v}: {'HELD' if held else 'FAILED'}")
