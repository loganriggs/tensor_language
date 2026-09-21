"""Score preregistered positional hypothesis and distinguish error from target variance."""
import json, math, hashlib
from pathlib import Path
import torch
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2)
 source=P/'POSITION_TRANSFER_V1.json'; data=json.loads(source.read_text())
 counts={}
 for panel in (1,2):
  maps=torch.load(P/f'FRONTIER_FRESH_DONORS_V{panel}.pt',weights_only=True)
  for domain,mapping in maps.items():
   ids=(mapping>=0).nonzero().flatten();positions=ids%240+16
   for band,(lo,hi) in {'early':(16,64),'middle':(64,128),'late':(128,256)}.items():
    counts[panel,domain,band]=int(((positions>=lo)&(positions<hi)).sum())
 rows=[]; detail=[]
 for domain in ('fineweb','stdlib'):
  for candidate in ('graph_generated','cartesian_generated','separate_generated','isotropic_generated'):
   bands={}
   for band in ('early','middle','late'):
    cells=[c for c in data['summaries'] if c['domain']==domain and c['family']=='natural' and c['cohort']=='all@'+band]
    assert len(cells)==2
    bands[band]=math.sqrt(sum(c['candidates'][candidate]['relative_error'][2]**2 for c in cells)/2)
    for c in cells:
     v=c['candidates'][candidate];n=counts[c['panel'],domain,band];e=v['error_energy'][2];r=v['relative_error'][2]
     assert n>0 and r>0
     detail.append(dict(panel=c['panel'],domain=domain,candidate=candidate,band=band,n=n,relative_error=r,error_rms=math.sqrt(e/n),target_centered_rms=math.sqrt(e/(r*r*n))))
   ratio=bands['late']/bands['early']
   rows.append(dict(domain=domain,candidate=candidate,relative_error_rms_across_panels=bands,late_over_early=ratio,registered_growth_pass=ratio>=1.25))
 required=[r for r in rows if r['candidate'] in ('graph_generated','cartesian_generated')]
 # Independent accounting control: target normalization must be shared by all candidates.
 target_groups={}
 for r in detail:
  key=(r['panel'],r['domain'],r['band']);target_groups.setdefault(key,[]).append(r['target_centered_rms'])
 denominator_replay=max((max(v)-min(v))/max(v) for v in target_groups.values())
 assert denominator_replay<1e-10
 early_continuation=[dict(panel=c['panel'],domain=c['domain'],candidate=k,relative_error=v['relative_error'][2]) for c in data['summaries'] if c['family']=='natural' and c['cohort']=='continuation@early' for k,v in c['candidates'].items() if k in ('graph_generated','cartesian_generated')]
 out=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),prefix_replay_pass=max(data['prefix_replay_checks'])<1e-4,primary_hypothesis_pass=all(r['registered_growth_pass'] for r in required),rows=rows,absolute_error_and_target_variance=detail,shared_denominator_replay=denominator_replay,early_continuation=early_continuation,scope='Opened panels, component 3 natural effects. RMS of panel relative errors is the registered statistic; per-panel error RMS uses exact valid donor-site counts. Target centered RMS is computed within each band, not globally.')
 (P/'POSITION_TRANSFER_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(dict(primary_hypothesis_pass=out['primary_hypothesis_pass'],denominator_replay=denominator_replay,early_continuation=early_continuation),indent=2))
 for domain in ('fineweb','stdlib'):
  for candidate in ('graph_generated','cartesian_generated'):
   for panel in (1,2):
    rr={r['band']:r for r in detail if (r['domain'],r['candidate'],r['panel'])==(domain,candidate,panel)}
    print(domain,candidate,panel,'late/early error RMS',rr['late']['error_rms']/rr['early']['error_rms'],'target RMS',rr['late']['target_centered_rms']/rr['early']['target_centered_rms'])
if __name__=='__main__':main()
