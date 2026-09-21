"""Frozen candidate comparison across all 32 historical chunks; not fresh validation."""
import json, hashlib
from pathlib import Path
import torch
from pairwise_reader_graph import source_reads
from read_error_terms import baseline_reads
from generated_residual_interface import components
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False)
 d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True)
 z,h=d['z'],d['h'];assert len(z)==2048
 s=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt()
 A=torch.stack([p['a'] for p in d['pairs']],-1);alpha=torch.stack([p['alpha'] for p in d['pairs']]);beta=torch.stack([p['beta'] for p in d['pairs']]);Qs=torch.stack([q for p in d['pairs'] for q in p['Qs']])
 reads=torch.stack([(z@q*z).sum(-1) for q in Qs],-1);carry=h@A-reads[:,::2];truth=components(reads,carry,s,alpha,beta)
 fitmask=torch.zeros(len(z),dtype=torch.bool);fitmask[d['indices']]=True
 assert all(bool(fitmask[i:i+64].all()) or not bool(fitmask[i:i+64].any()) for i in range(0,len(z),64))
 groups={'fitted':fitmask,'other_historical':~fitmask}
 groups.update({f'chunk_{i:02}':torch.arange(len(z))//64==i for i in range(32)})
 rows=[]; hashes={}; generated_values={}
 for name,file in [('original','FRONTIER_FRESH_GRAPH_V1.pt'),('paired','INTERCHANGE_READ_PAIRED_V2.pt'),('cartesian','INTERCHANGE_READ_CARTESIAN_V2.pt'),('covariance_baseline','FRONTIER_FRESH_BASELINE_CALIBRATION_SHAPED_V1.pt'),('isotropic_baseline','FRONTIER_FRESH_BASELINE_NATIVE_ISOTROPIC_V1.pt')]:
  hashes[file]=hashlib.sha256((P/file).read_bytes()).hexdigest();p=torch.load(P/file,weights_only=True)
  q=source_reads(z,p) if 'baseline' not in name else torch.cat([baseline_reads(z,p[str(j)]) for j in range(3)],-1)
  values={'generated':components(q,carry,s,alpha,beta),'boundary':((h@A-.5*q[:,::2])/s[:,None]-alpha)*(q[:,1::2]/s[:,None]-beta)}
  generated_values[name]=values['generated']
  for interface,v in values.items():
   for group,mask in groups.items():
    err=(v[mask]-truth[mask]);t=truth[mask];centered=t-t.mean(0)
    rows.append(dict(candidate=name,interface=interface,group=group,n=int(mask.sum()),relative_error=(err.norm(dim=0)/centered.norm(dim=0)).tolist(),error_rms=err.square().mean(0).sqrt().tolist(),target_centered_rms=centered.square().mean(0).sqrt().tolist()))
 scores={(r['candidate'],r['interface'],r['group']):r['relative_error'][2] for r in rows}
 comparisons=[]
 for name in ('paired','cartesian'):
  ratios={g:scores[name,'generated',g]/scores['original','generated',g] for g in groups}
  comparisons.append(dict(candidate=name,relative_to_original=ratios,other_chunks_improved=sum(v<1 for g,v in ratios.items() if g.startswith('chunk_') and not fitmask[int(g[-2:])*64]),predicted_other_error_increase_10pct=ratios['other_historical']>=1.1))
 paths=[]
 for name in ('paired','cartesian'):
  delta=generated_values[name][:,2]-generated_values['original'][:,2]
  error=generated_values['original'][:,2]-truth[:,2]
  for group in ('fitted','other_historical'):
   mask=groups[group];a=float(delta[mask].square().sum());b=float((delta[mask]*error[mask]).sum());c=float(error[mask].square().sum())
   paths.append(dict(candidate=name,group=group,error_energy_quadratic=[a,2*b,c],unconstrained_best_step=-b/a,derivative_at_original=2*b,full_step_error_ratio=((a+2*b+c)/c)**.5))
 out=dict(update_paths=paths,scope='All 32 historical cached chunks. Other chunks were not used by this specific 32-coefficient refit, but are not fresh, document-disjoint, or untouched by earlier metric/feature construction. RMS17/carry remain supplied. No fitting.',hashes=hashes,fit_chunks=(d['indices']//64).unique().tolist(),rows=rows,comparisons=comparisons)
 (P/'HISTORICAL_PREFIX_TRANSFER_V1.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'comparisons':comparisons,'aggregate':[r for r in rows if r['group'] in ('fitted','other_historical') and r['interface']=='generated']},indent=2))
if __name__=='__main__':main()
