"""Weights-only worst affine-quadratic read error on RMS-normalized input ball."""
import json,time
from pathlib import Path
import torch
from quadratic_ball_extrema import extrema,controls
from pairwise_reader_graph import expand
from local_shared_reader_graph import decode
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();control=controls();d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);radius=1152**.5
 bundles={'graph':expand(torch.load(P/'FRONTIER_FRESH_GRAPH_V1.pt',weights_only=True)),'covariance_baseline':torch.load(P/'FRONTIER_FRESH_BASELINE_CALIBRATION_SHAPED_V1.pt',weights_only=True),'isotropic_baseline':torch.load(P/'FRONTIER_FRESH_BASELINE_NATIVE_ISOTROPIC_V1.pt',weights_only=True)};rows=[]
 for j,pair in enumerate(d['pairs']):
  for parity,key in enumerate(('a','b')):
   true=pair['Qs'][parity];reference=radius**2*float(torch.linalg.eigvalsh(true).abs().max())
   for name,bundle in bundles.items():
    program=bundle[str(j)];Q=decode(program)[parity]-true;l=program[key+'_linear'];b=program[key+'_bias'];out=extrema(Q,l,b,radius)
    results={k:{kk:vv for kk,vv in out[k].items() if kk!='x'} for k in ('minimum','maximum')}
    rows.append(dict(component=j+1,read=key,candidate=name,worst_absolute_error=out['worst_absolute'],teacher_max_absolute=reference,relative_to_teacher_max=out['worst_absolute']/reference,extrema=results))
    print(j+1,key,name,rows[-1]['relative_to_teacher_max'],flush=True)
 comparisons=[]
 for j in (1,2,3):
  for key in ('a','b'):
   rr={r['candidate']:r['worst_absolute_error'] for r in rows if r['component']==j and r['read']==key}
   comparisons.append(dict(component=j,read=key,covariance_ratio=rr['graph']/rr['covariance_baseline'],isotropic_ratio=rr['graph']/rr['isotropic_baseline']))
 result=dict(rows=rows,controls=control,comparisons=comparisons,predictions=dict(pred_a_certificates=True,pred_b_within_both_baselines=all(max(c['covariance_ratio'],c['isotropic_ratio'])<=1.1 for c in comparisons)),seconds=time.monotonic()-start,scope='Worst source-read error over ||z||<=sqrt(1152). Contains RMS-normalized inputs; adversarial maximizers need not be reachable by text. Full affine corrections included. Teacher maximum is normalization, not a pointwise relative-error bound. Does not directly bound normalized composed component or endpoint errors.')
 (P/'NATIVE_WORST_READ_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['predictions']))
if __name__=='__main__':main()
