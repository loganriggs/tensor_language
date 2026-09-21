from pathlib import Path
import json,torch
from overlap_varpro import OverlapMetric,parameters_from_program
from overlap_export import export_overlap,assess_overlap
P=Path(__file__).parent;torch.set_num_threads(2)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);allprograms=torch.load(P/'LOCAL_SHARED_READER_PROGRAMS_V1.pt',weights_only=True);meta=json.loads((P/'LOCAL_SHARED_READER_V1.json').read_text());Q=torch.stack([q for pair in d['pairs'] for q in pair['Qs']]);S=torch.linalg.inv(d['inverse_root']);I=torch.eye(1152,dtype=Q.dtype);records=[]
for name,transform,inverse in [('calibration_shaped',S,d['inverse_root']),('native_isotropic',I,I)]:
 program=allprograms[name+'_128_160'];templates=[program['pairs'][str(j)] for j in range(3)];params=[p.requires_grad_() for p in parameters_from_program(program,transform)];metric=OverlapMetric(transform@Q@transform,templates);loss,W,_=metric.loss(params);loss.backward();grads=[float(p.grad.norm()) for p in params];assert all(torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in params)
 with torch.no_grad():
  replay=abs(float(loss.detach())-float(metric.loss(params,dense=True)[0]));assert replay<1e-8
  exported=export_overlap(params,W,program,inverse,d);scores=assess_overlap(exported,d);assert scores['stored_floats']==996876 and scores['source_total_multiplications']==986496
  previous=next(r for r in meta['records'] if r['key']==name+'_128_160');metric_key='covariance_error' if name=='calibration_shaped' else 'native_error';coefficient_replay=abs(scores[metric_key]-previous[metric_key]);assert coefficient_replay<1e-7
  records.append(dict(metric=name,dense_loss_replay=replay,nonzero_gradient_norms=grads,initial_coefficient_replay=coefficient_replay,export=scores))
(P/'OVERLAP_NATIVE_PREFLIGHT_V1.json').write_text(json.dumps(dict(records=records),indent=2)+'\n');print('PASS both native-width losses, all shared/private gradients, coordinate exports, source/component execution and exact cost')
